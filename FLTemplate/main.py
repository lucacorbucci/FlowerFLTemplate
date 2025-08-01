import argparse
import copy
import signal
import sys

import numpy as np
import pandas as pd
import wandb
from Aggregations.aggregations import Aggregation
from Client.client import FlowerClient
from ClientManager.client_manager import SimpleClientManager
from datasets import load_dataset
from flwr.client import ClientApp
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets.visualization import plot_label_distributions
from Models.models import LinearClassificationNet, Net
from Server.server import Server
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from Strategy.fed_avg import FedAvg
from torch.utils.data import DataLoader, Dataset
from Utils.preferences import Preferences
from Utils.utils import get_params


def signal_handler(sig, frame):
    print("Gracefully stopping your experiment! Keep calm!")
    # global wandb_run
    # if wandb_run:
    # wandb_run.finish()
    sys.exit(0)


class TabularDataset(Dataset):
    def __init__(self, x, z, y):
        """
        Initialize the custom dataset with x (features), z (sensitive values), and y (targets).

        Args:
        x (list of tensors): List of input feature tensors.
        z (list): List of sensitive values.
        y (list): List of target values.
        """
        self.samples = x
        self.sensitive_features = z
        self.targets = y
        self.indexes = range(len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a single data point from the dataset.

        Args:
        idx (int): Index to retrieve the data point.

        Returns:
        sample (dict): A dictionary containing 'x', 'z', and 'y'.
        """
        x_sample = self.samples[idx]
        z_sample = self.sensitive_features[idx]
        y_sample = self.targets[idx]

        return x_sample, z_sample, y_sample


def client_fn(context: Context):
    """Returns a FlowerClient containing its data partition."""

    partition_id = int(context.node_config["partition-id"])
    partition = fds.load_partition(partition_id, "train")
    # partition into train/validation
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    if preferences.sweep:
        partition_loader_train_val = partition_train_test["train"].train_test_split(
            test_size=0.2, seed=preferences.node_shuffle_seed
        )
        train = partition_loader_train_val["train"]
        val = partition_loader_train_val["test"]

        x_train, z_train, y_train, _ = prepare_dutch(
            dutch_df=train,
            scaler=preferences.scaler,
        )

        x_val, z_val, y_val, _ = prepare_dutch(
            dutch_df=val,
            scaler=preferences.scaler,
        )

        train_dataset = TabularDataset(
            x=np.hstack((x_train, np.ones((x_train.shape[0], 1)))).astype(np.float32),
            z=z_train.astype(np.float32),
            y=y_train.astype(np.float32),
        )
        val_dataset = TabularDataset(
            x=np.hstack((x_val, np.ones((x_val.shape[0], 1)))).astype(np.float32),
            z=z_val.astype(np.float32),
            y=y_val.astype(np.float32),
        )

        trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        return FlowerClient(trainloader=trainloader, valloader=val_loader).to_client()

    train = partition_train_test["train"].to_pandas()
    test = partition_train_test["test"].to_pandas()

    x_train, z_train, y_train, _ = prepare_dutch(
        dutch_df=train,
        scaler=preferences.scaler,
    )

    x_test, z_test, y_test, _ = prepare_dutch(
        dutch_df=test,
        scaler=preferences.scaler,
    )

    train_dataset = TabularDataset(
        x=np.hstack((x_train, np.ones((x_train.shape[0], 1)))).astype(np.float32),
        z=z_train.astype(np.float32),
        y=y_train.astype(np.float32),
    )
    test_dataset = TabularDataset(
        x=np.hstack((x_test, np.ones((x_test.shape[0], 1)))).astype(np.float32),
        z=z_test.astype(np.float32),
        y=y_test.astype(np.float32),
    )

    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))

    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return FlowerClient(trainloader=trainloader, valloader=test_loader).to_client()


def server_fn(context: Context):
    # instantiate the model
    model = LinearClassificationNet(input_size=12, output_size=2)
    ndarrays = get_params(model)
    # Convert model parameters to flwr.common.Parameters
    global_model_init = ndarrays_to_parameters(ndarrays)

    # Define the strategy
    strategy = FedAvg(
        fraction_fit=0.1,  # 10% clients sampled each round to do fit()
        fraction_evaluate=0.5,  # 50% clients sample each round to do evaluate()
        initial_parameters=global_model_init,  # initialised global model
        fit_metrics_aggregation_fn=Aggregation.agg_metrics_train,
        evaluate_metrics_aggregation_fn=Aggregation.agg_metrics_evaluation,
        test_metrics_aggregation_fn=Aggregation.agg_metrics_test,
        preferences=preferences,
        wandb_run=wandb_run,
    )

    config = ServerConfig(num_rounds=num_rounds)
    server = Server(client_manager=client_manager, strategy=strategy, preferences=preferences)

    # Wrap everything into a `ServerAppComponents` object
    return ServerAppComponents(server=server, config=config)


def get_dutch_scaler(
    sweep: bool,
    seed: int,
    dutch_df: pd.DataFrame | None = None,
    validation_seed: int | None = None,
) -> MinMaxScaler:
    if dutch_df is None:
        raise ValueError("dutch_df cannot be None")

    _, _, _, scaler = prepare_dutch(
        dutch_df=dutch_df,
    )
    return scaler


def prepare_dutch(
    dutch_df: pd.DataFrame,
    scaler: MinMaxScaler | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    # check the columns with missign values:
    missing_values_columns = dutch_df.columns[dutch_df.isna().any()].tolist()
    for column in missing_values_columns:
        dutch_df[column] = dutch_df[column].fillna(dutch_df[column].mode()[0])

    if len(dutch_df.columns[dutch_df.isna().any()].tolist()) != 0:
        error_message = "There are still missing values in the dataset"
        raise ValueError(error_message)

    dutch_df["sex_binary"] = np.where(dutch_df["sex"] == 1, 1, 0)
    dutch_df["occupation_binary"] = np.where(dutch_df["occupation"] >= 300, 1, 0)

    del dutch_df["sex"]
    del dutch_df["occupation"]

    y_train = dutch_df["occupation_binary"].astype(int).values
    z_train = dutch_df["sex_binary"].astype(int).values
    del dutch_df["occupation_binary"]
    dutch_df = pd.get_dummies(dutch_df, columns=None, drop_first=False)

    if scaler is None:
        scaler = MinMaxScaler()
    x_train = scaler.fit_transform(dutch_df)

    return x_train, np.array(z_train), np.array(y_train), scaler


def get_data_info(preferences: Preferences):
    match preferences.dataset_name:
        case "dutch":
            df = pd.read_csv(preferences.dataset_path)
            scaler = get_dutch_scaler(
                sweep=preferences.sweep,
                seed=preferences.seed,
                dutch_df=df,
                validation_seed=preferences.node_shuffle_seed,
            )

            return {"data_type": "csv", "target": "occupation", "sensitive_attribute": "sex", "scaler": scaler}
        case _:
            raise ValueError(f"Unsupported dataset: {preferences.dataset_name}")


def prepare_data(preferences: Preferences):
    data_info = get_data_info(preferences)

    preferences.scaler = data_info["scaler"]

    # Build a FederatedDataset directly from the CSV using the provided partitioner
    partitioner = IidPartitioner(num_partitions=num_clients)
    fds = FederatedDataset(
        dataset=data_info["data_type"],
        partitioners={"train": partitioner},
        data_files={"train": preferences.dataset_path},
    )

    # Retrieve the partitioner instance for the train split
    # partitioner = fds.partitioners["train"]
    # fig, ax, df = plot_label_distributions(
    #     partitioner,
    #     label_name=data_info["target"],
    #     plot_type="bar",
    #     size_unit="absolute",
    #     partition_id_axis="x",
    #     legend=True,
    #     verbose_labels=True,
    #     max_num_partitions=100,  # Note we are only showing the first 30 so the plot remains readable
    #     title="Per Partition Labels Distribution",
    # )
    # fig.savefig("per_partition_labels_distribution.png", bbox_inches="tight")
    return partitioner, fds


def setup_wandb(project_name: str, run_name: str | None):
    return wandb.init(project=project_name, name=run_name) if run_name else wandb.init(project=project_name)


parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
parser.add_argument("--num_clients", type=int, default=None, required=True)
parser.add_argument("--num_rounds", type=int, default=None, required=True)
parser.add_argument("--num_test_nodes", type=int, default=None)
parser.add_argument("--num_validation_nodes", type=int, default=None)
parser.add_argument("--num_train_nodes", type=int, default=None)
parser.add_argument("--sampled_validation_nodes_per_round", type=float, default=None)
parser.add_argument("--sampled_train_nodes_per_round", type=float, default=None)
parser.add_argument("--sampled_test_nodes_per_round", type=float, default=None)
parser.add_argument("--num_epochs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--node_shuffle_seed", type=int, default=None)
parser.add_argument("--fed_dir", type=str, default=None, required=True)
parser.add_argument("--FL_setting", type=str, default=None, required=True)
parser.add_argument("--dataset_path", type=str, default=None)
parser.add_argument("--dataset_name", type=str, default=None)
parser.add_argument("--sweep", type=bool, default=False)
parser.add_argument("--wandb", type=bool, default=True)
parser.add_argument("--project_name", type=str, default="FLTemplate")
parser.add_argument("--run_name", type=str, default=None)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    # remove files in tmp/ray
    args = parser.parse_args()

    num_clients = args.num_clients
    num_rounds = args.num_rounds

    cross_device = True if args.FL_setting == "cross_device" else False

    preferences = Preferences(
        num_clients=num_clients,
        num_rounds=num_rounds,
        cross_device=cross_device,
        num_test_nodes=args.num_test_nodes,
        num_validation_nodes=args.num_validation_nodes,
        num_train_nodes=args.num_train_nodes,
        num_epochs=args.num_epochs,
        sampled_validation_nodes_per_round=args.sampled_validation_nodes_per_round,
        sampled_training_nodes_per_round=args.sampled_train_nodes_per_round,
        sampled_test_nodes_per_round=args.sampled_test_nodes_per_round,
        seed=args.seed,
        node_shuffle_seed=args.node_shuffle_seed,
        fed_dir=args.fed_dir,
        fl_setting=args.FL_setting,
        sweep=args.sweep,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
    )

    wandb_run = (
        setup_wandb(
            project_name=args.project_name,
            run_name=args.run_name,
        )
        if args.wandb
        else None
    )

    # Create your ServerApp
    client_manager = SimpleClientManager(preferences=preferences)

    partitioner, fds = prepare_data(preferences=preferences)

    # Create your ServerApp
    server_app = ServerApp(server_fn=server_fn)

    # Concstruct the ClientApp passing the client generation function
    client_app = ClientApp(client_fn=client_fn)

    run_simulation(server_app=server_app, client_app=client_app, num_supernodes=num_clients)

    if wandb_run:
        wandb_run.finish()
