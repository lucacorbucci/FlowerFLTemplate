import argparse
import signal
import sys

import wandb
from Aggregations.aggregations import Aggregation
from ClientManager.client_manager import SimpleClientManager
from datasets import load_dataset
from Datasets.dataset_utils import get_data_info, prepare_data_for_cross_device, prepare_data_for_cross_silo
from flwr.client import ClientApp
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.simulation import run_simulation
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from flwr_datasets.visualization import plot_label_distributions
from Models.utils import get_model
from Server.server import Server
from Strategy.fed_avg import FedAvg
from Utils.preferences import Preferences
from Utils.utils import get_params


def signal_handler(sig, frame):
    print("Gracefully stopping your experiment! Keep calm!")
    # global wandb_run
    # if wandb_run:
    # wandb_run.finish()
    sys.exit(0)


def client_fn(context: Context):
    """Returns a FlowerClient containing its data partition."""
    partition_id = int(context.node_config["partition-id"])
    partition = partitioner.load_partition(partition_id)

    if preferences.cross_device:
        return prepare_data_for_cross_device(context, partition, preferences)

    return prepare_data_for_cross_silo(context, partition, preferences)

def server_fn(context: Context):
    # instantiate the model
    model = get_model(dataset=preferences.dataset_name)
    ndarrays = get_params(model)
    # Convert model parameters to flwr.common.Parameters
    global_model_init = ndarrays_to_parameters(ndarrays)

    # Define the strategy
    strategy = FedAvg(
        fraction_fit=preferences.sampled_training_nodes_per_round,  # 10% clients sampled each round to do fit()
        fraction_evaluate=preferences.sampled_validation_nodes_per_round if preferences.sampled_validation_nodes_per_round > 0 else preferences.sampled_test_nodes_per_round,  # 50% clients sample each round to do evaluate()
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

def get_partitioner(preferences):
    partitioner_type = preferences.partitioner_type

    match partitioner_type:
        case "iid":
            return IidPartitioner(num_partitions=preferences.num_clients)
        case "non_iid":
            return DirichletPartitioner(
                num_partitions=args.num_clients, alpha=preferences.partitioner_alpha, partition_by=preferences.partitioner_by,
            )
        case _:
            raise ValueError(f"Unsupported partitioner type: {partitioner_type}")


def prepare_data(preferences: Preferences):
    if preferences.dataset_name == "dutch":
        data_info = get_data_info(preferences)
        preferences.scaler = data_info.get("scaler", None)
        dataset_dict = load_dataset("csv", data_files=preferences.dataset_path)
    elif preferences.dataset_name == "mnist":
        data_info = get_data_info(preferences)
        dataset_dict = load_dataset(data_info["data_type"], data_dir=preferences.dataset_path)
    elif preferences.dataset_name == "abalone":
        data_info = get_data_info(preferences)
        preferences.scaler = data_info.get("scaler", None)
        dataset_dict = load_dataset("csv", data_files=preferences.dataset_path)
    else:
        raise ValueError(f"Unsupported dataset: {preferences.dataset_name}")
 
    data = dataset_dict["train"]
    if data:
        partitioner = get_partitioner(preferences)
        partitioner.dataset = data
    else:
        raise ValueError("No training data found in the dataset")

    if args.partitioner_by:
        plot, _, _ = plot_label_distributions(
                partitioner=partitioner,
                label_name=args.partitioner_by,
                plot_type="bar",
                size_unit="absolute",
                partition_id_axis="x",
                legend=True,
                verbose_labels=True,
                max_num_partitions=args.num_clients,
                title="Per Partition Labels Distribution",

            )
        plot.savefig(f"label_distribution_{args.partitioner_by}_{args.partitioner_type}.png",  bbox_inches='tight')

    return partitioner


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
parser.add_argument("--partitioner_type", type=str, default="iid")
parser.add_argument("--partitioner_alpha", type=float, default=None)
parser.add_argument("--partitioner_by", type=str, default=None)
parser.add_argument("--task", type=str, default="classification")

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-5)




if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    # remove files in tmp/ray
    args = parser.parse_args()

    num_clients = args.num_clients
    num_rounds = args.num_rounds

    cross_device = args.FL_setting == "cross_device"

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
        partitioner_type=args.partitioner_type,
        partitioner_alpha=args.partitioner_alpha,
        partitioner_by=args.partitioner_by,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer=args.optimizer,
        momentum=args.momentum,
        task=args.task,
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

    partitioner = prepare_data(preferences=preferences)

    # Create your ServerApp
    server_app = ServerApp(server_fn=server_fn)

    # Concstruct the ClientApp passing the client generation function
    client_app = ClientApp(client_fn=client_fn)

    run_simulation(server_app=server_app, client_app=client_app, num_supernodes=num_clients)

    if wandb_run:
        wandb_run.finish()
