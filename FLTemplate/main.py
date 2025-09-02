import argparse
import os
import signal
import sys
from logging import INFO, WARNING
from typing import cast

import pandas as pd
import wandb
from Aggregations.aggregations import Aggregation
from Client.client import FlowerClient
from ClientManager.client_manager import SimpleClientManager
from datasets import Dataset, load_dataset
from Datasets.dataset_utils import prepare_data_for_cross_device, prepare_data_for_cross_silo
from Datasets.dutch import get_dutch_scaler
from Datasets.mnist import download_mnist
from flwr.client import ClientApp
from flwr.common import Context, ndarrays_to_parameters
from flwr.common.logger import log
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from Models.models import get_model
from Server.server import Server
from Strategy.fed_avg import FedAvg
from Utils.preferences import Preferences
from Utils.utils import get_params
from Datasets.dataset_utils import get_data_info

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
    elif not preferences.cross_device:
        return prepare_data_for_cross_silo(context, partition, preferences)
    else:
        raise ValueError("Unsupported FL setting")
    return None # to satisfy the type checker

def server_fn(context: Context):
    # instantiate the model
    model = get_model(dataset=preferences.dataset_name)
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



def prepare_data(preferences: Preferences):
    if preferences.dataset_name == "dutch":
        data_info = get_data_info(preferences)
        preferences.scaler = data_info.get("scaler", None)
        partitioner = IidPartitioner(num_partitions=num_clients)
    elif preferences.dataset_name == "mnist":
        data_info = get_data_info(preferences)
        dataset_dict = load_dataset(data_info["data_type"], data_dir=preferences.dataset_path)
        print(dataset_dict["train"])
        data = dataset_dict["train"]
        if data:
            partitioner = IidPartitioner(num_partitions=num_clients)
            partitioner.dataset = data
        else:
            raise ValueError("No training data found in the MNIST dataset")

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
