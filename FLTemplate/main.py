import argparse
import signal
import sys

from Client.client import FlowerClient
from ClientManager.client_manager import SimpleClientManager
from flwr.client import ClientApp
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets.visualization import plot_label_distributions
from Models.simple_cnn import Net
from Server.server import Server
from Utils.dataset import get_mnist_dataloaders
from Utils.utils import get_params, weighted_average


def signal_handler(sig, frame):
    print("Gracefully stopping your experiment! Keep calm!")
    global wandb_run
    if wandb_run:
        wandb_run.finish()
    sys.exit(0)


def client_fn(context: Context):
    """Returns a FlowerClient containing its data partition."""

    partition_id = int(context.node_config["partition-id"])
    partition = fds.load_partition(partition_id, "train")
    # partition into train/validation
    partition_train_val = partition.train_test_split(test_size=0.1, seed=42)

    # Let's use the function defined earlier to construct the dataloaders
    # and apply the dataset transformations
    trainloader, testloader = get_mnist_dataloaders(partition_train_val, batch_size=32)

    return FlowerClient(trainloader=trainloader, valloader=testloader).to_client()


def server_fn(context: Context):
    # instantiate the model
    model = Net(num_classes=10)
    ndarrays = get_params(model)
    # Convert model parameters to flwr.common.Parameters
    global_model_init = ndarrays_to_parameters(ndarrays)

    # Define the strategy
    strategy = FedAvg(
        fraction_fit=0.1,  # 10% clients sampled each round to do fit()
        fraction_evaluate=0.5,  # 50% clients sample each round to do evaluate()
        evaluate_metrics_aggregation_fn=weighted_average,  # callback defined earlier
        initial_parameters=global_model_init,  # initialised global model
    )

    config = ServerConfig(num_rounds=num_rounds)
    server = Server(client_manager=client_manager, strategy=strategy)

    # Wrap everything into a `ServerAppComponents` object
    return ServerAppComponents(server=server, config=config)


def prepare_data():
    partitioner = IidPartitioner(num_partitions=num_clients)
    # Let's partition the "train" split of the MNIST dataset
    # The MNIST dataset will be downloaded if it hasn't been already
    fds = FederatedDataset(dataset="ylecun/mnist", partitioners={"train": partitioner})

    fds.load_partition(0)

    fig, ax, df = plot_label_distributions(
        partitioner,
        label_name="label",
        plot_type="bar",
        size_unit="absolute",
        partition_id_axis="x",
        legend=True,
        verbose_labels=True,
        max_num_partitions=30,  # Note we are only showing the first 30 so the plot remains readable
        title="Per Partition Labels Distribution",
    )
    fig.savefig("per_partition_labels_distribution.png", bbox_inches="tight")
    return partitioner, fds


parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
parser.add_argument("--num_clients", type=int, default=None)
parser.add_argument("--num_rounds", type=int, default=None)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    # remove files in tmp/ray
    args = parser.parse_args()

    num_clients = args.num_clients
    num_rounds = args.num_rounds

    # Create your ServerApp
    client_manager = SimpleClientManager()

    partitioner, fds = prepare_data()

    # Create your ServerApp
    server_app = ServerApp(server_fn=server_fn)

    # Concstruct the ClientApp passing the client generation function
    client_app = ClientApp(client_fn=client_fn)

    run_simulation(server_app=server_app, client_app=client_app, num_supernodes=num_clients)
