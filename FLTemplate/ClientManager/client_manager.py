import random
import threading
from abc import ABC, abstractmethod
from logging import INFO
from typing import Dict, List, Optional

import dill
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from Utils.preferences import Preferences


class SimpleClientManager(ClientManager):
    """Provides a pool of available clients."""

    def __init__(self, preferences: Preferences = None) -> None:
        """Creates a SimpleClientManager.

        Parameters:
        -----------
        preferences : Preferences
            Preferences object containing the configuration for the client manager.
        """

        self.clients: dict[str, ClientProxy] = {}
        self._cv = threading.Condition()
        self.preferences = preferences
        self.clients_list: List[str] = []
        self.num_round_train = 0
        self.num_round_validation = 0
        self.num_round_test = 0
        self.training_clients_list: List[str] = []
        self.validation_clients_list: List[str] = []
        self.test_clients_list: List[str] = []

    def __len__(self) -> int:
        """Return the number of available clients.

        Returns
        -------
        num_available : int
            The number of currently available clients.
        """
        return len(self.clients)

    def num_available(self, phase) -> int:
        """Return the number of available clients.

        Returns
        -------
        num_available : int
            The number of currently available clients.
        """
        if phase == "training":
            return len(self.training_clients_list)
        elif phase == "validation":
            return len(self.validation_clients_list)
        else:
            return len(self.test_clients_list)

    def wait_for(self, num_clients: int, timeout: int = 86400) -> bool:
        """Wait until at least `num_clients` are available.

        Blocks until the requested number of clients is available or until a
        timeout is reached. Current timeout default: 1 day.

        Parameters
        ----------
        num_clients : int
            The number of clients to wait for.
        timeout : int
            The time in seconds to wait for, defaults to 86400 (24h).

        Returns
        -------
        success : bool
        """
        with self._cv:
            return self._cv.wait_for(lambda: len(self.clients) >= num_clients, timeout=timeout)

    def pre_sample_clients(self, fraction, client_list):
        sampled_nodes = {}
        nodes_to_sample = int(fraction * len(client_list))
        for fl_round in range(self.preferences.num_rounds):
            if fraction == 1.0:
                start = 0
                end = len(client_list)
            else:
                start = fl_round * nodes_to_sample % len(client_list)
                end = (fl_round * nodes_to_sample + nodes_to_sample) % len(client_list)

            sampled_nodes[fl_round] = client_list[start:end]
        return sampled_nodes

    def sample_clients_per_round(self, fraction: float, client_list: List[str]) -> Dict[int, List[str]]:
        """Sample clients for each round.

        This method samples clients for each round based on the preferences set in the
        SimpleClientManager. It returns a dictionary where the keys are the round numbers
        and the values are lists of client IDs.
        """
        sampled_nodes = {}
        nodes_to_sample = int(fraction * len(client_list))
        for fl_round in range(self.preferences.num_rounds):
            # number of nodes we have to select in each round
            if fraction == 1.0:
                start = 0
                end = len(client_list)
            else:
                start = fl_round * nodes_to_sample % len(client_list)
                end = (fl_round * nodes_to_sample + nodes_to_sample) % len(client_list)
            print(f"Round {fl_round}: Sampling nodes from {start} to {end}")
            if start < end:
                sampled_nodes[fl_round] = client_list[start:end]
            else:
                sampled_nodes[fl_round] = client_list[start:] + client_list[:end]

        return sampled_nodes

    def register(self, client: ClientProxy) -> bool:
        """Register Flower ClientProxy instance.

        Parameters
        ----------
        client : flwr.server.client_proxy.ClientProxy

        Returns
        -------
        success : bool
            Indicating if registration was successful. False if ClientProxy is
            already registered or can not be registered for any reason.
        """
        if client.cid in self.clients:
            return False

        self.clients[client.cid] = client
        self.clients_list.append(client.cid)

        if self.preferences.num_clients == len(self.clients_list):
            if self.preferences.cross_device:
                # In the cross device case, we want to sample the clients so that we have
                # a training set of clients and a test set of clients.
                # The test set of clients should always be the same both during the training and the testing.
                # If we are doing an hyperparameter search, we want to sample the clients
                # so that we have a training set of clients, a validation set of clients and
                # a test set of clients.
                self.validation_clients_list = None
                random.seed(self.preferences.seed)
                self.clients_list = [
                    str(client_id) for client_id in sorted([int(client_id) for client_id in self.clients_list])
                ]
                print("Clients list: ", self.clients_list)

                # sample the test clients from the self.clients_list
                random.shuffle(self.clients_list)
                self.test_clients_list = self.clients_list[: self.preferences.num_test_nodes]
                print("Nodes to sample: ", self.preferences.sampled_test_nodes_per_round)
                sampled_nodes_test = self.sample_clients_per_round(
                    fraction=self.preferences.sampled_test_nodes_per_round,
                    client_list=self.test_clients_list,
                )
                print("Test Nodes: ", sampled_nodes_test)

                with open(f"{self.preferences.fed_dir}/test_nodes_per_round.pkl", "wb") as f:
                    dill.dump(sampled_nodes_test, f)

                with open(f"{self.preferences.fed_dir}/test_nodes_list.pkl", "wb") as f:
                    dill.dump(self.test_clients_list, f)

                remaining_nodes = self.clients_list[self.preferences.num_test_nodes :]

                random.seed(self.preferences.node_shuffle_seed)
                random.shuffle(remaining_nodes)

                # Now we check if we need to create the validation set
                if self.preferences.sweep and self.preferences.num_validation_nodes > 0:
                    self.validation_clients_list = remaining_nodes[: self.preferences.num_validation_nodes]
                    remaining_nodes = remaining_nodes[self.preferences.num_validation_nodes :]
                    sampled_nodes_validation = self.sample_clients_per_round(
                        fraction=self.preferences.sampled_validation_nodes_per_round,
                        client_list=self.validation_clients_list,
                    )
                    with open(f"{self.preferences.fed_dir}/validation_nodes_per_round.pkl", "wb") as f:
                        dill.dump(sampled_nodes_validation, f)

                    with open(f"{self.preferences.fed_dir}/validation_nodes_list.pkl", "wb") as f:
                        dill.dump(self.validation_clients_list, f)

                self.training_clients_list = remaining_nodes

                sampled_nodes_train = self.sample_clients_per_round(
                    fraction=self.preferences.sampled_training_nodes_per_round,
                    client_list=self.training_clients_list,
                )
                with open(f"{self.preferences.fed_dir}/train_nodes_per_round.pkl", "wb") as f:
                    dill.dump(sampled_nodes_train, f)

                with open(f"{self.preferences.fed_dir}/train_nodes_list.pkl", "wb") as f:
                    dill.dump(self.training_clients_list, f)

                counter_sampling = {}
                for sample_list in sampled_nodes_train.values():
                    for node in sample_list:
                        if node not in counter_sampling:
                            counter_sampling[str(node)] = 0
                        counter_sampling[str(node)] += 1

                with open(f"{self.preferences.fed_dir}/counter_sampling.pkl", "wb") as f:
                    dill.dump(counter_sampling, f)

                print("Train nodes: ", self.training_clients_list)
                print("Validation nodes: ", self.validation_clients_list)
                print("Test nodes: ", self.test_clients_list)
            else:
                random.seed(self.preferences.node_shuffle_seed)
                random.shuffle(self.clients_list)

                print("Clients list: ", self.clients_list)
                # In this case I'm in the cross-silo case
                # This means that each node has training, validation and test data
                # so each node could be used for training, validation and testing
                if self.preferences.sampled_validation_nodes_per_round:
                    print("Sampling validation nodes per round: ", self.preferences.sampled_validation_nodes_per_round)
                    sampled_nodes_validation = self.pre_sample_clients(
                        fraction=self.preferences.sampled_validation_nodes_per_round,
                        client_list=self.clients_list,
                    )
                    with open(f"{self.preferences.fed_dir}/validation_nodes_per_round.pkl", "wb") as f:
                        dill.dump(sampled_nodes_validation, f)

                        print("Validation nodes: ", sampled_nodes_validation)
                else:
                    print("No validation nodes sampled, using all clients for training and testing.")

                sampled_nodes_test = self.pre_sample_clients(
                    fraction=self.preferences.sampled_test_nodes_per_round,
                    client_list=self.clients_list,
                )

                with open(f"{self.preferences.fed_dir}/test_nodes_per_round.pkl", "wb") as f:
                    dill.dump(sampled_nodes_test, f)
                    print("Test nodes: ", sampled_nodes_test)

                sampled_nodes_train = self.pre_sample_clients(
                    fraction=self.preferences.sampled_training_nodes_per_round,
                    client_list=self.clients_list,
                )
                with open(f"{self.preferences.fed_dir}/train_nodes_per_round.pkl", "wb") as f:
                    dill.dump(sampled_nodes_train, f)

                print("Train nodes: ", sampled_nodes_train)

                counter_sampling = {}
                for sample_list in sampled_nodes_train.values():
                    for node in sample_list:
                        if node not in counter_sampling:
                            counter_sampling[str(node)] = 0
                        counter_sampling[str(node)] += 1

                with open(f"{self.preferences.fed_dir}/counter_sampling.pkl", "wb") as f:
                    dill.dump(counter_sampling, f)

                random.seed(self.preferences.seed)

        with self._cv:
            self._cv.notify_all()

        return True

    def unregister(self, client: ClientProxy) -> None:
        """Unregister Flower ClientProxy instance.

        This method is idempotent.

        Parameters
        ----------
        client : flwr.server.client_proxy.ClientProxy
        """
        if client.cid in self.clients:
            del self.clients[client.cid]

            with self._cv:
                self._cv.notify_all()

    def all(self) -> dict[str, ClientProxy]:
        """Return all available clients."""
        return self.clients

    def sample(
        self,
        num_clients: int,
        phase: str,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion

        if phase == "training":
            with open(f"{self.preferences.fed_dir}/train_nodes_per_round.pkl", "rb") as f:
                train_nodes = dill.load(f)

            sampled_clients = [self.clients[str(node)] for node in train_nodes[0]]
            self.num_round_train += 1

            print(
                f"===>>>> Sampled for training round {self.num_round_train}: ",
                [client.cid for client in sampled_clients],
            )
        elif phase == "validation":
            with open(f"{self.preferences.fed_dir}/validation_nodes_per_round.pkl", "rb") as f:
                validation_nodes = dill.load(f)

            sampled_clients = [self.clients[str(node)] for node in validation_nodes[self.num_round_validation]]
            self.num_round_validation += 1
            print(
                "===>>>> Sampled for validation: ",
                [client.cid for client in sampled_clients],
            )
        else:
            with open(f"{self.preferences.fed_dir}/test_nodes_per_round.pkl", "rb") as f:
                test_nodes = dill.load(f)

            sampled_clients = [self.clients[str(node)] for node in test_nodes[self.num_round_test]]
            self.num_round_test += 1

            print("===>>>> Sampled for test: ", [client.cid for client in sampled_clients])
        return sampled_clients
