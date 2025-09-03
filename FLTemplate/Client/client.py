from collections import OrderedDict
from typing import Dict, Tuple

import torch
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from Models.models import get_model
from Training.training import test, train
from Utils.preferences import Preferences
from Utils.utils import get_optimizer, get_params, set_params


class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, preferences: Preferences) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader
        self.model = get_model(dataset=preferences.dataset_name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.preferences = preferences

    def fit(self, parameters, config):
        """This method trains the model using the parameters sent by the
        server on the dataset of this client. At then end, the parameters
        of the locally trained model are communicated back to the server"""

        # copy parameters sent by the server into client's local model
        set_params(self.model, parameters)

        # Define the optimizer
        optim = get_optimizer(self.model, self.preferences)
        optim = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        # do local training (call same function as centralised setting)
        for _ in range(self.preferences.num_epochs):
            loss, accuracy = train(self.model, self.trainloader, optim, self.device)

        # return the model parameters to the server as well as extra info (number of training examples in this case)
        return get_params(self.model), len(self.trainloader), {"accuracy": accuracy, "loss": loss}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model sent by the server on this client's
        local validation set. Then return performance metrics."""

        set_params(self.model, parameters)
        # do local evaluation (call same function as centralised setting)
        loss, accuracy = test(self.model, self.valloader, self.device)
        # send statistics back to the server
        # print(f"Client - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}")
        return float(loss), len(self.valloader), {"accuracy": accuracy, "loss": loss}


