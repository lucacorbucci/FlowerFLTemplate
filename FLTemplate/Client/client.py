from typing import Dict

import torch
import torch.nn as nn
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from Models.simple_model import SimpleModel
from Models.utils import get_model

# from Training.training import test, train
from Utils.preferences import Preferences
from Utils.utils import get_optimizer, get_params, set_params


class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, preferences: Preferences) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.preferences = preferences
        trained_model = get_model(dataset=preferences.dataset_name)
        self.model = SimpleModel(
            model = trained_model,
            optimizer = get_optimizer(trained_model, self.preferences),
            criterion = nn.CrossEntropyLoss(),
            device = self.device
        )

    def fit(self, parameters, config):
        """This method trains the model using the parameters sent by the
        server on the dataset of this client. At then end, the parameters
        of the locally trained model are communicated back to the server"""

        # copy parameters sent by the server into client's local model
        set_params(self.model.model, parameters)

        # do local training (call same function as centralised setting)
        for _ in range(self.preferences.num_epochs):
            loss, accuracy = self.model.train(trainloader=self.trainloader, epochs=self.preferences.num_epochs,)

        # return the model parameters to the server as well as extra info (number of training examples in this case)
        return get_params(self.model.model), len(self.trainloader), {"accuracy": accuracy, "loss": loss}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model sent by the server on this client's
        local validation set. Then return performance metrics."""

        set_params(self.model.model, parameters)
        loss, accuracy = self.model.evaluate(testloader=self.valloader)
        return float(loss), len(self.valloader), {"accuracy": accuracy, "loss": loss}


