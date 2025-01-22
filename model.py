import torch
import torch.nn as nn


class BaseNetwork(nn.Module):
    """
    Base class for a neural network to reduce redundancy.
    """

    def __init__(self, layers):
        super(BaseNetwork, self).__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DeepNetwork(BaseNetwork):
    def __init__(self):
        layers = [
            nn.Linear(1, 20), nn.ReLU(),
            nn.Linear(20, 20), nn.ReLU(),
            nn.Linear(20, 20), nn.ReLU(),
            nn.Linear(20, 20), nn.ReLU(),
            nn.Linear(20, 1)
        ]
        super(DeepNetwork, self).__init__(layers)


class ShallowNetwork(BaseNetwork):
    def __init__(self):
        layers = [
            nn.Linear(1, 1000), nn.ReLU(),
            nn.Linear(1000, 1)
        ]
        super(ShallowNetwork, self).__init__(layers)
