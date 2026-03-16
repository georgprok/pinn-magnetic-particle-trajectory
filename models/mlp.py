import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, in_dim=1, hidden_dim=64, hidden_layers=3, out_dim=2):
        super().__init__()

        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)
