from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dims, activation=F.relu, output_activation=None):
        super().__init__()
        self.activation = activation
        self.output_activation = output_activation
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

    def forward(self, X):
        out = X
        for layer in self.layers[:-1]:
            out = layer(out)
            if self.activation is not None:
                out = self.activation(out)

        out = self.layers[-1](out)
        if self.output_activation is not None:
            out = self.output_activation(out)

        return out

