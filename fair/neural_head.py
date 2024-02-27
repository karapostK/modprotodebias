import typing

import torch
from torch import nn

from train.gradient_reversal import GradientReversalLayer


class NeuralHead(nn.Module):
    """
    NeuralHead is a configurable NN with ReLU activations. If gradient_scaling is set to anything than None, then it
    adds a Gradient Reverse Layer.
    """

    def __init__(self, layers_config: typing.List[int], gradient_scaling: float = None):
        """
        @param layers_config: List of integers, each integer represents the number of neurons in the layer
        @param gradient_scaling: the scaling factor that should be applied on the gradient in the backpropagation phase
        """
        super().__init__()

        self.layers_config = layers_config
        self.gradient_scaling = gradient_scaling
        self.grl_layer = None
        self.layers = []

        for i in range(1, len(layers_config)):
            self.layers.append(nn.Linear(layers_config[i - 1], layers_config[i]))
            # Add ReLU activation when it's not the last layer
            if i < len(layers_config) - 1:
                self.layers.append(nn.ReLU())

        self.layers = nn.Sequential(*self.layers)

        # Manual Init
        for i in range(0, len(self.layers), 2):
            relu_gain = nn.init.calculate_gain('relu')
            nn.init.xavier_normal_(self.layers[i].weight, relu_gain)
            nn.init.constant_(self.layers[i].bias, 0.01)

        if self.gradient_scaling is not None:
            self.grl_layer = GradientReversalLayer(grad_scaling=self.gradient_scaling)

    def forward(self, u_vect: torch.tensor):
        """
        @param u_vect: User representation Shape is [batch_size, start_neurons]
        @return:
        """
        if self.grl_layer is not None:
            u_vect = self.grl_layer(u_vect)

        return self.layers(u_vect)

    @torch.no_grad()
    def predict(self, u_vect: torch.tensor):
        out = self.layers(u_vect)
        return torch.argmax(out, dim=1)


class MultiHead(nn.Module):
    def __init__(self, n_heads: int, layers_config: typing.List[int], gradient_scaling: float = None):
        super().__init__()
        self.n_heads = n_heads
        self.heads = nn.ModuleList([NeuralHead(layers_config) for _ in range(n_heads)])
        self.layers_config = layers_config
        self.gradient_scaling = gradient_scaling
        self.grl_layer = None
        if self.gradient_scaling is not None:
            self.grl_layer = GradientReversalLayer(grad_scaling=self.gradient_scaling)

    def forward(self, u_vect: torch.tensor):
        if self.grl_layer is not None:
            u_vect = self.grl_layer(u_vect)
        return torch.stack([head(u_vect) for head in self.heads], dim=1)
