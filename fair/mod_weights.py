import logging
from abc import ABC, abstractmethod

import torch
from torch import nn


class ModularWeights(ABC, nn.Module):
    """
    Abstract Class used to hold the Δs. It is used to perturb the representations.
    Three cases are possible:
    - Single Δ for all the users
    - Define a Δ for each group of users
    - Define a Δ for each user
    """

    def __init__(self, latent_dim: int, n_delta_sets: int, user_to_delta_set: torch.tensor):
        """

        :param latent_dim: dimension of the representation
        :param n_delta_sets: number of delta sets to user
        :param user_to_delta_set: how the user idxs are mapped to the delta sets. Shape is [n_users]
        """
        super().__init__()
        assert n_delta_sets > 0, "n_delta_sets should be greater than 0!"
        assert user_to_delta_set.max() == n_delta_sets - 1, "user_to_delta_set should be in the range [0, n_delta_sets-1]!"

        self.latent_dim = latent_dim
        self.n_delta_sets = n_delta_sets
        self.user_to_delta_set = user_to_delta_set

        logging.info(f'Built {self.__class__.__name__} module \n')

    @abstractmethod
    def forward(self, in_repr: torch.Tensor, user_idxs: torch.Tensor):
        """

        :param in_repr: Shape is [batch_size, latent_dim]
        :param user_idxs: Shape is [batch_size]
        :return:
        """
        pass

    def to(self, device: str):
        super().to(device)
        self.user_to_delta_set = self.user_to_delta_set.to(device)
        return self


class AddModularWeights(ModularWeights):

    def __init__(self, latent_dim: int, n_delta_sets: int, user_to_delta_set: torch.tensor, init_std: float = .01,
                 clamp_boundaries: tuple = None):
        super().__init__(latent_dim, n_delta_sets, user_to_delta_set)

        self.init_std = init_std
        self.clamp_boundaries = clamp_boundaries

        self.deltas = nn.Parameter(init_std * torch.randn(self.n_delta_sets, self.latent_dim), requires_grad=True)

        logging.info(f'Built {self.__class__.__name__} module \n')

    def forward(self, in_repr: torch.Tensor, user_idxs: torch.Tensor):
        """

        :param in_repr: Shape is [batch_size, latent_dim]
        :param user_idxs: Shape is [batch_size]
        :return:
        """
        deltas = self.deltas[self.user_to_delta_set[user_idxs]]  # [batch_size, latent_dim]
        out_repr = in_repr + deltas
        if self.clamp_boundaries is not None:
            out_repr = torch.clamp(out_repr, self.clamp_boundaries[0], self.clamp_boundaries[1])
        return out_repr


class MultiplyModularWeights(ModularWeights):
    def __init__(self, latent_dim: int, n_delta_sets: int, user_to_delta_set: torch.tensor, init_std: float = .01,
                 clamp_boundaries: tuple = None):
        super().__init__(latent_dim, n_delta_sets, user_to_delta_set)

        self.init_std = init_std
        self.clamp_boundaries = clamp_boundaries

        self.deltas = nn.Parameter(init_std * torch.randn(self.n_delta_sets, self.latent_dim), requires_grad=True)

        logging.info(f'Built {self.__class__.__name__} module \n')

    def forward(self, in_repr: torch.Tensor, user_idxs: torch.Tensor):
        """

        :param in_repr: Shape is [batch_size, latent_dim]
        :param user_idxs: Shape is [batch_size]
        :return:
        """
        deltas = self.deltas[self.user_to_delta_set[user_idxs]]  # [batch_size, latent_dim]
        out_repr = in_repr * deltas
        if self.clamp_boundaries is not None:
            out_repr = torch.clamp(out_repr, self.clamp_boundaries[0], self.clamp_boundaries[1])
        return out_repr


class ScaleModularWeights(ModularWeights):
    def __init__(self, latent_dim: int, n_delta_sets: int, user_to_delta_set: torch.tensor, init_std: float = .01,
                 clamp_boundaries: tuple = None):
        super().__init__(latent_dim, n_delta_sets, user_to_delta_set)

        self.init_std = init_std
        self.clamp_boundaries = clamp_boundaries  # ignored

        self.deltas = nn.Parameter(init_std * torch.randn(self.n_delta_sets, self.latent_dim), requires_grad=True)

        logging.info(f'Built {self.__class__.__name__} module \n')

    def forward(self, in_repr: torch.Tensor, user_idxs: torch.Tensor):
        """

        :param in_repr: Shape is [batch_size, latent_dim]
        :param user_idxs: Shape is [batch_size]
        :return:
        """
        deltas = self.deltas[self.user_to_delta_set[user_idxs]]  # [batch_size, latent_dim]
        deltas = nn.Sigmoid()(deltas)
        out_repr = in_repr * deltas
        return out_repr
