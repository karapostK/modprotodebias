import logging
from abc import ABC, abstractmethod

import torch
from torch import nn


class RepresentationPerturb(ABC, nn.Module):
    """
    Abstract Class used to hold the Δs. It is used to perturb the representations.
    Three cases are possible:
    - Single Δ for all the users
    - Define a Δ for each group of users
    - Define a Δ for each user
    """

    def __init__(self, repr_dim: int, n_masks: int, user_idx_to_mask_idx: torch.tensor):
        """

        :param repr_dim: dimiension of the representation
        :param n_masks: number of masks to define
        :param user_idx_to_mask_idx: how the user idxs are mapped to the masks. Shape is [n_users]
        """
        super().__init__()
        assert n_masks > 0, "n_masks should be greater than 0!"
        assert user_idx_to_mask_idx.max() == n_masks - 1, "user_idx_to_mask_idx should be in the range [0, n_masks-1]!"

        self.repr_dim = repr_dim
        self.n_masks = n_masks
        self.user_idx_to_mask_idx = user_idx_to_mask_idx

        logging.info(f'Built {self.__class__.__name__} module \n')

    @abstractmethod
    def forward(self, in_repr: torch.Tensor, user_idxs: torch.Tensor):
        """

        :param in_repr: Shape is [batch_size, repr_dim]
        :param user_idxs: Shape is [batch_size]
        :return:
        """
        pass


class AddRepresentationPerturb(RepresentationPerturb):

    def __init__(self, repr_dim: int, n_masks: int, user_idx_to_mask_idx: torch.tensor, init_std: float = .01,
                 clamp_boundaries: tuple = None):
        super().__init__(repr_dim, n_masks, user_idx_to_mask_idx)

        self.init_std = init_std
        self.clamp_boundaries = clamp_boundaries

        self.deltas = nn.Parameter(init_std * torch.randn(self.n_masks, self.repr_dim), requires_grad=True)

        logging.info(f'Built {self.__class__.__name__} module \n')

    def forward(self, in_repr: torch.Tensor, user_idxs: torch.Tensor):
        """

        :param in_repr: Shape is [batch_size, repr_dim]
        :param user_idxs: Shape is [batch_size]
        :return:
        """
        deltas = self.deltas[self.user_idx_to_mask_idx[user_idxs]]  # [batch_size, repr_dim]
        out_repr = in_repr + deltas
        if self.clamp_boundaries is not None:
            out_repr = torch.clamp(out_repr, self.clamp_boundaries[0], self.clamp_boundaries[1])
        return out_repr


class MultiplyRepresentationPerturb(RepresentationPerturb):
    def __init__(self, repr_dim: int, n_masks: int, user_idx_to_mask_idx: torch.tensor, init_std: float = .01,
                 clamp_boundaries: tuple = None):
        super().__init__(repr_dim, n_masks, user_idx_to_mask_idx)

        self.init_std = init_std
        self.clamp_boundaries = clamp_boundaries

        self.deltas = nn.Parameter(init_std * torch.randn(self.n_masks, self.repr_dim), requires_grad=True)

        logging.info(f'Built {self.__class__.__name__} module \n')

    def forward(self, in_repr: torch.Tensor, user_idxs: torch.Tensor):
        """

        :param in_repr: Shape is [batch_size, repr_dim]
        :param user_idxs: Shape is [batch_size]
        :return:
        """
        deltas = self.deltas[self.user_idx_to_mask_idx[user_idxs]]  # [batch_size, repr_dim]
        out_repr = in_repr * deltas
        if self.clamp_boundaries is not None:
            out_repr = torch.clamp(out_repr, self.clamp_boundaries[0], self.clamp_boundaries[1])
        return out_repr


class ScaleRepresentationPerturb(RepresentationPerturb):
    def __init__(self, repr_dim: int, n_masks: int, user_idx_to_mask_idx: torch.tensor, init_std: float = .01,
                 clamp_boundaries: tuple = None):
        super().__init__(repr_dim, n_masks, user_idx_to_mask_idx)

        self.init_std = init_std
        self.clamp_boundaries = clamp_boundaries  # ignored

        self.deltas = nn.Parameter(init_std * torch.randn(self.n_masks, self.repr_dim), requires_grad=True)

        logging.info(f'Built {self.__class__.__name__} module \n')

    def forward(self, in_repr: torch.Tensor, user_idxs: torch.Tensor):
        """

        :param in_repr: Shape is [batch_size, repr_dim]
        :param user_idxs: Shape is [batch_size]
        :return:
        """
        deltas = self.deltas[self.user_idx_to_mask_idx[user_idxs]]  # [batch_size, repr_dim]
        deltas = nn.Sigmoid()(deltas)
        out_repr = in_repr * deltas
        return out_repr
