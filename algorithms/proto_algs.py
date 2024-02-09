import logging
from typing import Union, Tuple, Dict

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data

from algorithms.base_classes import PrototypeWrapper
from explanations.utils import protomf_post_val_light
from train.utils import general_weight_init


def compute_shifted_cosine_sim(x: torch.Tensor, y: torch.Tensor):
    """
    Computes the shifted cosine similarity between two tensors.
    x and y have the same last dimension.
    """
    x_norm = F.normalize(x)
    y_norm = F.normalize(y)

    sim_mtx = (1 + x_norm @ y_norm.T)
    sim_mtx = torch.clamp(sim_mtx, min=0., max=2.)

    return sim_mtx


class UProtoMF(PrototypeWrapper):
    """
    Implements the ProtoMF model with user prototypes as defined in https://dl.acm.org/doi/abs/10.1145/3523227.3546756
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 100, n_prototypes: int = 20,
                 sim_proto_weight: float = 1., sim_batch_weight: float = 1.):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_prototypes = n_prototypes
        self.sim_proto_weight = sim_proto_weight
        self.sim_batch_weight = sim_batch_weight

        self.user_embed = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embed = nn.Embedding(self.n_items, self.n_prototypes)

        self.prototypes = nn.Parameter(torch.randn([self.n_prototypes, self.embedding_dim]) * .1 / self.embedding_dim,
                                       requires_grad=True)

        self.user_embed.apply(general_weight_init)
        self.item_embed.apply(general_weight_init)

        self._acc_r_proto = 0
        self._acc_r_batch = 0

        self.name = 'UProtoMF'

        logging.info(f'Built {self.name} model \n'
                     f'- n_users: {self.n_users} \n'
                     f'- n_items: {self.n_items} \n'
                     f'- embedding_dim: {self.embedding_dim} \n'
                     f'- n_prototypes: {self.n_prototypes} \n'
                     f'- sim_proto_weight: {self.sim_proto_weight} \n'
                     f'- sim_batch_weight: {self.sim_batch_weight} \n')

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        u_repr = self.get_user_representations(u_idxs)
        i_repr = self.get_item_representations(i_idxs)

        dots = self.combine_user_item_representations(u_repr, i_repr)

        self.compute_reg_losses(u_repr)

        return dots

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        u_embed = self.user_embed(u_idxs)
        sim_mtx = compute_shifted_cosine_sim(u_embed, self.prototypes)  # [batch_size, n_prototypes]

        return sim_mtx

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return self.item_embed(i_idxs)

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        dots = (u_repr.unsqueeze(-2) * i_repr).sum(dim=-1)
        return dots

    def compute_reg_losses(self, sim_mtx):
        # Compute regularization losses
        sim_mtx = sim_mtx.reshape(-1, self.n_prototypes)
        dis_mtx = (2 - sim_mtx)  # Equivalent to maximizing the similarity.
        self._acc_r_proto += dis_mtx.min(dim=0).values.mean()
        self._acc_r_batch += dis_mtx.min(dim=1).values.mean()

    def get_and_reset_other_loss(self) -> Dict:
        acc_r_proto, acc_r_batch = self._acc_r_proto, self._acc_r_batch
        self._acc_r_proto = self._acc_r_batch = 0
        proto_loss = self.sim_proto_weight * acc_r_proto
        batch_loss = self.sim_batch_weight * acc_r_batch
        return {
            'reg_loss': proto_loss + batch_loss,
            'proto_loss': proto_loss,
            'batch_loss': batch_loss
        }

    def get_user_representations_pre_tune(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        u_repr = self.get_user_representations(u_idxs)
        return u_repr

    def get_user_representations_post_tune(self, u_repr: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return u_repr

    @staticmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        return UProtoMF(dataset.n_users, dataset.n_items, conf['embedding_dim'], conf['n_prototypes'],
                        conf['sim_proto_weight'], conf['sim_batch_weight'])

    def post_val(self, curr_epoch: int):
        return protomf_post_val_light(
            self.prototypes,
            self.user_embed.weight,
            compute_shifted_cosine_sim,
            lambda x: 2 - x,
            "Users",
            curr_epoch)


class IProtoMF(PrototypeWrapper):
    """
    Implements the ProtoMF model with item prototypes as defined in https://dl.acm.org/doi/abs/10.1145/3523227.3546756
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 100, n_prototypes: int = 20,
                 sim_proto_weight: float = 1., sim_batch_weight: float = 1.):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_prototypes = n_prototypes
        self.sim_proto_weight = sim_proto_weight
        self.sim_batch_weight = sim_batch_weight

        self.user_embed = nn.Embedding(self.n_users, self.n_prototypes)
        self.item_embed = nn.Embedding(self.n_items, self.embedding_dim)

        self.prototypes = nn.Parameter(torch.randn([self.n_prototypes, self.embedding_dim]) * .1 / self.embedding_dim,
                                       requires_grad=True)

        self.user_embed.apply(general_weight_init)
        self.item_embed.apply(general_weight_init)

        self._acc_r_proto = 0
        self._acc_r_batch = 0

        self.name = 'IProtoMF'

        logging.info(f'Built {self.name} model \n'
                     f'- n_users: {self.n_users} \n'
                     f'- n_items: {self.n_items} \n'
                     f'- embedding_dim: {self.embedding_dim} \n'
                     f'- n_prototypes: {self.n_prototypes} \n'
                     f'- sim_proto_weight: {self.sim_proto_weight} \n'
                     f'- sim_batch_weight: {self.sim_batch_weight} \n')

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        u_repr = self.get_user_representations(u_idxs)
        i_repr = self.get_item_representations(i_idxs)

        dots = self.combine_user_item_representations(u_repr, i_repr)

        self.compute_reg_losses(i_repr)

        return dots

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return self.user_embed(u_idxs)

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        i_embed = self.item_embed(i_idxs)
        i_embed = i_embed.reshape(-1, i_embed.shape[-1])
        sim_mtx = compute_shifted_cosine_sim(i_embed, self.prototypes)
        sim_mtx = sim_mtx.reshape(list(i_idxs.shape) + [self.n_prototypes])

        return sim_mtx

    def get_item_representations_pre_tune(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        i_repr = self.get_item_representations(i_idxs)
        return i_repr

    def get_item_representations_post_tune(self, i_repr: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return i_repr

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        dots = (u_repr.unsqueeze(-2) * i_repr).sum(dim=-1)
        return dots

    def compute_reg_losses(self, sim_mtx):
        # Compute regularization losses
        sim_mtx = sim_mtx.reshape(-1, self.n_prototypes)
        dis_mtx = (2 - sim_mtx)  # Equivalent to maximizing the similarity.
        self._acc_r_proto += dis_mtx.min(dim=0).values.mean()
        self._acc_r_batch += dis_mtx.min(dim=1).values.mean()

    def get_and_reset_other_loss(self) -> Dict:
        acc_r_proto, acc_r_batch = self._acc_r_proto, self._acc_r_batch
        self._acc_r_proto = self._acc_r_batch = 0
        proto_loss = self.sim_proto_weight * acc_r_proto
        batch_loss = self.sim_batch_weight * acc_r_batch
        return {
            'reg_loss': proto_loss + batch_loss,
            'proto_loss': proto_loss,
            'batch_loss': batch_loss
        }

    @staticmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        return IProtoMF(dataset.n_users, dataset.n_items, conf['embedding_dim'], conf['n_prototypes'],
                        conf['sim_proto_weight'], conf['sim_batch_weight'])

    def post_val(self, curr_epoch: int):
        return protomf_post_val_light(
            self.prototypes,
            self.item_embed.weight,
            compute_shifted_cosine_sim,
            lambda x: 2 - x,
            "Items",
            curr_epoch)


class UIProtoMF(PrototypeWrapper):
    """
    Implements the ProtoMF model with item and user prototypes as defined in https://dl.acm.org/doi/abs/10.1145/3523227.3546756
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 100, u_n_prototypes: int = 20,
                 i_n_prototypes: int = 20, u_sim_proto_weight: float = 1., u_sim_batch_weight: float = 1.,
                 i_sim_proto_weight: float = 1., i_sim_batch_weight: float = 1.):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        self.uprotomf = UProtoMF(n_users, n_items, embedding_dim, u_n_prototypes,
                                 u_sim_proto_weight, u_sim_batch_weight)

        self.iprotomf = IProtoMF(n_users, n_items, embedding_dim, i_n_prototypes,
                                 i_sim_proto_weight, i_sim_batch_weight)

        self.u_to_i_proj = nn.Linear(self.embedding_dim, i_n_prototypes, bias=False)  # UProtoMF -> IProtoMF
        self.i_to_u_proj = nn.Linear(self.embedding_dim, u_n_prototypes, bias=False)  # IProtoMF -> UProtoMF

        self.u_to_i_proj.apply(general_weight_init)
        self.i_to_u_proj.apply(general_weight_init)

        # Deleting unused parameters

        del self.uprotomf.item_embed
        del self.iprotomf.user_embed

        self.name = 'UIProtoMF'

        logging.info(f'Built {self.name} model \n')

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        u_sim_mtx = self.uprotomf.get_user_representations(u_idxs)
        u_proj = self.u_to_i_proj(self.uprotomf.user_embed(u_idxs))

        return u_sim_mtx, u_proj

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        i_sim_mtx = self.iprotomf.get_item_representations(i_idxs)
        i_proj = self.i_to_u_proj(self.iprotomf.item_embed(i_idxs))

        return i_sim_mtx, i_proj

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        u_sim_mtx, u_proj = u_repr
        i_sim_mtx, i_proj = i_repr

        u_dots = (u_sim_mtx.unsqueeze(-2) * i_proj).sum(dim=-1)
        i_dots = (u_proj.unsqueeze(-2) * i_sim_mtx).sum(dim=-1)
        dots = u_dots + i_dots
        return dots

    def get_item_representations_pre_tune(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        i_repr = self.get_item_representations(i_idxs)
        return i_repr

    def get_item_representations_post_tune(self, i_repr: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return i_repr

    def get_user_representations_pre_tune(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        u_repr = self.get_user_representations(u_idxs)
        return u_repr

    def get_user_representations_post_tune(self, u_repr: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return u_repr

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        u_repr = self.get_user_representations(u_idxs)
        i_repr = self.get_item_representations(i_idxs)

        dots = self.combine_user_item_representations(u_repr, i_repr)

        u_sim_mtx, _ = u_repr
        i_sim_mtx, _ = i_repr
        self.uprotomf.compute_reg_losses(u_sim_mtx)
        self.iprotomf.compute_reg_losses(i_sim_mtx)

        return dots

    def get_and_reset_other_loss(self) -> Dict:
        u_reg = {'user_' + k: v for k, v in self.uprotomf.get_and_reset_other_loss().items()}
        i_reg = {'item_' + k: v for k, v in self.iprotomf.get_and_reset_other_loss().items()}
        return {
            'reg_loss': u_reg.pop('user_reg_loss') + i_reg.pop('item_reg_loss'),
            **u_reg,
            **i_reg
        }

    def post_val(self, curr_epoch: int):
        uprotomf_post_val = {'user_' + k: v for k, v in self.uprotomf.post_val(curr_epoch).items()}
        iprotomf_post_val = {'item_' + k: v for k, v in self.iprotomf.post_val(curr_epoch).items()}
        return {**uprotomf_post_val, **iprotomf_post_val}

    @staticmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        return UIProtoMF(dataset.n_users, dataset.n_items, conf['embedding_dim'], conf['u_n_prototypes'],
                         conf['i_n_prototypes'], conf['u_sim_proto_weight'], conf['u_sim_batch_weight'],
                         conf['i_sim_proto_weight'], conf['i_sim_batch_weight'])
