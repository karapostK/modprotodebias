import os

import torch

import wandb
from fair.mod_weights import AddModularWeights
from train_adversarial import train_adversarial
from train_probe import train_probe
from utilities.utils import generate_id
import random
import numpy as np

WANDB_PROJECT = 'fair_rec'
# WANDB_ENTITY = 'domainadaptors'

id = generate_id()
adv_config = {
    'best_run_sweep_id': 'sshfyfwu',
    'group_type': 'gender',
    'delta_on': 'users',

    'inner_layers_config': [128],
    'use_clamping': False,

    'lr_adv': 1e-3,
    "lr_deltas": 2e-4,
    'wd': 1e-5,
    'eta_min': 1e-6,

    'lam_adv': 50.,
    'gradient_scaling': 1.,
    'init_std': 0.01,
    "debiasing": "adv",

    'n_epochs': 25,
    'train_batch_size': 128,
    'eval_batch_size': 8,

    'device': 'cuda',
    'seed': 59,
    'verbose': True,
    'running_settings': {'eval_n_workers': 3, 'train_n_workers': 8},
    'save_path': f'./saved_models/baseline/{id}/',

}

# for sd in adv_config["seed"]:
#     torch.manual_seed(sd)
#     np.random.seed(sd)
#     random.seed(sd)
sd = adv_config["seed"]
wandb.init(project=WANDB_PROJECT, config=adv_config, name=f'{adv_config["debiasing"]}_{adv_config["lam_adv"]}_lr_{adv_config["lr_deltas"]}_{sd}_{id}', )
n_delta_sets, user_to_delta_set = train_adversarial(adv_config)

probe_config = {
    **adv_config,
    'inner_layers_config': [128],
    'n_epochs': 50,
    'lr': 5e-4,
    'wd': 1e-5,
    'eta_min': 1e-6
}
print("Probe Config:")
print(probe_config)

# Modular Weights
mod_weights = AddModularWeights(
    latent_dim=64,
    n_delta_sets=n_delta_sets,
    user_to_delta_set=user_to_delta_set,
    use_clamping=probe_config['use_clamping']
)
mod_weights_state_dict = torch.load(
    os.path.join(probe_config['save_path'], 'last.pth'), map_location=probe_config['device']
)['mod_weights']
mod_weights.load_state_dict(mod_weights_state_dict)
mod_weights.requires_grad_(False)

train_probe(
    probe_config=probe_config,
    eval_type='test',
    wandb_log_prefix=f'final_',
    mod_weights=mod_weights
)

wandb.finish()
