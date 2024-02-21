import os

import torch

import wandb
from fair.mod_weights import AddModularWeights
from train_adversarial import train_adversarial
from train_probe import train_probe
from utilities.utils import generate_id

WANDB_PROJECT = 'fair_rec'
WANDB_ENTITY = 'karapost'

id = generate_id()
adv_config = {
    'best_run_sweep_id': 'sshfyfwu',
    'group_type': 'gender',
    'delta_on': 'users',

    'neural_layers_config': [128],
    'use_clamping': True,

    'lr': 1e-5,
    'wd': 1e-5,
    'eta_min': 1e-6,

    'lam_adv': 2.,
    'gradient_scaling': 1.,
    'init_std': .1,

    'n_epochs': 25,
    'train_batch_size': 512,
    'eval_batch_size': 8,

    'device': 'cuda',
    'seed': 59,
    'verbose': True,
    'running_settings': {'eval_n_workers': 3, 'train_n_workers': 8},
    'save_path': f'./saved_models/baseline/{id}/',

}

wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, config=adv_config, name=f'baseline_{id}', )
n_delta_sets, user_to_delta_set = train_adversarial(adv_config)

probe_config = {
    **adv_config,
    'neural_layers_config': [128],
    'n_epochs': 25,
    'lr': 5e-4,
    'wd': 1e-5
}

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
