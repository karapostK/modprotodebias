import os

import torch

import wandb
from conf.probe.probe_configs import probe_configs
from fair.utils import generate_run_name, get_mod_weights_module
from train_adversarial import train_adversarial
from train_mmd import train_mmd
from train_probe import train_probe
from utilities.utils import generate_id

WANDB_PROJECT = 'fair_rec'

run_id = generate_id()

debias_conf = {
    # --- General --- #
    'dataset': 'lfm2bdemobias',
    'group_type': 'gender',
    'delta_on': 'users',

    # --- Model --- #
    'inner_layers_config': [128],
    'use_clamping': False,

    # --- Training --- #
    'n_epochs': 25,

    # Debiasing
    'debiasing_method': 'adv',  # 'adv' or 'mmd'
    'how_use_deltas': 'multiply',  # 'add' or 'multiply'
    'lam': 50.,  # Strength  of the debiasing
    'init_std': 0.01,
    'gradient_scaling': 1.,  # Ignored if debiasing_method == 'mmd'

    # Learning Rates
    'lr_adv': 1e-3,  # Ignored if debiasing_method == 'mmd'
    'lr_deltas': 2e-4,
    'wd': 1e-5,
    'eta_min': 1e-6,

    # Batch Sizes
    'train_batch_size': 128,
    'eval_batch_size': 32,

    # --- Others --- #
    'device': 'cuda',
    'seed': 59,
    'verbose': True,
    'running_settings': {'eval_n_workers': 2, 'train_n_workers': 8},
    'run_id': run_id,
    'save_path': f'./saved_models/baseline/{run_id}/',

}
# Change here if you want to give your run a different name
run_name = generate_run_name(debias_conf, ['debiasing_method', 'lam', 'lr_deltas', 'seed', 'run_id'])

wandb.init(project=WANDB_PROJECT, config=debias_conf, name=run_name)

print("------ Debiasing -----")
if debias_conf['debiasing_method'] == 'adv':
    print("Using Adversarial Debiasing")
    n_delta_sets, user_to_delta_set = train_adversarial(debias_conf)
elif debias_conf['debiasing_method'] == 'mmd':
    print("Using MMD Debiasing")
    n_delta_sets, user_to_delta_set = train_mmd(debias_conf)
else:
    raise ValueError(f"Unknown debiasing method: {debias_conf['debiasing_method']}")

print("----- Debiasing is over -----")
print("----- Starting Final Attack -----")

# Refer to the ./conf/probe/probe_configs.py file for the configuration
probe_config = probe_configs[debias_conf['dataset']][debias_conf['group_type']]

# Additional options
probe_config = {
    **probe_config,
    # --- Others --- #
    'device': 'cuda',
    'seed': 59,
    'verbose': True,
    'running_settings': {'eval_n_workers': 2, 'train_n_workers': 8},
}

# Modular Weights
mod_weights = get_mod_weights_module(
    how_use_deltas=debias_conf['how_use_deltas'],
    latent_dim=64,
    n_delta_sets=n_delta_sets,
    user_to_delta_set=user_to_delta_set,
    use_clamping=debias_conf['use_clamping']
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
