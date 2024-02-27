import os

import torch

import wandb
from conf.probe.probe_configs import probe_configs
from fair.utils import get_mod_weights_module
from train_adversarial import train_adversarial
from train_mmd import train_mmd
from train_probe import train_probe


def train_val_agent():
    # Initialization and gathering hyperparameters
    run = wandb.init(job_type='train/val')
    run_id = run.id
    sweep_id = run.sweep_id

    debias_conf = {k: v for k, v in wandb.config.items() if k[0] != '_'}

    save_path = './saved_models/{}/{}/'.format(sweep_id, run_id)
    debias_conf['save_path'] = save_path

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
    # Probe is tested on the same seed of the debiasing method
    # Refer to the ./conf/probe/probe_configs.py file for the configuration
    probe_config = probe_configs[debias_conf['dataset']][debias_conf['group_type']]

    # Additional options
    probe_config = {
        **probe_config,
        # --- Others --- #
        'device': 'cuda',
        'seed': debias_conf['seed'],
        'verbose': True,
        'running_settings': {'eval_n_workers': 2, 'train_n_workers': 6},
    }

    # Modular Weights
    mod_weights = get_mod_weights_module(
        how_use_deltas=debias_conf['how_use_deltas'],
        latent_dim=debias_conf['latent_dim'],
        n_delta_sets=n_delta_sets,
        user_to_delta_set=user_to_delta_set,
        use_clamping=debias_conf['use_clamping']
    )

    mod_weights_state_dict = torch.load(
        os.path.join(debias_conf['save_path'], 'last.pth'), map_location=probe_config['device']
    )['mod_weights']
    mod_weights.load_state_dict(mod_weights_state_dict)
    mod_weights.requires_grad_(False)

    train_probe(
        probe_config=probe_config,
        eval_type='test',
        wandb_log_prefix=f'final_',
        mod_weights=mod_weights
    )


train_val_agent()
