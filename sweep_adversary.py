import os

import torch

import wandb
from fair.repr_perturb import AddRepresentationPerturb
from train_adversarial import train_adversarial
from train_probe import train_probe


def train_val_agent():
    # Initialization and gathering hyperparameters
    run = wandb.init(job_type='train/val')
    run_id = run.id
    sweep_id = run.sweep_id

    adv_config = {k: v for k, v in wandb.config.items() if k[0] != '_'}

    save_path = './saved_models/{}/{}/'.format(sweep_id, run_id)
    adv_config['save_path'] = save_path
    print("------ Adversarial Learning -----")

    n_groups, n_masks, user_idx_to_mask_idx = train_adversarial(adv_config)

    print("----- Adversarial Learning is over -----")
    print("----- Starting Probe Training -----")

    # --- Probe check with default hyperparameters --- #
    probe_config = {
        **adv_config,
        'neural_layers_config': [128],
        'n_epochs': 25,
        'lr': 5e-4,
        'wd': 1e-5,
    }

    # Modular Weights
    repr_perturb = AddRepresentationPerturb(
        repr_dim=64,
        n_masks=n_masks,
        user_idx_to_mask_idx=user_idx_to_mask_idx,
        clamp_boundaries=(0, 2)
    )
    repr_perturb_state_dict = torch.load(
        os.path.join(save_path, 'last.pth'), map_location=adv_config['device']
    )['repr_perturb']
    repr_perturb.load_state_dict(repr_perturb_state_dict)
    repr_perturb.requires_grad_(False)

    train_probe(
        probe_config=probe_config,
        eval_type='test',
        wandb_log_prefix='final_',
        repr_perturb=repr_perturb
    )

    wandb.finish()


train_val_agent()
