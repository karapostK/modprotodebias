import math

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import trange

import wandb
from conf.protofair_conf_parser import parse_conf
from fair.fair_eval import evaluate
from fair.mod_weights import ModularWeights
from fair.neural_head import NeuralHead
from fair.utils import generate_log_str, get_rec_model, get_dataloaders, \
    get_user_group_data, get_evaluators, summarize
from utilities.utils import reproducible
from utilities.wandb_utils import fetch_best_in_sweep


def train_probe(probe_config: dict,
                eval_type: str,
                wandb_log_prefix: str = None,
                mod_weights: ModularWeights = None
                ):
    assert eval_type in ['val', 'test'], "eval_type must be either 'val' or 'test'"

    probe_config = parse_conf(probe_config, 'probing')
    # --- Fetching the Best Run Configuration --- #
    rec_conf = fetch_best_in_sweep(
        probe_config['best_run_sweep_id'],
        good_faith=True,
    )

    # --- Preparing the Rec Model, Data & Evaluators --- #

    # Data
    data_loaders = get_dataloaders({
        **rec_conf,
        **probe_config,
    })

    # For Training the probe, we just need the user indexes (with multiplicity according to the number of interactions)
    u_idxs = torch.tensor(data_loaders['train'].dataset.iteration_matrix.row)  # Will be permuted later

    user_to_user_group, n_groups, ce_weights = get_user_group_data(
        train_dataset=data_loaders['train'].dataset,
        group_type=probe_config['group_type'],
        dataset_name=rec_conf['dataset']
    )

    # Recommender Model
    rec_model = get_rec_model(
        rec_conf=rec_conf,
        dataset=data_loaders['train'].dataset
    )

    # Evaluators
    rec_evaluator, fair_evaluator = get_evaluators(
        n_groups=n_groups,
        user_to_user_group=user_to_user_group,
        dataset_name=rec_conf['dataset'],
        group_type=probe_config['group_type']
    )

    # --- Setting up the Model (Probe/Adversary) --- #

    reproducible(probe_config['seed'])

    # Neural Head
    layers_config = [probe_config['latent_dim']] + probe_config['inner_layers_config'] + [n_groups]
    probe = NeuralHead(layers_config=layers_config)

    print()
    print('Probe Summary: ')
    summarize(probe, input_size=(10, probe_config['latent_dim']), dtypes=[torch.float])
    print()

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(probe.parameters(), lr=probe_config['lr'], weight_decay=probe_config['wd'])
    scheduler = CosineAnnealingLR(optimizer, T_max=probe_config['n_epochs'],
                                  eta_min=probe_config['eta_min'])

    # Loss
    probe_loss = nn.CrossEntropyLoss(weight=ce_weights.to(probe_config['device']))

    # --- Training the Model --- #
    u_idxs = u_idxs.to(probe_config['device'])
    user_to_user_group = user_to_user_group.to(probe_config['device'])
    rec_model.to(probe_config['device'])
    probe.to(probe_config['device'])

    if mod_weights is not None:
        print()
        print('Modular Weights Summary: ')
        summarize(mod_weights, input_size=[(10, probe_config['latent_dim']), (10,)], dtypes=[torch.float, torch.long])
        print()
        mod_weights.to(probe_config['device'])

    best_value = -torch.inf
    best_epoch = -1

    tqdm_epoch = trange(probe_config['n_epochs'])
    for curr_epoch in tqdm_epoch:
        print(f"Epoch {curr_epoch}")

        avg_epoch_loss = 0

        # Permuting the data every epoch
        u_idxs = u_idxs[torch.randperm(len(u_idxs), device=probe_config['device'])]

        for i in trange(0, len(u_idxs), probe_config['train_batch_size']):
            u_idxs_batch = u_idxs[i:i + probe_config['train_batch_size']]

            u_p, _ = rec_model.get_user_representations(u_idxs_batch)

            if mod_weights is not None:
                u_p = mod_weights(u_p, u_idxs_batch)

            probe_out = probe(u_p)
            probe_loss_value = probe_loss(probe_out, user_to_user_group[u_idxs_batch])

            avg_epoch_loss += probe_loss_value.item()

            probe_loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()

        epoch_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        avg_epoch_loss /= (math.ceil(len(u_idxs) / probe_config['train_batch_size']))

        tqdm_epoch.set_description("avg_epoch_loss: {:.3f} ".format(avg_epoch_loss))
        tqdm_epoch.update()

        rec_results, fair_results = evaluate(
            rec_model=rec_model,
            neural_head=probe,
            mod_weights=mod_weights,
            eval_loader=data_loaders[eval_type],
            rec_evaluator=rec_evaluator,
            fair_evaluator=fair_evaluator,
            device=probe_config['device'],
            verbose=True
        )

        print(f"Epoch {curr_epoch} - ", generate_log_str(fair_results, n_groups))

        if fair_results['balanced_acc'] > best_value:
            print(f"Epoch {curr_epoch} found best value.")
            best_value = fair_results['balanced_acc']
            best_epoch = curr_epoch

        log_dict = {
            **rec_results,
            **fair_results,
            'best_balanced_acc': best_value,
            'avg_epoch_loss': avg_epoch_loss,
            'best_epoch': best_epoch,
            'epoch_lr': epoch_lr
        }
        if wandb_log_prefix is not None:
            log_dict = {wandb_log_prefix + key: val for key, val in log_dict.items()}
        wandb.log(log_dict)
