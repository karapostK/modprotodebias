import os

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from tqdm import trange

import wandb
from conf.protofair_conf_parser import parse_conf
from fair.fair_eval import evaluate
from fair.mmd import MMD
from fair.utils import get_rec_model, get_mod_weights_settings, \
    get_dataloaders, get_user_group_data, get_evaluators, get_mod_weights_module
from train.rec_losses import RecSampledSoftmaxLoss
from utilities.utils import reproducible, fetch_rec_model


def train_mmd(debias_conf: dict):
    debias_conf = parse_conf(debias_conf, 'debiasing')

    rec_conf = fetch_rec_model(
        debias_conf['best_run_sweep_id'],
        './'
    )

    # --- Preparing the Rec Model, Data & Evaluators --- #

    # Data
    data_loaders = get_dataloaders({
        **rec_conf,
        **debias_conf,
    })

    user_to_user_group, n_groups, ce_weights = get_user_group_data(
        train_dataset=data_loaders['train'].dataset,
        group_type=debias_conf['group_type'],
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
        group_type=debias_conf['group_type']
    )

    # --- Setting up the Model (Probe/Adversary) --- #

    reproducible(debias_conf['seed'])

    # Modular Weights
    n_delta_sets, user_to_delta_set = get_mod_weights_settings(
        debias_conf['delta_on'],
        data_loaders['train'].dataset,
        group_type=debias_conf['group_type']
    )

    mod_weights = get_mod_weights_module(
        how_use_deltas=debias_conf['how_use_deltas'],
        latent_dim=debias_conf['latent_dim'],
        n_delta_sets=n_delta_sets,
        user_to_delta_set=user_to_delta_set,
        init_std=debias_conf['init_std'],
        use_clamping=debias_conf['use_clamping']
    )

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        [
            {
                'params': mod_weights.parameters(),
                'lr': debias_conf['lr_deltas']
            },
        ],
        weight_decay=debias_conf['wd']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=debias_conf['n_epochs'], eta_min=debias_conf['eta_min'])

    # Loss
    mmd_loss = MMD(debias_conf['mmd_default_class'])
    rec_loss = RecSampledSoftmaxLoss.build_from_conf(rec_conf, data_loaders['train'].dataset)

    # Save path
    os.makedirs(os.path.dirname(debias_conf['save_path']), exist_ok=True)
    wandb.config.update(debias_conf, allow_val_change=True)

    # --- Training the Model --- #
    user_to_user_group = user_to_user_group.to(debias_conf['device'])
    rec_model.to(debias_conf['device'])
    mod_weights.to(debias_conf['device'])

    best_recacc_value = -torch.inf
    best_recacc_epoch = -1

    wandb.watch(mod_weights, log='all')

    tqdm_epoch = trange(debias_conf['n_epochs'])
    for curr_epoch in tqdm_epoch:
        print(f"Epoch {curr_epoch}")

        avg_epoch_loss = 0
        avg_mmd_loss = 0
        avg_rec_loss = 0

        tqdm_step = tqdm(data_loaders['train'])
        for u_idxs, i_idxs, labels in tqdm_step:
            u_idxs = u_idxs.to(debias_conf['device'])
            i_idxs = i_idxs.to(debias_conf['device'])
            labels = labels.to(debias_conf['device'])

            i_repr = rec_model.get_item_representations(i_idxs)

            u_p, u_other = rec_model.get_user_representations(u_idxs)

            # Perturbing
            u_p = mod_weights(u_p, u_idxs)

            ### Rec Loss ###
            u_repr = u_p, u_other
            rec_scores = rec_model.combine_user_item_representations(u_repr, i_repr)
            rec_loss_value = rec_loss.compute_loss(rec_scores, labels)

            ### MMD ###
            mmd_loss_value = mmd_loss(u_p, user_to_user_group[u_idxs])

            ### Total Loss ###
            tot_loss = debias_conf['lam_rec'] * rec_loss_value + debias_conf['lam'] * mmd_loss_value

            avg_epoch_loss += tot_loss.item()
            avg_mmd_loss += mmd_loss_value.item()
            avg_rec_loss += rec_loss_value.item()

            tot_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Setting the description of the tqdm bar
            tqdm_step.set_description(
                "tot_loss: {:.3f} | mmd_loss: {:.3f} | rec_loss: {:.3f} ".format(
                    tot_loss.item(), mmd_loss_value.item(), rec_loss_value.item()
                ))
            tqdm_step.update()

        epoch_lrs = scheduler.get_last_lr()
        scheduler.step()

        avg_epoch_loss /= len(data_loaders['train'])
        avg_mmd_loss /= len(data_loaders['train'])
        avg_rec_loss /= len(data_loaders['train'])

        tqdm_epoch.set_description(
            "avg_tot_loss: {:.3f} | avg_mmd_loss: {:.3f} | avg_rec_loss: {:.3f}".format(
                avg_epoch_loss, avg_mmd_loss, avg_rec_loss
            )
        )
        tqdm_epoch.update()

        rec_results, _ = evaluate(
            rec_model=rec_model,
            neural_head=None,
            mod_weights=mod_weights,
            eval_loader=data_loaders['val'],
            rec_evaluator=rec_evaluator,
            fair_evaluator=fair_evaluator,
            device=debias_conf['device'],
            verbose=True
        )

        saving_dict = {
            'mod_weights': mod_weights.state_dict(),
            'epoch': curr_epoch,
            'rec_results': rec_results,
        }

        if rec_results['ndcg@10'] > best_recacc_value:
            print(f"Epoch {curr_epoch} found best value.")
            best_recacc_value = rec_results['ndcg@10']
            best_recacc_epoch = curr_epoch

            # Save
            torch.save(saving_dict, os.path.join(debias_conf['save_path'], 'best_recacc.pth'))

        if curr_epoch % 5 == 0:
            torch.save(saving_dict, os.path.join(debias_conf['save_path'], f'epoch_{curr_epoch}.pth'))

        # Save last
        torch.save(saving_dict, os.path.join(debias_conf['save_path'], 'last.pth'))

        wandb.log(
            {
                **rec_results,
                'best_recacc_value': best_recacc_value,
                'best_recacc_epoch': best_recacc_epoch,
                'avg_epoch_loss': avg_epoch_loss,
                'avg_mmd_loss': avg_mmd_loss,
                'avg_rec_loss': avg_rec_loss,
                'epoch_lr_deltas': epoch_lrs[0],
                'max_delta': mod_weights.deltas.max().item(),
                'min_delta': mod_weights.deltas.min().item(),
                'mean_delta': mod_weights.deltas.mean().item(),
            }
        )

    return n_delta_sets, user_to_delta_set
