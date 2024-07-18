import os

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from tqdm import trange

import wandb
from conf.conf_parser import parse_conf
from fair.fair_eval import evaluate
from fair.neural_head import MultiHead
from fair.utils import generate_log_str, get_rec_model, get_mod_weights_settings, \
    get_dataloaders, get_user_group_data, get_evaluators, summarize, get_mod_weights_module, get_users_gradient_scaling
from train.gradient_manipulation import GradientScalingLayer
from train.rec_losses import RecSampledSoftmaxLoss
from utilities.utils import reproducible, fetch_rec_model_config


def train_adversarial(debias_conf: dict):
    debias_conf = parse_conf(debias_conf, 'debiasing')

    rec_conf = fetch_rec_model_config(debias_conf['pre_trained_model_id'])

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

    # Neural Head
    layers_config = [debias_conf['latent_dim']] + debias_conf['inner_layers_config'] + [n_groups]
    adv_head = MultiHead(
        debias_conf['adv_n_heads'],
        layers_config,
        debias_conf['gradient_scaling']
    )
    print()
    print('Adversarial Head Summary: ')
    summarize(adv_head, input_size=(10, debias_conf['latent_dim']), dtypes=[torch.float])
    print()

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

    # Gradient Scaling Layer
    user_gradient_scaling = get_users_gradient_scaling(
        data_loaders['train'].dataset,
        debias_conf['user_updates_normalization']
    )
    gs_layer = GradientScalingLayer(user_gradient_scaling)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        [
            {
                'params': mod_weights.parameters(),
                'lr': debias_conf['lr_deltas']
            },
            {
                'params': adv_head.parameters(),
                'lr': debias_conf['lr_adv']
            },
        ],
        weight_decay=debias_conf['wd']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=debias_conf['n_epochs'], eta_min=debias_conf['eta_min'])

    # Loss
    adv_loss = nn.CrossEntropyLoss(weight=ce_weights.to(debias_conf['device']))
    rec_loss = RecSampledSoftmaxLoss.build_from_conf(rec_conf, data_loaders['train'].dataset)

    # Save path
    os.makedirs(os.path.dirname(debias_conf['save_path']), exist_ok=True)
    wandb.config.update(debias_conf, allow_val_change=True)

    # --- Training the Model --- #
    user_to_user_group = user_to_user_group.to(debias_conf['device'])
    rec_model.to(debias_conf['device'])
    mod_weights.to(debias_conf['device'])
    adv_head.to(debias_conf['device'])
    gs_layer.to(debias_conf['device'])

    best_recacc_value = -torch.inf
    best_recacc_epoch = -1
    worst_bacc_value = torch.inf
    worst_bacc_epoch = -1

    wandb.watch(mod_weights, log='all')

    tqdm_epoch = trange(debias_conf['n_epochs'])
    for curr_epoch in tqdm_epoch:
        print(f"Epoch {curr_epoch}")

        avg_epoch_loss = 0
        avg_adv_loss = 0
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
            # Possibly scaling the gradients
            u_p = gs_layer(u_p, u_idxs)

            ### Rec Loss ###
            u_repr = u_p, u_other
            rec_scores = rec_model.combine_user_item_representations(u_repr, i_repr)
            rec_loss_value = rec_loss.compute_loss(rec_scores, labels)

            ### Adversarial Head ###
            adv_out = adv_head(u_p)  # Shape is [batch_size, n_heads, n_groups]
            adv_out = adv_out.reshape(-1, n_groups)
            adv_labels = torch.repeat_interleave(user_to_user_group[u_idxs], repeats=debias_conf['adv_n_heads'])
            adv_loss_value = adv_loss(adv_out, adv_labels)

            ### Total Loss ###
            tot_loss = debias_conf['lam_rec'] * rec_loss_value + debias_conf['lam'] * adv_loss_value

            avg_epoch_loss += tot_loss.item()
            avg_adv_loss += adv_loss_value.item()
            avg_rec_loss += rec_loss_value.item()

            tot_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Setting the description of the tqdm bar
            tqdm_step.set_description(
                "tot_loss: {:.3f} | adv_loss: {:.3f} | rec_loss: {:.3f} ".format(
                    tot_loss.item(), adv_loss_value.item(), rec_loss_value.item()
                ))
            tqdm_step.update()

        epoch_lrs = scheduler.get_last_lr()
        scheduler.step()

        avg_epoch_loss /= len(data_loaders['train'])
        avg_adv_loss /= len(data_loaders['train'])
        avg_rec_loss /= len(data_loaders['train'])

        tqdm_epoch.set_description(
            "avg_tot_loss: {:.3f} | avg_adv_loss: {:.3f} | avg_rec_loss: {:.3f}".format(
                avg_epoch_loss, avg_adv_loss, avg_rec_loss
            )
        )
        tqdm_epoch.update()

        rec_results, fair_results = evaluate(
            rec_model=rec_model,
            neural_head=adv_head,
            mod_weights=mod_weights,
            eval_loader=data_loaders['val'],
            rec_evaluator=rec_evaluator,
            fair_evaluator=fair_evaluator,
            device=debias_conf['device'],
            verbose=True
        )

        print(f"Epoch {curr_epoch} - ", generate_log_str(fair_results, n_groups))

        saving_dict = {
            'mod_weights': mod_weights.state_dict(),
            'epoch': curr_epoch,
            'rec_results': rec_results,
            'fair_results': fair_results,
        }

        if rec_results['ndcg@10'] > best_recacc_value:
            print(f"Epoch {curr_epoch} found best value.")
            best_recacc_value = rec_results['ndcg@10']
            best_recacc_epoch = curr_epoch

            # Save
            torch.save(saving_dict, os.path.join(debias_conf['save_path'], 'best_recacc.pth'))

        if fair_results['balanced_acc'] < worst_bacc_value:
            print(f"Epoch {curr_epoch} found worst value.")
            worst_bacc_value = fair_results['balanced_acc']
            worst_bacc_epoch = curr_epoch

            # Save
            torch.save(saving_dict, os.path.join(debias_conf['save_path'], 'worst_bacc.pth'))

        if curr_epoch % 5 == 0:
            torch.save(saving_dict, os.path.join(debias_conf['save_path'], f'epoch_{curr_epoch}.pth'))

        # Save last
        torch.save(saving_dict, os.path.join(debias_conf['save_path'], 'last.pth'))

        wandb.log(
            {
                **rec_results,
                **fair_results,
                'best_recacc_value': best_recacc_value,
                'worst_bacc_value': worst_bacc_value,
                'best_recacc_epoch': best_recacc_epoch,
                'worst_bacc_epoch': worst_bacc_epoch,
                'avg_epoch_loss': avg_epoch_loss,
                'avg_adv_loss': avg_adv_loss,
                'avg_rec_loss': avg_rec_loss,
                'epoch_lr_deltas': epoch_lrs[0],
                'epoch_lr_adv': epoch_lrs[1],
                'max_delta': mod_weights.deltas.max().item(),
                'min_delta': mod_weights.deltas.min().item(),
                'mean_delta': mod_weights.deltas.mean().item(),
            }
        )

    return n_delta_sets, user_to_delta_set
