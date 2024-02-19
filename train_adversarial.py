import os

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from tqdm import trange

import wandb
from fair.fair_eval import evaluate
from fair.neural_head import NeuralHead
from fair.repr_perturb import AddRepresentationPerturb
from fair.utils import generate_log_str, get_rec_model, get_delta_conf, \
    get_dataloaders, get_user_group_data, get_evaluators
from train.rec_losses import RecSampledSoftmaxLoss
from utilities.utils import reproducible
from utilities.wandb_utils import fetch_best_in_sweep


def train_adversarial(adv_config: dict, ):
    rec_conf = fetch_best_in_sweep(
        adv_config['best_run_sweep_id'],
        good_faith=True,
        preamble_path="~",
        project_base_directory='.'
    )

    # --- Preparing the Rec Model, Data & Evaluators --- #

    # Data
    data_loaders = get_dataloaders({
        **rec_conf,
        'eval_batch_size': adv_config['eval_batch_size'],
        'train_batch_size': adv_config['train_batch_size'],
        'running_settings': {'eval_n_workers': 2, 'train_n_workers': 6}
    })

    user_to_user_group, n_groups, ce_weights = get_user_group_data(
        train_dataset=data_loaders['train'].dataset,
        group_type=adv_config['group_type'],
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
        group_type=adv_config['group_type']
    )

    # --- Setting up the Model (Probe/Adversary) --- #

    reproducible(adv_config['seed'])

    # Neural Head
    adv_config['neural_layers_config'] = [64] + adv_config['neural_layers_config'] + [n_groups]
    adv_head = NeuralHead(
        layers_config=adv_config['neural_layers_config'],
        gradient_scaling=adv_config['gradient_scaling']
    )

    # Modular Weights
    n_masks, user_idx_to_mask_idx = get_delta_conf(adv_config, data_loaders['train'].dataset)

    repr_perturb = AddRepresentationPerturb(
        repr_dim=64,
        n_masks=n_masks,
        user_idx_to_mask_idx=user_idx_to_mask_idx,
        init_std=.1,
        clamp_boundaries=(0, 2)
    )

    # Optimizer & Scheduler
    adv_optimizer = torch.optim.AdamW(
        [
            {'params': repr_perturb.parameters()},
            {'params': adv_head.parameters()}
        ],
        lr=adv_config['lr'],
        weight_decay=adv_config['wd']
    )
    scheduler = CosineAnnealingLR(adv_optimizer, T_max=adv_config['n_epochs'], eta_min=1e-6)

    # Loss
    adv_loss = nn.CrossEntropyLoss(weight=ce_weights.to(adv_config['device']))
    rec_loss = RecSampledSoftmaxLoss.build_from_conf(rec_conf, data_loaders['train'].dataset)

    # Save path

    os.makedirs(os.path.dirname(adv_config['save_path']), exist_ok=True)
    wandb.config.update(adv_config)

    # --- Training the Model --- #
    user_to_user_group = user_to_user_group.to(adv_config['device'])
    rec_model.to(adv_config['device'])
    repr_perturb.to(adv_config['device'])
    adv_head.to(adv_config['device'])

    best_recacc_value = -torch.inf
    best_recacc_epoch = -1
    worst_bacc_value = torch.inf
    worst_bacc_epoch = -1

    for curr_epoch in trange(adv_config['n_epochs']):
        print(f"Epoch {curr_epoch}")

        avg_epoch_loss = 0
        avg_adv_loss = 0
        avg_rec_loss = 0

        for u_idxs, i_idxs, labels in tqdm(data_loaders['train']):
            u_idxs = u_idxs.to(adv_config['device'])
            i_idxs = i_idxs.to(adv_config['device'])
            labels = labels.to(adv_config['device'])

            i_repr = rec_model.get_item_representations(i_idxs)

            u_p, u_other = rec_model.get_user_representations(u_idxs)

            # Perturbing
            u_p = repr_perturb(u_p, u_idxs)

            ### Rec Loss ###
            u_repr = u_p, u_other
            rec_scores = rec_model.combine_user_item_representations(u_repr, i_repr)
            rec_loss_value = rec_loss.compute_loss(rec_scores, labels)

            ### Adversarial Head ###
            adv_out = adv_head(u_p)
            adv_loss_value = adv_loss(adv_out, user_to_user_group[u_idxs].long())

            ### Total Loss ###
            tot_loss = rec_loss_value + adv_config['lam_adv'] * adv_loss_value

            avg_epoch_loss += tot_loss.item()
            avg_adv_loss += adv_loss_value.item()
            avg_rec_loss += rec_loss_value.item()

            tot_loss.backward()
            adv_optimizer.step()
            adv_optimizer.zero_grad()

        epoch_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        avg_epoch_loss /= len(data_loaders['train'])
        avg_adv_loss /= len(data_loaders['train'])
        avg_rec_loss /= len(data_loaders['train'])

        rec_results, fair_results = evaluate(
            rec_model=rec_model,
            neural_head=adv_head,
            repr_perturb=repr_perturb,
            eval_loader=data_loaders['val'],
            rec_evaluator=rec_evaluator,
            fair_evaluator=fair_evaluator, device=adv_config['device'],
            verbose=True
        )

        print("Epoch {} - Avg. Train Loss is {:.3f} ({:.3f} Adv. Loss - {:.3f} Rec. Loss)"
              .format(curr_epoch, avg_epoch_loss, avg_adv_loss, avg_rec_loss))
        print(f"Epoch {curr_epoch} - ", generate_log_str(fair_results, n_groups))

        saving_dict = {
            'repr_perturb': repr_perturb.state_dict(),
            'epoch': curr_epoch,
            'rec_results': rec_results,
            'fair_results': fair_results,
        }

        if rec_results['ndcg@10'] > best_recacc_value:
            print(f"Epoch {curr_epoch} found best value.")
            best_recacc_value = rec_results['ndcg@10']
            best_recacc_epoch = curr_epoch

            # Save
            torch.save(saving_dict, os.path.join(adv_config['save_path'], 'best_recacc.pth'))

        if fair_results['balanced_acc'] < worst_bacc_value:
            print(f"Epoch {curr_epoch} found worst value.")
            worst_bacc_value = fair_results['balanced_acc']
            worst_bacc_epoch = curr_epoch

            # Save
            torch.save(saving_dict, os.path.join(adv_config['save_path'], 'worst_bacc.pth'))

        # Save last
        torch.save(saving_dict, os.path.join(adv_config['save_path'], 'last.pth'))

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
                'epoch_lr': epoch_lr
            }
        )

    return n_groups, n_masks, user_idx_to_mask_idx
