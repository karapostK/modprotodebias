import os

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from tqdm import trange

import wandb
from conf.protofair_conf_parser import parse_conf
from fair.fair_eval import evaluate
from fair.mod_weights import AddModularWeights
from fair.neural_head import NeuralHead
from fair.utils import generate_log_str, get_rec_model, get_mod_weights_settings, \
    get_dataloaders, get_user_group_data, get_evaluators, summarize
from train.rec_losses import RecSampledSoftmaxLoss
from utilities.utils import reproducible
from utilities.wandb_utils import fetch_best_in_sweep
import numpy as np
import math


def train_adversarial(adv_config: dict):
    adv_config = parse_conf(adv_config)

    rec_conf = fetch_best_in_sweep(
        adv_config['best_run_sweep_id'],
        good_faith=True,
    )

    # --- Preparing the Rec Model, Data & Evaluators --- #

    # Data
    data_loaders = get_dataloaders({
        **rec_conf,
        **adv_config,
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
    layers_config = [64] + adv_config['inner_layers_config'] + [n_groups]
    adv_head = NeuralHead(
        layers_config=layers_config,
        gradient_scaling=adv_config['gradient_scaling']
    )
    print()
    if adv_config["debiasing"] == "mmd":
        print('Maximum Mean Discrepancy Debiasing ')
    elif adv_config["debiasing"] == "adv":
        print('Adversarial Head Summary: ')
        summarize(adv_head, input_size=(10, 64), dtypes=[torch.float])
        print()

    # Modular Weights
    n_delta_sets, user_to_delta_set = get_mod_weights_settings(
        adv_config['delta_on'],
        data_loaders['train'].dataset,
        group_type=adv_config['group_type']
    )

    mod_weights = AddModularWeights(
        latent_dim=64,
        n_delta_sets=n_delta_sets,
        user_to_delta_set=user_to_delta_set,
        init_std=adv_config['init_std'],
        use_clamping=adv_config['use_clamping']
    )
    print()
    print('Modular Weights Summary: ')
    summarize(mod_weights, input_size=[(10, 64), (10,)], dtypes=[torch.float, torch.long])
    print()

    # Optimizer & Scheduler
    adv_optimizer = torch.optim.AdamW(
        [
            {
                'params': mod_weights.parameters(),
                'lr': adv_config['lr_deltas']
            },
            {
                'params': adv_head.parameters(),
                'lr': adv_config['lr_adv'] if adv_config["debiasing"] == "adv" else 0.
            },
        ],
        # lr=adv_config['lr'],
        weight_decay=adv_config['wd']
    )
    scheduler = CosineAnnealingLR(adv_optimizer, T_max=adv_config['n_epochs'], eta_min=adv_config['eta_min'])

    # Loss
    if adv_config['debiasing'] == 'adv':
        adv_loss = nn.CrossEntropyLoss(weight=ce_weights.to(adv_config['device']))
    elif adv_config['debiasing'] == 'mmd':
        adv_loss  = MMD()
    rec_loss = RecSampledSoftmaxLoss.build_from_conf(rec_conf, data_loaders['train'].dataset)

    # Save path

    os.makedirs(os.path.dirname(adv_config['save_path']), exist_ok=True)
    wandb.config.update(adv_config, allow_val_change=True)

    # --- Training the Model --- #
    user_to_user_group = user_to_user_group.to(adv_config['device'])
    rec_model.to(adv_config['device'])
    mod_weights.to(adv_config['device'])
    adv_head.to(adv_config['device'])

    best_recacc_value = -torch.inf
    best_recacc_epoch = -1
    worst_bacc_value = torch.inf
    worst_bacc_epoch = -1

    wandb.watch(mod_weights, log='all')
    tqdm_epoch = trange(adv_config['n_epochs'])

    for curr_epoch in tqdm_epoch:
        print(f"Epoch {curr_epoch}")

        avg_epoch_loss = 0
        avg_adv_loss = 0
        avg_rec_loss = 0
        tqdm_step = tqdm(data_loaders['train'])
        for u_idxs, i_idxs, labels in tqdm_step:
            u_idxs = u_idxs.to(adv_config['device'])
            i_idxs = i_idxs.to(adv_config['device'])
            labels = labels.to(adv_config['device'])

            i_repr = rec_model.get_item_representations(i_idxs)

            u_p, u_other = rec_model.get_user_representations(u_idxs)

            # Perturbing
            u_p = mod_weights(u_p, u_idxs)

            ### Rec Loss ###
            u_repr = u_p, u_other
            rec_scores = rec_model.combine_user_item_representations(u_repr, i_repr)
            rec_loss_value = rec_loss.compute_loss(rec_scores, labels)
            ### Adversarial Head ###
            adv_out = adv_head(u_p)
            if adv_config["debiasing"] == "adv":
                adv_loss_value = adv_loss(adv_out, user_to_user_group[u_idxs])
            elif adv_config["debiasing"] == "mmd":
                adv_loss_value = adv_loss(u_p, user_to_user_group[u_idxs])

            ### Total Loss ###
            tot_loss = rec_loss_value + adv_config['lam_adv'] * adv_loss_value

            avg_epoch_loss += tot_loss.item()
            avg_adv_loss += adv_loss_value.item()
            avg_rec_loss += rec_loss_value.item()

            tot_loss.backward()
            adv_optimizer.step()
            adv_optimizer.zero_grad()
            tqdm_step.set_description(f"total_loss:{np.round(tot_loss.item(),3)} | "
                                      f"adv_loss:{np.round(adv_loss_value.item(),3)} |"
                                      f" rec_loss:{np.round(rec_loss_value.item(),3)} |"
                                      f" w_deltas:{np.round(mod_weights.deltas[0,0:3].detach().cpu().numpy(),4)} |"
                                      f" adv_head:{np.round(adv_head.layers[-1].weight[0,0].item(),3)} |")
            tqdm_step.update()

        epoch_lrs = scheduler.get_last_lr()
        scheduler.step()

        avg_epoch_loss /= len(data_loaders['train'])
        avg_adv_loss /= len(data_loaders['train'])
        avg_rec_loss /= len(data_loaders['train'])
        tqdm_epoch.set_description(f"AVG total_train_loss:{np.round(avg_epoch_loss,3)} | "
                                   f"AVG adv_train_loss:{np.round(avg_adv_loss,3)} |"
                                   f"AVG rec_train_loss:{np.round(avg_rec_loss,3)}")
        tqdm_epoch.update()

        rec_results, fair_results = evaluate(
            rec_model=rec_model,
            neural_head=adv_head,
            mod_weights=mod_weights,
            eval_loader=data_loaders['val'],
            rec_evaluator=rec_evaluator,
            fair_evaluator=fair_evaluator,
            device=adv_config['device'],
            verbose=True
        )

        # print("Epoch {} - Avg. Train Loss is {:.3f} ({:.3f} Adv. Loss - {:.3f} Rec. Loss)"
        #       .format(curr_epoch, avg_epoch_loss, avg_adv_loss, avg_rec_loss))
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
            torch.save(saving_dict, os.path.join(adv_config['save_path'], 'best_recacc.pth'))

        if fair_results['balanced_acc'] < worst_bacc_value:
            print(f"Epoch {curr_epoch} found worst value.")
            worst_bacc_value = fair_results['balanced_acc']
            worst_bacc_epoch = curr_epoch

            # Save
            torch.save(saving_dict, os.path.join(adv_config['save_path'], 'worst_bacc.pth'))

        if curr_epoch % 5 == 0:
            torch.save(saving_dict, os.path.join(adv_config['save_path'], f'epoch_{curr_epoch}.pth'))

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
                'epoch_lr_deltas': epoch_lrs[0],
                'epoch_lr_adv': epoch_lrs[1],
                'max_delta': mod_weights.deltas.max().item(),
                'min_delta': mod_weights.deltas.min().item(),
                'mean_delta': mod_weights.deltas.mean().item(),
            }
        )

    return n_delta_sets, user_to_delta_set




class MMD(nn.Module):
    def __init__(self):
        super(MMD, self).__init__()

    def forward(self, embeddings, labels):
        # # mask = [0 for label in labels if label != 1]
        # # labels = torch.tensor(mask).to(labels.device)
        labels[labels != 1] = 0
        loss = self.mmd_loss(embeddings, labels, kernel_mul=4, kernel_num=4, fix_sigma=True)
        return loss

    def mmd_loss(self, embeds, domain_labels, kernel_mul, kernel_num, fix_sigma):
        loss = torch.tensor(0., device=domain_labels.device)
        # split into source and target samples
        unique_dl = domain_labels.unique()
        # if a batch with samples of only one domain is encountered - return 0 as loss
        if len(unique_dl) == 1:
            return loss
        src_mask = domain_labels == unique_dl[0]
        tgt_mask = domain_labels == unique_dl[1]
        # for all embeds calculate the mmd loss and sum them together

        # for embed in embeds:
            # try:
        src_embed = embeds[domain_labels == 0]
        tgt_embed = embeds[domain_labels == 1]
        embed_loss = self.mmd(src_embed, tgt_embed, kernel_mul, kernel_num, fix_sigma)
        loss += embed_loss
            # except:
            #     return loss
        return loss

    def gaussian_kernel(self, src_embed, tgt_embed, kernel_mul, kernel_num, fix_sigma):
        """Given source and target embeddings calculates kernel matrix based on given specific parameters."""
        if torch.mean(torch.abs(src_embed) + torch.abs(tgt_embed)) <= 1e-7:
            print("Warning: feature representations tend towards zero. "
                  "Consider decreasing 'da_lambda' or using lambda schedule.")
        n_samples = int(src_embed.size()[0] + tgt_embed.size()[0])
        total = torch.cat([src_embed, tgt_embed], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        l2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(l2_distance.detach()) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)  # shift bandwidth to the left
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-l2_distance / (bandwidth_temp + 1e-5)) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def mmd(self, src_embed, tgt_embed, kernel_mul, kernel_num, fix_sigma):
        src_batch_size = src_embed.size(0)
        tgt_batch_size = tgt_embed.size(0)
        # handle case when source and target are not of same size
        # we extend both to the fixed size 'batch size', because sizes should not vary between batches
        batch_size = src_batch_size + tgt_batch_size
        src_repeats = math.ceil(batch_size / src_batch_size)
        tgt_repeats = math.ceil(batch_size / tgt_batch_size)
        src_embed_rep = torch.cat([src_embed] * src_repeats, dim=0)[:batch_size]
        tgt_embed_rep = torch.cat([tgt_embed] * tgt_repeats, dim=0)[:batch_size]
        kernels = self.gaussian_kernel(src_embed_rep, tgt_embed_rep, kernel_mul, kernel_num, fix_sigma)
        # use different parts of the kernel matrix to calculate final loss
        xx = kernels[:batch_size, :batch_size]
        yy = kernels[batch_size:, batch_size:]
        xy = kernels[:batch_size, batch_size:]
        yx = kernels[batch_size:, :batch_size]
        loss = torch.mean(xx + yy - xy - yx)

        return loss