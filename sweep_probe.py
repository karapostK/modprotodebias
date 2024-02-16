import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from tqdm import trange

import wandb
from fair.fair_eval import evaluate
from fair.neural_head import NeuralHead
from fair.utils import generate_log_str, prepare_data, prepare_rec_model, prepare_evaluators
from utilities.utils import reproducible
from utilities.wandb_utils import fetch_best_in_sweep


def train_val_agent():
    # Initialization and gathering hyperparameters
    wandb.init(job_type='train/val')

    probe_config = {k: v for k, v in wandb.config.items() if k[0] != '_'}
    rec_conf = fetch_best_in_sweep(probe_config['best_run_sweep_id'], good_faith=True, preamble_path="~",
                                   wandb_project_name="test-sweep", wandb_entitiy_name='karapost',
                                   project_base_directory='.')

    # --- Preparing the Rec Model, Data & Evaluators --- #

    # Data
    data_loaders, user_to_user_group, n_groups, ce_weights = prepare_data(rec_conf, probe_config)
    # Recommender Model
    rec_model = prepare_rec_model(rec_conf, data_loaders['train'].dataset)
    # Evaluators
    rec_evaluator, fair_evaluator = prepare_evaluators(probe_config, rec_conf['dataset'], n_groups,
                                                       user_to_user_group)

    # --- Setting up the Model (Probe/Adversary) --- #

    reproducible(probe_config['seed'])

    # Neural Head
    probe_config['neural_layers_config'] = [64] + probe_config['neural_layers_config'] + [n_groups]
    probe = NeuralHead(layers_config=probe_config['neural_layers_config'])

    # Optimizer & Scheduler
    probe_optimizer = torch.optim.AdamW(probe.parameters(), lr=probe_config['lr'], weight_decay=probe_config['wd'])
    scheduler = CosineAnnealingLR(probe_optimizer, T_max=probe_config['n_epochs'], eta_min=1e-6)

    # Loss
    probe_loss = nn.CrossEntropyLoss(weight=ce_weights.to(probe_config['device']))

    # --- Training the Model --- #
    user_to_user_group = user_to_user_group.to(probe_config['device'])
    rec_model.to(probe_config['device'])
    probe.to(probe_config['device'])

    best_value = -torch.inf
    best_epoch = -1

    for curr_epoch in trange(probe_config['n_epochs']):
        print(f"Epoch {curr_epoch}")

        avg_epoch_loss = 0

        for u_idxs, _, _ in tqdm(data_loaders['train']):
            u_idxs = u_idxs.to('cuda')

            u_p, u_other = rec_model.get_user_representations(u_idxs)

            probe_out = probe(u_p)
            probe_loss_value = probe_loss(probe_out, user_to_user_group[u_idxs].long())

            avg_epoch_loss += probe_loss_value.item()

            probe_loss_value.backward()
            probe_optimizer.step()
            probe_optimizer.zero_grad()

        scheduler.step()

        avg_epoch_loss /= len(data_loaders['train'])

        rec_results, fair_results = evaluate(rec_model=rec_model, attr_model=probe,
                                             eval_loader=data_loaders['val'],
                                             rec_evaluator=rec_evaluator,
                                             fair_evaluator=fair_evaluator, device=probe_config['device'],
                                             verbose=True)

        print("Epoch {} - Avg. Train Loss is {:.3f}".format(curr_epoch, avg_epoch_loss))
        print(f"Epoch {curr_epoch} - ", generate_log_str(fair_results, n_groups))

        if fair_results['balanced_acc'] > best_value:
            print(f"Epoch {curr_epoch} found best value.")
            best_value = fair_results['balanced_acc']
            best_epoch = curr_epoch

        wandb.log(
            {
                **rec_results,
                **fair_results,
                'best_balanced_acc': best_value,
                'avg_epoch_loss': avg_epoch_loss,
                'best_epoch': best_epoch,
            }
        )

    wandb.finish()


train_val_agent()
