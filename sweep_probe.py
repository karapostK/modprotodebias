import warnings

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from tqdm import trange

import wandb
from algorithms.algorithms_utils import AlgorithmsEnum
from data.data_utils import DatasetsEnum, get_dataloader
from eval.eval import FullEvaluator
from eval.metrics import ndcg_at_k_batch
from fair.fair_eval import FairEvaluator, evaluate
from fair.neural_head import NeuralHead
from fair.utils import generate_log_str
from utilities.utils import generate_id
from utilities.utils import reproducible
from utilities.wandb_utils import fetch_best_in_sweep


def train_val_agent():
    # Initialization and gathering hyperparameters
    run = wandb.init(job_type='train/val')

    probe_config = {k: v for k, v in wandb.config.items() if k[0] != '_'}
    best_conf = fetch_best_in_sweep(probe_config['best_run_sweep_id'], good_faith=True, preamble_path="~",
                                    wandb_project_name="test-sweep", wandb_entitiy_name='karapost',
                                    project_base_directory='.')

    alg = AlgorithmsEnum[best_conf['alg']]
    dataset = DatasetsEnum[best_conf['dataset']]
    print(f'Algorithm is {alg.name} and dataset is {dataset.name}')

    best_conf['train_iterate_over'] = 'interactions'
    best_conf['eval_batch_size'] = probe_config['eval_batch_size']
    best_conf['train_batch_size'] = probe_config['train_batch_size']

    best_conf['running_settings']['eval_n_workers'] = 2
    best_conf['running_settings']['train_n_workers'] = 6

    train_loader = get_dataloader(best_conf, 'train')
    val_loader = get_dataloader(best_conf, 'val')

    rec_model = alg.value.build_from_conf(best_conf, val_loader.dataset)
    rec_model.load_model_from_path(best_conf['model_path'])
    rec_model.requires_grad_(False)

    train_dataset = train_loader.dataset

    user_to_user_group = train_dataset.user_to_user_group[probe_config['group_type']]
    n_groups = train_dataset.n_user_groups[probe_config['group_type']]

    print(f"Analysis is carried on <{probe_config['group_type']}> with {n_groups} groups")

    # Computing UpSampling Values. This is used so the loss is balanced.
    # For each group we compute the total # of interactions.
    # A group receives as weight (# interactions of the group with the highest number of interactions) / (# interactions of the group)
    train_mtx = train_dataset.sampling_matrix
    ce_weights = torch.zeros(n_groups, dtype=torch.float)

    for group_idx in range(n_groups):
        group_mask = user_to_user_group == group_idx
        ce_weights[group_idx] = train_mtx[group_mask].sum()

    ce_weights = ce_weights.max() / ce_weights

    if DatasetsEnum.lfm2bdemobias == dataset and probe_config['group_type'] == 'age':
        # Last class is outliers. We ignore it
        ce_weights[-1] = 0.

    print('Current weights are:', ce_weights)

    id_run = generate_id()

    # Adjusting the values
    probe_config['neural_layers_config'] = [64] + probe_config['neural_layers_config'] + [n_groups]
    probe_config['id_run'] = id_run

    # --- Preparing the Model & Data --- #
    rec_evaluator = FullEvaluator(aggr_by_group=True, n_groups=n_groups, user_to_user_group=user_to_user_group)
    rec_evaluator.K_VALUES = [10]
    rec_evaluator.METRICS = [ndcg_at_k_batch]
    rec_evaluator.METRIC_NAMES = ['ndcg@{}']
    fair_evaluator = FairEvaluator(fair_attribute=probe_config['group_type'], dataset_name=dataset.name,
                                   n_groups=n_groups,
                                   user_to_user_group=user_to_user_group,
                                   device=probe_config['device'])

    reproducible(probe_config['seed'])

    probe = NeuralHead(layers_config=probe_config['neural_layers_config'])
    probe_optimizer = torch.optim.AdamW(probe.parameters(), lr=probe_config['lr'], weight_decay=probe_config['wd'])
    scheduler = CosineAnnealingLR(probe_optimizer, T_max=probe_config['n_epochs'], eta_min=1e-6)

    # Loss
    probe_loss = nn.CrossEntropyLoss(weight=ce_weights.to(probe_config['device']))
    user_to_user_group = user_to_user_group.to(probe_config['device'])

    rec_model.to(probe_config['device'])
    probe.to(probe_config['device'])

    best_value = -torch.inf
    best_epoch = -1

    for curr_epoch in trange(probe_config['n_epochs']):
        print(f"Epoch {curr_epoch}")

        avg_epoch_loss = 0

        for u_idxs, _, _ in tqdm(train_loader):
            u_idxs = u_idxs.to('cuda')

            u_p, u_other = rec_model.get_user_representations(u_idxs)

            probe_out = probe(u_p)
            probe_loss_value = probe_loss(probe_out, user_to_user_group[u_idxs].long())

            avg_epoch_loss += probe_loss_value.item()

            probe_loss_value.backward()
            probe_optimizer.step()
            probe_optimizer.zero_grad()

        scheduler.step()

        avg_epoch_loss /= len(train_loader)

        rec_results, fair_results = evaluate(rec_model=rec_model, attr_model=probe,
                                             eval_loader=val_loader,
                                             rec_evaluator=rec_evaluator,
                                             fair_evaluator=fair_evaluator, device=probe_config['device'],
                                             verbose=True)

        print("Epoch {} - Avg. Train Loss is {:.3f}".format(curr_epoch, avg_epoch_loss))
        log_str = generate_log_str(fair_results, n_groups)
        print(f"Epoch {curr_epoch} - ", log_str)

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
