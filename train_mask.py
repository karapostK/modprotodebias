import argparse
import warnings

import torch
import yaml
from torch import nn
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
from train.rec_losses import RecSampledSoftmaxLoss
from utilities.utils import generate_id
from utilities.utils import reproducible
from utilities.wandb_utils import fetch_best_in_sweep

parser = argparse.ArgumentParser(description='Start mask experiment')

warnings.filterwarnings("ignore")

parser.add_argument('--sweep_id', '-s', type=str, help='ID of the sweep', default='sshfyfwu')
parser.add_argument('--conf_path', '-c', type=str, help='Path to the configuration file.', default='./conf.yml')

args = parser.parse_args()

sweep_id = args.sweep_id
conf_path = args.conf_path

# Loading configuration file
with open(conf_path, 'r') as conf_file:
    adv_config = yaml.safe_load(conf_file)

# ML1M: ib8rad9q
# LFM2BDEMOBIAS: sshfyfwu

best_conf = fetch_best_in_sweep(sweep_id, good_faith=True, preamble_path="~/PycharmProjects",
                                wandb_project_name="test-sweep", wandb_entitiy_name='karapost',
                                project_base_directory='..')

alg = AlgorithmsEnum[best_conf['alg']]
dataset = DatasetsEnum[best_conf['dataset']]
print(f'Algorithm is {alg.name} and dataset is {dataset.name}')

best_conf['train_iterate_over'] = 'interactions'
best_conf['eval_batch_size'] = adv_config['eval_batch_size']
best_conf['train_batch_size'] = adv_config['train_batch_size']

train_loader = get_dataloader(best_conf, 'train')
val_loader = get_dataloader(best_conf, 'val')
test_loader = get_dataloader(best_conf, 'test')

rec_model = alg.value.build_from_conf(best_conf, val_loader.dataset)
rec_model.load_model_from_path(best_conf['model_path'])
rec_model.requires_grad_(False)

GROUP_TYPE = adv_config['group_type']

train_dataset = train_loader.dataset

user_to_user_group = train_dataset.user_to_user_group[GROUP_TYPE]
n_groups = train_dataset.n_user_groups[GROUP_TYPE]

print(f"Analysis is carried on <{GROUP_TYPE}> with {n_groups} groups")

# Computing UpSampling Values. This is used so the loss is balanced.
# For each group we compute the total # of interactions.
# A group receives as weight (# interactions of the group with the highest number of interactions) / (# interactions of the group)
train_mtx = train_dataset.sampling_matrix
ce_weights = torch.zeros(n_groups, dtype=torch.float)

for group_idx in range(n_groups):
    group_mask = user_to_user_group == group_idx
    ce_weights[group_idx] = train_mtx[group_mask].sum()

ce_weights = ce_weights.max() / ce_weights

if DatasetsEnum.lfm2bdemobias == dataset and GROUP_TYPE == 'age':
    # Last class is outliers. We ignore it
    ce_weights[-1] = 0.
print('Current weights are:', ce_weights)

id_run = generate_id()

# Adjusting the values
adv_config['neural_layers_config'] = [64] + adv_config['neural_layers_config'] + [n_groups]
adv_config['id_run'] = id_run

from fair.repr_perturb import AddRepresentationPerturb

rec_evaluator = FullEvaluator(aggr_by_group=True, n_groups=n_groups, user_to_user_group=user_to_user_group)
rec_evaluator.K_VALUES = [10]
rec_evaluator.METRICS = [ndcg_at_k_batch]
rec_evaluator.METRIC_NAMES = ['ndcg@{}']
fair_evaluator = FairEvaluator(fair_attribute=GROUP_TYPE, n_groups=n_groups, user_to_user_group=user_to_user_group,
                               device=adv_config['device'])

reproducible(adv_config['seed'])

adv_head = NeuralHead(layers_config=adv_config['neural_layers_config'],
                      gradient_scaling=adv_config['gradient_scaling'])

n_masks = None
user_idx_to_mask_idx = None
# Define deltas depending on the configuration
if adv_config['delta_on'] == 'all':
    n_masks = 1
    user_idx_to_mask_idx = torch.zeros(train_dataset.n_users, dtype=torch.long).to(adv_config['device'])
    print("Using a single mask for all users")
elif adv_config['delta_on'] == 'groups':
    n_masks = n_groups
    user_idx_to_mask_idx = user_to_user_group.to(adv_config['device'])
    print(f"Using a mask for each user group ({n_groups})")
elif adv_config['delta_on'] == 'users':
    n_masks = train_dataset.n_users
    user_idx_to_mask_idx = torch.arange(train_dataset.n_users, dtype=torch.long).to(adv_config['device'])
    print(f"Using a mask for each user ({train_dataset.n_users})")
else:
    raise ValueError(f"Unknown value for delta_on: {adv_config['delta_on']}")

# repr_perturb = ScaleRepresentationPerturb(repr_dim=64, n_masks=n_masks, user_idx_to_mask_idx=user_idx_to_mask_idx,
#                                init_std=.01, clamp_boundaries=(0, 2))
# repr_perturb = MultiplyRepresentationPerturb(repr_dim=64, n_masks=n_masks, user_idx_to_mask_idx=user_idx_to_mask_idx,
#                                             init_std=.01, clamp_boundaries=(0, 2))
repr_perturb = AddRepresentationPerturb(repr_dim=64, n_masks=n_masks, user_idx_to_mask_idx=user_idx_to_mask_idx,
                                        init_std=.01, clamp_boundaries=(0, 2))

adv_optimizer = torch.optim.AdamW(
    [
        {'params': repr_perturb.parameters()},
        {'params': adv_head.parameters()}
    ],
    lr=adv_config['lr'],
    weight_decay=adv_config['wd']
)

# Loss
adv_loss = nn.CrossEntropyLoss(weight=ce_weights.to(adv_config['device']))
user_to_user_group = user_to_user_group.to(adv_config['device'])
rec_loss = RecSampledSoftmaxLoss.build_from_conf(best_conf, train_dataset)

rec_model.to(adv_config['device'])
repr_perturb.to(adv_config['device'])
adv_head.to(adv_config['device'])

rec_model.requires_grad_(False)

# Wandb
wandb.init(project="fair_rec", entity="karapost", config=adv_config,
           name=f"{adv_config['delta_on']}_debiasing_{adv_config['id_run']}",
           job_type='train - val',
           tags=[dataset.name, alg.name, 'debiasing', GROUP_TYPE, adv_config['delta_on']])

wandb.watch(repr_perturb, log='all', log_freq=2194)

best_value = -torch.inf
worst_value = torch.inf
best_epoch = -1
curr_patience = adv_config['max_patience']

for curr_epoch in trange(adv_config['n_epochs']):
    print(f"Epoch {curr_epoch}")

    if curr_patience == 0:
        print("Ran out of patience, Stopping ")
        break

    avg_epoch_loss = 0
    avg_adv_loss = 0
    avg_rec_loss = 0

    for u_idxs, i_idxs, labels in tqdm(train_loader):
        u_idxs = u_idxs.to('cuda')
        i_idxs = i_idxs.to('cuda')
        labels = labels.to('cuda')

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
        tot_loss = adv_loss_value + adv_config['rec_loss_weight'] * rec_loss_value

        avg_epoch_loss += tot_loss.item()
        avg_adv_loss += adv_loss_value.item()
        avg_rec_loss += rec_loss_value.item()

        tot_loss.backward()
        adv_optimizer.step()
        adv_optimizer.zero_grad()

    avg_epoch_loss /= len(train_loader)
    avg_adv_loss /= len(train_loader)
    avg_rec_loss /= len(train_loader)

    # todo: how do I avoid the last group when considering age?
    rec_results, fair_results = evaluate(rec_model=rec_model, attr_model=adv_head, repr_perturb=repr_perturb,
                                         eval_loader=val_loader,
                                         rec_evaluator=rec_evaluator,
                                         fair_evaluator=fair_evaluator, device=adv_config['device'],
                                         verbose=True, )

    print(f"Epoch {curr_epoch} - Avg. Train Loss is {avg_epoch_loss}")
    log_str = generate_log_str(fair_results, n_groups)  # todo: what about age?
    print(f"Epoch {curr_epoch} - ", log_str)

    if fair_results['balanced_acc'] > best_value:
        print(f"Epoch {curr_epoch} found best value.")
        best_value = fair_results['balanced_acc']
        best_epoch = curr_epoch

        # Resetting patience
        curr_patience = adv_config['max_patience']
    else:
        curr_patience -= 1

    if fair_results['balanced_acc'] < worst_value:
        print(f"Epoch {curr_epoch} found worst value.")
        worst_value = fair_results['balanced_acc']

    wandb.log(
        {
            **rec_results,
            **fair_results,
            'best_balanced_acc': best_value,
            'worst_balanced_acc': worst_value,
            'avg_epoch_loss': avg_epoch_loss,
            'avg_adv_loss': avg_adv_loss,
            'avg_rec_loss': avg_rec_loss,
            'best_epoch': best_epoch,
        }
    )

wandb.finish()
