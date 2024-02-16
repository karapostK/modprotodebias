import torch

from algorithms.algorithms_utils import AlgorithmsEnum
from data.data_utils import DatasetsEnum, get_dataloader
from eval.eval import FullEvaluator
from eval.metrics import ndcg_at_k_batch
from fair.fair_eval import FairEvaluator


def generate_log_str(fair_results, n_groups=2):
    val_list = []

    for i in range(n_groups):
        val_list.append(fair_results[f'recall_group_{i}'])

    log_str = "Balanced Accuracy: {:.3f} ".format(fair_results['balanced_acc'])

    recall_str = "("
    for i, v in enumerate(val_list):
        recall_str += "g{}: {:.3f}".format(i, v)
        if i != len(val_list) - 1:
            recall_str += " - "
    recall_str += ")"

    log_str += recall_str
    log_str += " - Unbalanced Accuracy: {:.3f}".format(fair_results['unbalanced_acc'])

    return log_str


def get_upsampling_values(train_mtx, n_groups, user_to_user_group, dataset_name: str, attribute: str):
    """
    This function computes the upsampling values for the CrossEntropy Loss.
    For each group we compute the total # of interactions. A group receives as weight (# interactions of the group with
    the highest number of interactions) / (# interactions of the group).
    """

    ce_weights = torch.zeros(n_groups, dtype=torch.float)

    for group_idx in range(n_groups):
        group_mask = user_to_user_group == group_idx
        ce_weights[group_idx] = train_mtx[group_mask].sum()

    ce_weights = ce_weights.max() / ce_weights

    if dataset_name == 'lfm2bdemobias' and attribute == 'age':
        # Last class is outliers. We ignore it
        ce_weights[-1] = 0.

    print('Current weights are:', ce_weights)
    return ce_weights


def prepare_data(rec_conf, debias_conf):
    dataset = DatasetsEnum[rec_conf['dataset']]
    print(f'Dataset is {dataset.name}')

    rec_conf['train_iterate_over'] = 'interactions'
    rec_conf['eval_batch_size'] = debias_conf['eval_batch_size']
    rec_conf['train_batch_size'] = debias_conf['train_batch_size']

    rec_conf['running_settings']['eval_n_workers'] = 2
    rec_conf['running_settings']['train_n_workers'] = 6

    data_loaders = {
        'train': get_dataloader(rec_conf, 'train'),
        'val': get_dataloader(rec_conf, 'val'),
        'test': get_dataloader(rec_conf, 'test')
    }

    train_dataset = data_loaders['train'].dataset

    user_to_user_group = train_dataset.user_to_user_group[debias_conf['group_type']]
    n_groups = train_dataset.n_user_groups[debias_conf['group_type']]

    print(f"Analysis is carried on <{debias_conf['group_type']}> with {n_groups} groups")

    ce_weights = get_upsampling_values(train_dataset.sampling_matrix, n_groups, user_to_user_group, dataset.name,
                                       debias_conf['group_type'])


    return data_loaders, user_to_user_group, n_groups, ce_weights


def prepare_rec_model(rec_conf, dataset):
    alg = AlgorithmsEnum[rec_conf['alg']]
    print(f'Algorithm is {alg.name}')

    rec_model = alg.value.build_from_conf(rec_conf, dataset)
    rec_model.load_model_from_path(rec_conf['model_path'])
    rec_model.requires_grad_(False)
    return rec_model


def prepare_evaluators(debias_conf, dataset_name, n_groups, user_to_user_group):
    rec_evaluator = FullEvaluator(aggr_by_group=True, n_groups=n_groups, user_to_user_group=user_to_user_group)
    rec_evaluator.K_VALUES = [10]
    rec_evaluator.METRICS = [ndcg_at_k_batch]
    rec_evaluator.METRIC_NAMES = ['ndcg@{}']
    fair_evaluator = FairEvaluator(fair_attribute=debias_conf['group_type'], dataset_name=dataset_name,
                                   n_groups=n_groups,
                                   user_to_user_group=user_to_user_group,
                                   device=debias_conf['device'])
    return rec_evaluator, fair_evaluator


def get_delta_conf(debias_conf, train_dataset):
    if debias_conf['delta_on'] == 'all':
        n_masks = 1
        user_idx_to_mask_idx = torch.zeros(train_dataset.n_users, dtype=torch.long).to(debias_conf['device'])
        print("Using a single mask for all users")
    elif debias_conf['delta_on'] == 'groups':
        n_masks = train_dataset.n_user_groups[debias_conf['group_type']]
        user_idx_to_mask_idx = train_dataset.user_to_user_group[debias_conf['group_type']].to(debias_conf['device'])
        print(f"Using a mask for each user group ({n_masks})")
    elif debias_conf['delta_on'] == 'users':
        n_masks = train_dataset.n_users
        user_idx_to_mask_idx = torch.arange(train_dataset.n_users, dtype=torch.long).to(debias_conf['device'])
        print(f"Using a mask for each user ({n_masks})")
    else:
        raise ValueError(f"Unknown value for delta_on: {debias_conf['delta_on']}")
    return n_masks, user_idx_to_mask_idx
