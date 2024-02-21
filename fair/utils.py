import torch
from torchinfo import summary

from algorithms.algorithms_utils import AlgorithmsEnum
from data.data_utils import DatasetsEnum, get_dataloader
from data.dataset import RecDataset
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


def get_dataloaders(config: dict):
    """
    Returns the dataloaders for the training, validation and test datasets.
    :param config: Configuration file containing the following fields:
    - dataset: The dataset to use
    (optional) - eval_batch_size: The batch size for the evaluation dataloader
    (optional) - train_batch_size: The batch size for the training dataloader
    (optional) - running_settings['eval_n_workers']: The number of workers for the evaluation dataloader
    (optional) - running_settings['train_n_workers']: The number of workers for the training dataloader
    :return: dict with keys 'train', 'val', 'test' and values the corresponding dataloaders
    """
    assert 'dataset' in config, 'The dataset is not specified in the configuration file'
    dataset = DatasetsEnum[config['dataset']]
    print(f'Dataset is {dataset.name}')

    config['train_iterate_over'] = 'interactions'

    data_loaders = {
        'train': get_dataloader(config, 'train'),
        'val': get_dataloader(config, 'val'),
        'test': get_dataloader(config, 'test')
    }

    return data_loaders


def get_user_group_data(train_dataset: RecDataset, dataset_name: str, group_type: str):
    """
    Returns the user to user group mapping, the number of groups for the specified group type, and the cross entropy
    weights for the specified group type.
    :param train_dataset: The training dataset
    :param dataset_name: Name of the dataset
    :param group_type: Name of the group
    :return:
    """
    assert group_type in train_dataset.user_to_user_group, f'Group type <{group_type}> not found in the dataset'
    assert dataset_name in ['lfm2bdemobias', 'ml1m'], f'Dataset <{dataset_name}> not supported'

    user_to_user_group = train_dataset.user_to_user_group[group_type]
    n_groups = train_dataset.n_user_groups[group_type]

    print(f"Analysis is carried on <{group_type}> with {n_groups} groups")

    ce_weights = get_upsampling_values(train_mtx=train_dataset.sampling_matrix, n_groups=n_groups,
                                       user_to_user_group=user_to_user_group, dataset_name=dataset_name,
                                       attribute=group_type)

    return user_to_user_group.long(), n_groups, ce_weights


def get_rec_model(rec_conf: dict, dataset: RecDataset):
    """
    Returns the recommender model. It builds the model from the configuration file and loads the model from the path.
    :param rec_conf: Configuration file for the recommender model
    :param dataset: Dataset used for training
    :return:
    """
    alg = AlgorithmsEnum[rec_conf['alg']]
    print(f'Algorithm is {alg.name}')

    rec_model = alg.value.build_from_conf(rec_conf, dataset)
    rec_model.load_model_from_path(rec_conf['model_path'])
    rec_model.requires_grad_(False)

    print('Rec Model Summary: ')
    summary(rec_model, input_size=[(10,), (10,)], device='cpu', dtypes=[torch.long, torch.long],
            col_names=['input_size', 'output_size', 'num_params'])
    print()

    return rec_model


def get_evaluators(n_groups: int, user_to_user_group: torch.Tensor, dataset_name: str, group_type: str):
    # Rec Evaluator
    rec_evaluator = FullEvaluator(
        aggr_by_group=True,
        n_groups=n_groups,
        user_to_user_group=user_to_user_group
    )
    # We only use NDCG@10
    rec_evaluator.K_VALUES = [10]
    rec_evaluator.METRICS = [ndcg_at_k_batch]
    rec_evaluator.METRIC_NAMES = ['ndcg@{}']

    # Fair Evaluator
    fair_evaluator = FairEvaluator(
        fair_attribute=group_type,
        dataset_name=dataset_name,
        n_groups=n_groups,
        user_to_user_group=user_to_user_group,
    )

    return rec_evaluator, fair_evaluator


def get_mod_weights_settings(delta_on: str, train_dataset, group_type: str = None):
    """
    Returns the number of delta sets and the mapping from users to delta sets.
    :param delta_on: Whether to use a single delta set for all users, a delta set for each user group, or a delta set for each user
    :param train_dataset: The training dataset
    :param group_type: If delta_on is 'groups', the group type should be specified
    :return:
    """
    assert delta_on in ['all', 'groups', 'users'], f'Unknown value for delta_on: {delta_on}'
    assert group_type is not None or delta_on != 'groups', 'group_type should be specified when delta_on is groups'

    if delta_on == 'all':
        n_delta_sets = 1
        user_to_delta_set = torch.zeros(train_dataset.n_users, dtype=torch.long)
        print("Using a single delta set for all users")
    elif delta_on == 'groups':
        n_delta_sets = train_dataset.n_user_groups[group_type]
        user_to_delta_set = train_dataset.user_to_user_group[group_type]
        print(f"Using a delta set for each user group ({n_delta_sets})")
    elif delta_on == 'users':
        n_delta_sets = train_dataset.n_users
        user_to_delta_set = torch.arange(train_dataset.n_users, dtype=torch.long)
        print(f"Using a delta set for each user ({n_delta_sets})")
    else:
        raise ValueError(f'Unknown value for delta_on: {delta_on}')

    return n_delta_sets, user_to_delta_set
