import typing
from collections import defaultdict

import torch
from tqdm import tqdm

from algorithms.base_classes import PrototypeWrapper
from eval.eval import FullEvaluator
from fair.mod_weights import ModularWeights
from fair.neural_head import NeuralHead


class FairEvaluator:
    """
    It computes the recall for each user group and the Balanced/Unbalanced Accuracy.
    """

    def __init__(self, fair_attribute: str, dataset_name: str, n_groups: int, user_to_user_group: torch.Tensor):
        """
        @param n_groups: How many user groups are there in the dataset
        @param user_to_user_group: Maps the user_idx to group_idx. Shape is [n_users]
        """
        assert fair_attribute in ['age', 'gender'], f'Fair attribute {fair_attribute} not recognized'
        assert dataset_name in ['ml1m', 'lfm2bdemobias'], f'Dataset {dataset_name} not recognized'

        self.dataset_name = dataset_name
        self.fair_attribute = fair_attribute
        self.n_groups = n_groups
        self.user_to_user_group = user_to_user_group

        self.group_metrics = None
        self.n_entries = None

        self._reset_internal_dict()

    def get_n_groups(self):
        return self.n_groups

    def get_user_to_user_group(self):
        return self.user_to_user_group

    def _reset_internal_dict(self):
        self.group_metrics = defaultdict(lambda: defaultdict(int))
        self.n_entries = defaultdict(int)

    def to(self, device: str):
        self.user_to_user_group = self.user_to_user_group.to(device)

    def eval_batch(self, u_idxs: torch.Tensor, logits: torch.Tensor):
        """
        :param u_idxs: User indexes. Shape is [batch_size]
        :param logits: Logits. Shape is [batch_size, n_classes]
        :return:
        """
        assert self.n_groups == logits.shape[
            1], f'Number of groups {self.n_groups} is different from logits shape at position 1 {logits.shape}'

        # n_entries[-1] holds the total amount of users.
        self.n_entries[-1] += u_idxs.shape[0]

        preds = torch.argmax(logits, dim=1)

        batch_group_labels = self.user_to_user_group[u_idxs]

        for group_idx in range(self.n_groups):
            group_metric_idx = torch.where(batch_group_labels == group_idx)[0]
            self.n_entries[group_idx] += len(group_metric_idx)
            self.group_metrics[group_idx]['TP'] += (preds[group_metric_idx] == group_idx).sum().item()

    def get_results(self):
        """
        Recall per group is defined as # TP (for that group) / # Entries in the group
        Unbalanced Accuracy is defined as # TP / # Total Entries
        Balanced Accuracy is defined as the mean of the recall per group (e.g. 0.5 Recall_Group_0 + 0.5 Recall_Group_1)
        NB. For LFM2BDemobias - Age the last group contains the outliers and hence is ignored from accuracy/balanced accuracy.
        """
        metrics_dict = dict()
        metrics_dict['balanced_acc'] = 0
        metrics_dict['unbalanced_acc'] = 0

        if self.n_entries[-1] == 0:
            return metrics_dict

        for group_idx in self.group_metrics:
            metrics_dict[f'recall_group_{group_idx}'] = self.group_metrics[group_idx]['TP'] / self.n_entries[group_idx]

            if self.dataset_name == 'lfm2bdemobias' and self.fair_attribute == 'age' and group_idx == self.n_groups - 1:
                continue
            metrics_dict['balanced_acc'] += metrics_dict[f'recall_group_{group_idx}']
            metrics_dict['unbalanced_acc'] += self.group_metrics[group_idx]['TP']

        if self.dataset_name == 'lfm2bdemobias' and self.fair_attribute == 'age':
            metrics_dict['balanced_acc'] /= (self.n_groups - 1)
            metrics_dict['unbalanced_acc'] /= (self.n_entries[-1] - self.n_entries[self.n_groups - 1])
        else:
            metrics_dict['balanced_acc'] /= self.n_groups
            metrics_dict['unbalanced_acc'] /= self.n_entries[-1]

        self._reset_internal_dict()

        return metrics_dict


def evaluate(rec_model: PrototypeWrapper,
             neural_head: typing.Union[NeuralHead, None],
             eval_loader: torch.utils.data.DataLoader,
             rec_evaluator: FullEvaluator,
             fair_evaluator: FairEvaluator,
             mod_weights: ModularWeights = None,
             device: str = 'cpu',
             verbose: bool = False):
    if neural_head is not None:
        neural_head.eval()

    if verbose:
        iterator = tqdm(eval_loader)
    else:
        iterator = eval_loader

    fair_evaluator.to(device)
    rec_evaluator.to(device)
    with torch.no_grad():
        # We generate the item representation once (usually the bottleneck of evaluation)
        i_idxs = torch.arange(eval_loader.dataset.n_items).to(device)
        i_repr = rec_model.get_item_representations(i_idxs)

        for u_idxs, labels, batch_mask in iterator:
            u_idxs = u_idxs.to(device)
            labels = labels.to(device)
            batch_mask = batch_mask.to(device)

            u_p, u_other = rec_model.get_user_representations(u_idxs)

            # Perturbing the representation if needed
            if mod_weights is not None:
                u_p = mod_weights(u_p, u_idxs)

            # Recommendation Evaluation
            u_repr = u_p, u_other
            rec_scores = rec_model.combine_user_item_representations(u_repr, i_repr)
            rec_scores[batch_mask] = -torch.inf

            rec_evaluator.eval_batch(u_idxs, rec_scores, labels)

            # Fairness Evaluation
            if neural_head is not None:
                attr_scores = neural_head(u_p)

                fair_evaluator.eval_batch(u_idxs, attr_scores)

    rec_results = rec_evaluator.get_results()
    fair_results = fair_evaluator.get_results()

    if neural_head is not None:
        neural_head.train()
    return rec_results, fair_results
