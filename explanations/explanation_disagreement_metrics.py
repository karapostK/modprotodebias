from collections import defaultdict

import torch
from scipy.stats import spearmanr


def feature_agreement(feat_scores_1: torch.Tensor, feat_scores_2: torch.Tensor, k: int):
    """
    Computes the fraction of common features between the sets of top-k features of two sets of feature scores
    :param feat_scores_1:
    :param feat_scores_2:
    :param k:
    :return:
    """
    arg_tops_1 = torch.argsort(-feat_scores_1)[:k]
    arg_tops_2 = torch.argsort(-feat_scores_2)[:k]
    return len(set(arg_tops_1.numpy()).intersection(arg_tops_2.numpy())) / k


def rank_agreement(feat_scores_1: torch.Tensor, feat_scores_2: torch.Tensor, k: int):
    """
    Computes the fraction of features that are both in common between the two sets of feature scores and
    have the same rank in both sets.
    :param feat_scores_1:
    :param feat_scores_2:
    :param k:
    :return:
    """
    arg_tops_1 = torch.argsort(-feat_scores_1)[:k]
    arg_tops_2 = torch.argsort(-feat_scores_2)[:k]

    return (sum(arg_tops_1 == arg_tops_2) / k).item()


def rank_correlation(feat_scores_1: torch.Tensor, feat_scores_2: torch.Tensor, return_p_val=False):
    """
    Computes the Spearman's rank correlation between two sets of feature scores
    :param feat_scores_1:
    :param feat_scores_2:
    :return:
    """
    rho, p_val = spearmanr(feat_scores_1, feat_scores_2)
    if return_p_val:
        return rho, p_val
    return rho


def compute_disagreement_metrics(g_avg_sims, ks=(3, 5, 10, 20)):
    disagreement_metrics = defaultdict(dict)
    n_groups = len(g_avg_sims)
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            disagreement_metrics[(i, j)] = {'rank_correlation': rank_correlation(g_avg_sims[i], g_avg_sims[j])}
            for k in ks:
                disagreement_metrics[(i, j)][f'rank_agreement@{k}'] = rank_agreement(g_avg_sims[i], g_avg_sims[j], k)
                disagreement_metrics[(i, j)][f'feature_agreement@{k}'] = feature_agreement(g_avg_sims[i], g_avg_sims[j],
                                                                                           k)
    return disagreement_metrics
