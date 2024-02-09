import pandas as pd
import torch
from torch.utils import data

ALPHAS = [.9, .7, .5, .3, .1, .0]
TOP_K_RECOMMENDATIONS = 10


def compute_rec_gap(results_dict: dict, metric_name: str = 'ndcg@10', return_perc: bool = True):
    """
    Easy function to compute the percentage difference, respect to the mean, of the difference between group_0 and group_1
    NB. The value returned has a sign and might also return a negative percentage. This represents the case where group_1
    receives better values than group 0.
    @param results_dict:
    @param metric_name:
    @param return_perc:
    @return:
    """
    rec_gap = (results_dict['group_0_' + metric_name] - results_dict['group_1_' + metric_name]) / results_dict[
        metric_name]
    if return_perc:
        return 100 * rec_gap
    else:
        return rec_gap


def compute_tag_user_frequencies(users_top_items, item_idxs_to_tags, user_idxs_to_group,
                                 reduce_weight_multiple_tags=True):
    """
    The function returns user-group tags frequencies of their recommendations. This is computed as follows:
    - Each user's recommendations are mapped to tags. If an item has multiple tags (and reduce_weight_multiple_tags=True)
        then each tags gets 1/len(tags) for weight.
    - The user's tag frequencies are computed accordingly by simply aggregating the tags from the recommendations.
    - The frequencies are averaged on group level.

    @param users_top_items: [n_users, *] Represents the recommendations for each user of the system. Each entry is an item id
    @param item_idxs_to_tags: Maps the item_idx to a list of tags
    @param user_idxs_to_group: Maps the user_idx to groups
    @param reduce_weight_multiple_tags: Whether to consider 1/len(tags) as weight when an item has multiple tags
    @return: group frequencies.
    """

    # Getting all the tags
    tags_names = sorted(list(set(item_idxs_to_tags.explode())))
    n_groups = user_idxs_to_group.max() + 1

    # Cumulative frequencies
    groups_cumulative_tag_frequencies = []
    for group_idxs in range(n_groups):
        groups_cumulative_tag_frequencies.append(pd.Series(data=[0.] * len(tags_names), index=tags_names,
                                                           name='tags'))

    for user_idx, user_top_items in users_top_items:
        # Tags of the top items
        n_items = len(user_top_items)
        user_top_items_tags = item_idxs_to_tags.loc[user_top_items]

        if reduce_weight_multiple_tags:
            # Giving smaller weights to the tags of an items when the items itself has multiple tags.
            # e.g. i_1 with genre 'Drama', 'Drama' gets weight of 1.
            # e.g. i_2 with genres 'Drama, Action' then both genres weight 0.5 instead of 1.
            # This partially counteract the interactions of tags appearing in every movie
            user_top_items_tags_lens = user_top_items_tags.apply(len).array
            user_top_items_tags_lens = user_top_items_tags_lens.repeat(user_top_items_tags_lens)
            tags_normalizing_values = user_top_items_tags_lens
        else:
            tags_normalizing_values = 1

        user_top_items_tags = user_top_items_tags.explode().to_frame()
        user_top_items_tags['frequency'] = 1 / tags_normalizing_values

        user_top_items_tags_frequencies = user_top_items_tags.groupby('tags').aggregate(sum).frequency

        if reduce_weight_multiple_tags:
            user_top_items_tags_frequencies /= n_items
        else:
            user_top_items_tags_frequencies /= len(user_top_items_tags_frequencies)

        user_group_idx = user_idxs_to_group.loc[user_idx]

        groups_cumulative_tag_frequencies[user_group_idx] = groups_cumulative_tag_frequencies[user_group_idx].add(
            user_top_items_tags_frequencies, fill_value=0)

    # Normalize by # of users
    for group_idxs in range(n_groups):
        groups_cumulative_tag_frequencies[group_idxs] = groups_cumulative_tag_frequencies[group_idxs] / \
                                                        user_idxs_to_group.value_counts().loc[group_idxs]

    return groups_cumulative_tag_frequencies




class ConcatDataLoaders:
    def __init__(self, dataloader_0: torch.utils.data.DataLoader, dataloader_1: torch.utils.data.DataLoader):
        self.dataloader_0 = dataloader_0
        self.dataloader_1 = dataloader_1
        self.zipped_dataloader = None

    def __iter__(self):
        self.zipped_dataloader = zip(self.dataloader_0, self.dataloader_1)
        return self

    def __next__(self):
        (u_idxs_0, i_idxs_0, labels_0), (u_idxs_1, i_idxs_1, labels_1) = self.zipped_dataloader.__next__()
        u_idxs = torch.cat([u_idxs_0, u_idxs_1], dim=0)
        i_idxs = torch.cat([i_idxs_0, i_idxs_1], dim=0)
        labels = torch.cat([labels_0, labels_1], dim=0)
        return u_idxs, i_idxs, labels

    def __len__(self):
        return min(len(self.dataloader_0), len(self.dataloader_1))
