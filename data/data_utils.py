import enum
import logging
import math
import os.path
import zipfile

import numpy as np
import pandas as pd
import requests
import scipy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataloader import TrainDataLoader, NegativeSampler
from data.dataset import TrainRecDataset, FullEvalDataset, TrainUserRecDataset

LOG_FILT_DATA_PATH = "log_filtering_data.txt"

MOVIELENS_1M_DATASET_LINK = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


class DatasetsEnum(enum.Enum):
    """
    Enum to keep track of all the dataset available. Note that the name of the dataset  should correspond to a folder
    in data. e.g. ml1m has a corresponding folder in /data
    """
    ml1m = enum.auto()
    lfm2bdemobias = enum.auto()


def print_and_log(log_file, n_lhs, n_users, n_items, text):
    """
    Prints to screen and logs to file statistics of the data during the processing steps.
    @param log_file: Name of the file to log information to
    @param n_lhs: Number of listening events
    @param n_users: Number of users
    @param n_items: Number of items
    @param text: Text to be added
    @return:
    """
    info_string = "{:10d} entries {:7d} users {:7d} items for {}".format(n_lhs, n_users, n_items, text)
    log_file.write(info_string + '\n')
    print(info_string)


def download_movielens1m_dataset(save_path: str = './'):
    """
    Downloads a movielens dataset.

    @type save_path: path to the folder where to save the raw dataset. Default to "./"
    """
    url = MOVIELENS_1M_DATASET_LINK

    print("Downloading the dataset...")
    req = requests.get(url)
    dataset_zip_name = os.path.join(save_path, "dataset.zip")

    with open(dataset_zip_name, 'wb') as fw:
        fw.write(req.content)

    # Unzipping
    with zipfile.ZipFile(dataset_zip_name, 'r') as zipr:
        zipr.extractall(save_path)

    os.remove(dataset_zip_name)

    os.rename('ml-1m', 'raw_dataset')

    print('Dataset downloaded')


def k_core_filtering(lhs: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Performs core-filtering on the dataset.
    @param lhs: Pandas Dataframe containing the listening records. Has columns ["user", "item"]
    @param k: Threshold for performing the k-core filtering.
    @return: Filtered Dataframe
    """
    while True:
        start_number = len(lhs)

        # Item pass
        item_counts = lhs.item.value_counts()
        item_above = set(item_counts[item_counts >= k].index)
        lhs = lhs[lhs.item.isin(item_above)]
        print('Records after item pass: ', len(lhs))

        # User pass
        user_counts = lhs.user.value_counts()
        user_above = set(user_counts[user_counts >= k].index)
        lhs = lhs[lhs.user.isin(user_above)]
        print('Records after user pass: ', len(lhs))

        if len(lhs) == start_number:
            print('Exiting...')
            break
    return lhs


def create_index(lhs: pd.DataFrame):
    """
    Associate an index for each user and item after having performed filtering steps. In order to avoid confusion, it
    sorts the data by timestamp, user, and item before creating the indexes.
    @param lhs: Pandas Dataframe containing the listening records. Has columns ["timestamp", "user", "item"]
    @return:
        lhs: Pandas Dataframe containing the listening records, now with the user_idx and item_idx columns
        user_idxs: Pandas Dataframe containing the user to user_idx mapping
        item_idxs: Pandas Dataframe containing the item to item_idx mapping
    """
    # Defining a unique order for the index assignment
    lhs = lhs.sort_values(['timestamp', 'user', 'item'])

    # Creating simple integer indexes used for sparse matrices
    user_idxs = lhs.user.drop_duplicates().reset_index(drop=True)
    item_idxs = lhs.item.drop_duplicates().reset_index(drop=True)
    user_idxs.index.name = 'user_idx'
    item_idxs.index.name = 'item_idx'
    user_idxs = user_idxs.reset_index()
    item_idxs = item_idxs.reset_index()
    lhs = lhs.merge(user_idxs).merge(item_idxs)
    return lhs, user_idxs, item_idxs


def split_temporal_order_ratio_based(lhs: pd.DataFrame, ratios=(0.8, 0.1, 0.1)):
    """
    Split the interation time-wise, for each user, using the ratio specified as parameters. E.g. A split with (0.7,
    0.2,0.1) will first sort the data by timestamp then take the first 70% of the interactions as train data,
    the subsequent 20% as validation data, and the remaining last 10% as test data. In order to avoid confusion,
    it sorts the data by timestamp, user_idx, and item_idx before splitting.
    @param lhs: lhs: Pandas Dataframe
    containing the listening records. Has columns ["timestamp", "user_idx", "item_idx"]
    @param ratios: float values that denote the ratios for train, val, test. The values must sum to 1.
    @return:
        lhs: Pandas Dataframe containing the listening records sorted.
        train_data: Pandas Dataframe containing the train data.
        val_data: Pandas Dataframe containing the val data.
        test_data: Pandas Dataframe containing the test data.

    """
    assert sum(ratios) == 1, 'Ratios do not sum to 1!'

    lhs = lhs.sort_values(['timestamp', 'user', 'item'])
    train_idxs = []
    val_idxs = []
    test_idxs = []
    for user, user_group in tqdm(lhs.groupby('user')):
        # Data is already sorted by timestamp
        n_test = math.ceil(len(user_group) * ratios[-1])
        n_val = math.ceil(len(user_group) * ratios[-2])
        n_train = len(user_group) - n_val - n_test

        train_idxs += list(user_group.index[:n_train])
        val_idxs += list(user_group.index[n_train:n_train + n_val])
        test_idxs += list(user_group.index[-n_test:])

    train_data = lhs.loc[train_idxs]
    val_data = lhs.loc[val_idxs]
    test_data = lhs.loc[test_idxs]

    return lhs, train_data, val_data, test_data


def split_random_order_ratio_based(lhs: pd.DataFrame, ratios=(0.8, 0.1, 0.1), seed=13):
    """
    Split the interaction in a random fashion, for each user, using the ratio specified as parameters. E.g. A split with (0.7,
    0.2,0.1) will first randomize the data then take the first 70% of the interactions as train data,
    the subsequent 20% as validation data, and the remaining last 10% as test data.
    @param lhs: lhs: Pandas Dataframe
    containing the listening records. Has columns ["timestamp", "user_idx", "item_idx"]
    @param ratios: float values that denote the ratios for train, val, test. The values must sum to 1.
    @param seed: seed for the ranomization
    @return:
        lhs: Pandas Dataframe containing the listening records sorted.
        train_data: Pandas Dataframe containing the train data.
        val_data: Pandas Dataframe containing the val data.
        test_data: Pandas Dataframe containing the test data.

    """
    assert sum(ratios) == 1, 'Ratios do not sum to 1!'
    lhs = lhs.sample(frac=1., random_state=seed)
    train_idxs = []
    val_idxs = []
    test_idxs = []
    for user, user_group in tqdm(lhs.groupby('user')):
        n_test = math.ceil(len(user_group) * ratios[-1])
        n_val = math.ceil(len(user_group) * ratios[-2])
        n_train = len(user_group) - n_val - n_test

        train_idxs += list(user_group.index[:n_train])
        val_idxs += list(user_group.index[n_train:n_train + n_val])
        test_idxs += list(user_group.index[-n_test:])

    train_data = lhs.loc[train_idxs]
    val_data = lhs.loc[val_idxs]
    test_data = lhs.loc[test_idxs]

    return lhs, train_data, val_data, test_data


def get_dataloader(conf: dict, split_set: str) -> DataLoader:
    """
        Returns the dataloader associated to the configuration in conf
    """

    if split_set == 'train':
        if conf['train_iterate_over'] == 'interactions':
            train_dataset = TrainRecDataset(data_path=conf['dataset_path'])
        elif conf['train_iterate_over'] == 'users':
            train_dataset = TrainUserRecDataset(data_path=conf['dataset_path'], n_pos=conf['pos_train'])
        else:
            raise ValueError(
                f"train_iterate_over value '{conf['train_iterate_over']}' is invalid! Please choose from [interactions, users]")

        sampler = NegativeSampler(
            train_dataset=train_dataset,
            n_neg=conf['neg_train'],
            neg_sampling_strategy=conf['train_neg_strategy']
        )
        dataloader = TrainDataLoader(
            sampler,
            train_dataset,
            batch_size=conf['train_batch_size'],
            shuffle=True,
            num_workers=conf['running_settings']['train_n_workers'],
            prefetch_factor=conf['running_settings']['prefetch_factor'] if 'prefetch_factor' in conf[
                'running_settings'] else 2
        )
        logging.info(f"Built Train DataLoader module \n"
                     f"- batch_size: {conf['train_batch_size']} \n"
                     f"- train_n_workers: {conf['running_settings']['train_n_workers']} \n")

    elif split_set == 'val':
        dataloader = DataLoader(
            FullEvalDataset(
                data_path=conf['dataset_path'],
                split_set='val',
            ),
            batch_size=conf['eval_batch_size'],
            num_workers=conf['running_settings']['eval_n_workers'],
        )
        logging.info(f"Built Val DataLoader module \n"
                     f"- batch_size: {conf['eval_batch_size']} \n"
                     f"- eval_n_workers: {conf['running_settings']['eval_n_workers']} \n")

    elif split_set == 'test':
        dataloader = DataLoader(
            FullEvalDataset(
                data_path=conf['dataset_path'],
                split_set='test',
            ),
            batch_size=conf['eval_batch_size'],
            num_workers=conf['running_settings']['eval_n_workers'],
        )
        logging.info(f"Built Test DataLoader module \n"
                     f"- batch_size: {conf['eval_batch_size']} \n"
                     f"- eval_n_workers: {conf['running_settings']['eval_n_workers']} \n")
    else:
        raise ValueError(f"split_set value '{split_set}' is invalid! Please choose from [train, val, test]")
    return dataloader

