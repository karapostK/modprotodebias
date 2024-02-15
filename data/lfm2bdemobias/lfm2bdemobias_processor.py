import os
import shutil

import pandas as pd

from data.data_utils import LOG_FILT_DATA_PATH, print_and_log, create_index, split_temporal_order_ratio_based

if os.path.exists('./processed_dataset'):
    shutil.rmtree('./processed_dataset')
os.mkdir('./processed_dataset')

users_data_path = './raw_dataset/sampled_100000_items_demo.txt'
listening_events_path = './raw_dataset/sampled_100000_items_inter.txt'
log_filt_data_file = open(os.path.join('./processed_dataset', LOG_FILT_DATA_PATH), 'w+')

lhs = pd.read_csv(listening_events_path, sep='\t', header=None, names=['user', 'item'])
print_and_log(log_filt_data_file, len(lhs), lhs.user.nunique(), lhs.item.nunique(), 'Original Data')

# Loading users
users = pd.read_csv(users_data_path, delimiter='\t', names=['age', 'gender'], usecols=[2, 3])
users['user'] = users.index

# Data is already sorted by timestamp and filtered
# Adding back a timestamp column
lhs['timestamp'] = lhs.index

lhs, user_idxs, item_idxs = create_index(lhs)

print('Splitting the data, temporal ordered - ratio-based (80-10-10)')

lhs, train_data, val_data, test_data = split_temporal_order_ratio_based(lhs)

print_and_log(log_filt_data_file, len(train_data), train_data.user.nunique(), train_data.item.nunique(), 'Train Data')
print_and_log(log_filt_data_file, len(val_data), val_data.user.nunique(), val_data.item.nunique(), 'Val Data')
print_and_log(log_filt_data_file, len(test_data), test_data.user.nunique(), test_data.item.nunique(), 'Test Data')

log_filt_data_file.close()

# Adding grouping information
user_idxs = user_idxs.merge(users)
# Gender
user_idxs['gender_group_idx'] = (user_idxs.gender == 'f').astype(int)  # 0 is Male 1 is Female
# Age
valid_users = (user_idxs.age >= 10) & (user_idxs.age <= 95)

age_mean = user_idxs[valid_users].age.mean()
age_std = user_idxs[valid_users].age.std()
bins_boundaries = [age_mean - 2 * age_std, age_mean - age_std, age_mean + age_std, age_mean + 2 * age_std, 99]
bins_boundaries = [round(x) for x in bins_boundaries]
age_groups = pd.cut(user_idxs[valid_users].age, bins=bins_boundaries, labels=False)

user_idxs.loc[valid_users, 'age_group_idx'] = age_groups
user_idxs.loc[~valid_users, 'age_group_idx'] = len(bins_boundaries) - 1
user_idxs['age_group_idx'] = user_idxs['age_group_idx'].astype(int)

# Saving locally
print('Saving data to ./processed_dataset')

lhs.to_csv('./processed_dataset/listening_history.csv', index=False)
train_data.to_csv('./processed_dataset/listening_history_train.csv', index=False)
val_data.to_csv('./processed_dataset/listening_history_val.csv', index=False)
test_data.to_csv('./processed_dataset/listening_history_test.csv', index=False)

user_idxs.to_csv('./processed_dataset/user_idxs.csv', index=False)
item_idxs.to_csv('./processed_dataset/item_idxs.csv', index=False)
