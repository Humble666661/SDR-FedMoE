import random
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import h5py


class Depression_Dataset(Dataset):
    def __init__(self, data, labels, gender, score, transform=None):
        super().__init__()
        self.data = data
        self.labels = labels
        self.gender = gender
        self.score = score
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        label_item = self.labels[idx][0]
        score_item = self.score[idx][0]

        data_item = torch.from_numpy(data_item).float()
        label_item = torch.tensor(label_item, dtype=torch.float)
        score_item = torch.tensor(score_item, dtype=torch.float)

        if self.transform:
            data_item = self.transform(data_item)

        return data_item, label_item

def preprocess_features_md(dataset_type, feature_dir, F_bins=80, Window_size=120):
    if dataset_type == "MODMA":
        h5 = h5py.File(os.path.join(feature_dir, "MODMA.h5"), "r")
        gender = h5["gender"][:]
    elif dataset_type == "AVEC13":
        h5 = h5py.File(os.path.join(feature_dir, "AVEC13.h5"), "r")
        gender = h5["gender"][:]
    elif dataset_type == "DAIC":
        h5 = h5py.File(os.path.join(feature_dir, "DAIC-WOZ.h5"), "r")
        gender = h5["gender"][:]
    else:
        h5 = h5py.File(os.path.join(feature_dir, "AVEC14.h5"), "r")
        gender = h5["gender"][:]

    features = h5["features"]
    labels = h5["class"][:]
    score = h5["score"][:]

    updated_data = []
    dim_1_li = []
    num_files = len(features)
    for i in range(num_files):
        data_item = features[i][0]
        dim_1 = data_item.shape[0] // F_bins
        reshaped_data = np.reshape(data_item, (F_bins, dim_1))
        updated_data.append(reshaped_data)
        dim_1_li.append(dim_1)

    if dataset_type in {"MODMA", "DAIC", "AVEC13"}:
        min_dim = min(dim_1_li)
    else:
        min_dim = 1001
        updated_data_1, updated_labels, updated_gender, updated_score, dim_11_li = [], [], [], [], []
        for i in range(len(dim_1_li)):
            if dim_1_li[i] >= min_dim:
                updated_data_1.append(updated_data[i])
                updated_labels.append(labels[i])
                updated_gender.append(gender[i])
                updated_score.append(score[i])
                dim_11_li.append(len(updated_data[i][0]))
        updated_data, labels, gender, score, dim_1_li = (
            updated_data_1,
            updated_labels,
            updated_gender,
            updated_score,
            dim_11_li,
        )
        num_files = len(updated_data)

    if min_dim % Window_size == 0:
        length = min_dim // Window_size
    else:
        length = (min_dim // Window_size) + 1

    D_width = length * Window_size - min_dim
    pad_with = ((0, 0), (0, D_width))

    for i in range(num_files):
        D_dim = dim_1_li[i] - min_dim
        if D_dim >= 0:
            random_pointer = random.randint(0, D_dim)
            updated_data[i] = updated_data[i][:, random_pointer: random_pointer + min_dim]
            padded_data = np.pad(updated_data[i], pad_with, mode="constant", constant_values=0)
            padded_data = padded_data.reshape(F_bins, length, Window_size)
            updated_data[i] = padded_data.transpose(1, 0, 2)

    updated_data = np.array(updated_data)
    labels = np.array(labels)
    gender = np.array(gender)
    score = np.array(score)
    return updated_data, labels, gender, score


def split_sample_data(train_type, dataset_type, features, labels, gender, scores, gender_type):
    if dataset_type == "MODMA":
        sampled_size = 40
        sampled_indices = np.random.choice(62, size=sampled_size, replace=False)
        sampled_features = features[:, sampled_indices, :, :]
    elif dataset_type == "AVEC13":
        sampled_size = 94
        sampled_features = features
    elif dataset_type == "DAIC":
        sampled_size = 18
        sampled_indices = np.random.choice(100, size=sampled_size, replace=False)
        sampled_features = features
    else:
        sampled_size = 9
        sampled_features = features

    if gender_type == "F":
        F_indices = np.where(gender == 0)[0]
        F_subset, F_sub_labels = sampled_features[F_indices, :, :, :], labels[F_indices]
        if dataset_type == "MODMA":
            train_size, test_zero_size, test_one_size = 12, 2, 2
        else:
            train_size, test_zero_size, test_one_size = 109, 17, 11
        sampled_features, labels = F_subset, F_sub_labels
    elif gender_type == "M":
        M_indices = np.where(gender == 1)[0]
        M_subset, M_sub_labels = sampled_features[M_indices, :, :, :], labels[M_indices]
        if dataset_type == "MODMA":
            train_size, test_zero_size, test_one_size = 27, 5, 4
        else:
            train_size, test_zero_size, test_one_size = 73, 8, 10
        sampled_features, labels = M_subset, M_sub_labels
    else:
        if dataset_type == "MODMA":
            train_size, test_zero_size, test_one_size = 42, 6, 4
        elif dataset_type == "AVEC13":
            train_size, test_zero_size, test_one_size = 116, 15, 14
        elif dataset_type == "DAIC":
            train_size, test_zero_size, test_one_size = 151, 26, 12
        else:
            train_size, test_zero_size, test_one_size = 182, 25, 21

    zero_li, one_li = [], []
    for i in range(len(labels)):
        if labels[i] == 0:
            zero_li.append(i)
        else:
            one_li.append(i)

    zero_li = random.sample(zero_li, test_zero_size)
    one_li = random.sample(one_li, test_one_size)
    test_indices = np.array(sorted(zero_li + one_li))

    dataset_indices = np.arange(sampled_features.shape[0])
    train_indices = np.setdiff1d(dataset_indices, test_indices)

    train_features = sampled_features[train_indices, :, :, :]
    train_labels = labels[train_indices]
    train_score = scores[train_indices]
    train_gender = gender[train_indices]

    train_labels = np.expand_dims(np.tile(train_labels[:, np.newaxis], (1, sampled_size)), axis=-1)
    train_gender = np.expand_dims(np.tile(train_gender[:, np.newaxis], (1, sampled_size)), axis=-1)
    train_score = np.expand_dims(np.tile(train_score[:, np.newaxis], (1, sampled_size)), axis=-1)

    train_d_shape = train_features.shape
    train_l_shape = train_labels.shape
    train_s_shape = train_score.shape
    train_g_shape = train_gender.shape

    train_features = train_features.reshape(-1, train_d_shape[-2], train_d_shape[-1])
    train_labels = train_labels.reshape(-1, train_l_shape[-1])
    train_gender = train_gender.reshape(-1, train_g_shape[-1])
    train_score = train_score.reshape(-1, train_s_shape[-1])

    test_features = sampled_features[test_indices, :, :, :]
    test_labels = labels[test_indices]
    test_score = scores[test_indices]
    test_gender = gender[test_indices]

    test_labels = np.expand_dims(np.tile(test_labels[:, np.newaxis], (1, sampled_size)), axis=-1)
    test_gender = np.expand_dims(np.tile(test_gender[:, np.newaxis], (1, sampled_size)), axis=-1)
    test_score = np.expand_dims(np.tile(test_score[:, np.newaxis], (1, sampled_size)), axis=-1)

    test_features = test_features.reshape(-1, train_d_shape[-2], train_d_shape[-1])
    test_labels = test_labels.reshape(-1, train_l_shape[-1])
    test_gender = test_gender.reshape(-1, train_g_shape[-1])
    test_score = test_score.reshape(-1, train_s_shape[-1])

    train_count_1 = np.count_nonzero(train_labels)
    train_count_0 = train_labels.shape[0] - train_count_1

    return (
        train_features,
        train_labels,
        train_gender,
        train_score,
        test_features,
        test_labels,
        test_gender,
        test_score,
        train_count_0,
        train_count_1,
    )


def bdi_to_binary_label(labels, threshold=14):
    num_zero, num_one = 0, 0
    for i in range(len(labels)):
        if labels[i] < threshold:
            labels[i] = 0
            num_zero += 1
        else:
            labels[i] = 1
            num_one += 1
    return labels, num_zero, num_one
