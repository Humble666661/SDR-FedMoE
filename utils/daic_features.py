import random
import numpy as np
import pandas as pd
import os
import h5py


def preprocess_features(feature_dir, f_bins=80, window_size=120):
    h5 = h5py.File(os.path.join(feature_dir, "daic-woz.h5"), "r")
    features = h5["features"]

    updated_data = []
    dim_1_li = []
    num_files = len(features)

    for i in range(num_files):
        data_item = features[i][0]
        dim_1 = data_item.shape[0] // f_bins
        reshaped_data = np.reshape(data_item, (f_bins, dim_1))
        updated_data.append(reshaped_data)
        dim_1_li.append(dim_1)

    min_dim = min(dim_1_li)
    if min_dim % window_size == 0:
        length = min_dim // window_size
    else:
        length = (min_dim // window_size) + 1

    d_width = length * window_size - min_dim
    pad_with = ((0, 0), (0, d_width))

    for i in range(num_files):
        d_dim = dim_1_li[i] - min_dim
        if d_dim >= 0:
            random_pointer = random.randint(0, d_dim)
            updated_data[i] = updated_data[i][:, random_pointer : random_pointer + min_dim]
            padded_data = np.pad(updated_data[i], pad_with, mode="constant", constant_values=0)
            padded_data = padded_data.reshape(f_bins, length, window_size)
            updated_data[i] = padded_data.transpose(1, 0, 2)

    updated_data = np.array(updated_data)
    return updated_data


def split_data(data_type, split_csv, features, id_li, gender):
    df = pd.read_csv(split_csv)
    if data_type == "train":
        data_id = df["Participant_ID"].tolist()
        data_labels = df["PHQ8_Binary"].tolist()
        data_gender = df["Gender"].tolist()
        data_score = df["PHQ8_Score"].tolist()
    else:
        zero_li = df[df["PHQ_Binary"] == 0]["Participant_ID"].tolist()
        one_li = df[df["PHQ_Binary"] == 1]["Participant_ID"].tolist()
        zero_li = random.sample(zero_li, 33)
        one_li = random.sample(one_li, 14)
        data_id = sorted(zero_li + one_li)
        data_labels = df[df["Participant_ID"].isin(data_id)]["PHQ_Binary"].tolist()
        data_gender = df[df["Participant_ID"].isin(data_id)]["Gender"].tolist()
        data_score = df[df["Participant_ID"].isin(data_id)]["PHQ_Score"].tolist()

    data_li = []
    for i in range(len(id_li)):
        if id_li[i] in data_id:
            data_li.append(features[i])

    if gender == "F":
        f_subset, f_sub_labels = [], []
        for i in range(len(data_gender)):
            if data_gender[i] == 0:
                f_subset.append(data_li[i])
                f_sub_labels.append(data_labels[i])
        f_subset, f_sub_labels = np.array(f_subset), np.array(f_sub_labels)
        f_sub_labels = np.expand_dims(np.tile(f_sub_labels[:, np.newaxis], (1, 18)), axis=-1)

        data_shape = f_subset.shape
        label_shape = f_sub_labels.shape
        f_subset = f_subset.reshape(-1, data_shape[-2], data_shape[-1])
        f_sub_labels = f_sub_labels.reshape(-1, label_shape[-1])

        if data_type == "train":
            count_1_f = np.count_nonzero(f_sub_labels)
            count_0_f = f_sub_labels.shape[0] - count_1_f
            return f_subset, f_sub_labels, count_0_f, count_1_f
        return f_subset, f_sub_labels

    if gender == "M":
        m_subset, m_sub_labels = [], []
        for i in range(len(data_gender)):
            if data_gender[i] == 1:
                m_subset.append(data_li[i])
                m_sub_labels.append(data_labels[i])
        m_subset, m_sub_labels = np.array(m_subset), np.array(m_sub_labels)
        m_sub_labels = np.expand_dims(np.tile(m_sub_labels[:, np.newaxis], (1, 18)), axis=-1)

        data_shape = m_subset.shape
        label_shape = m_sub_labels.shape
        m_subset = m_subset.reshape(-1, data_shape[-2], data_shape[-1])
        m_sub_labels = m_sub_labels.reshape(-1, label_shape[-1])
        if data_type == "train":
            count_1_m = np.count_nonzero(m_sub_labels)
            count_0_m = m_sub_labels.shape[0] - count_1_m
            return m_subset, m_sub_labels, count_0_m, count_1_m
        return m_subset, m_sub_labels

    data_li = np.array(data_li)
    data_labels = np.array(data_labels)
    data_gender = np.array(data_gender)
    data_score = np.array(data_score)

    data_labels = np.expand_dims(np.tile(data_labels[:, np.newaxis], (1, 18)), axis=-1)
    data_gender = np.expand_dims(np.tile(data_gender[:, np.newaxis], (1, 18)), axis=-1)
    data_score = np.expand_dims(np.tile(data_score[:, np.newaxis], (1, 18)), axis=-1)

    data_shape = data_li.shape
    label_shape = data_labels.shape
    gender_shape = data_gender.shape
    score_shape = data_score.shape
    data_li = data_li.reshape(-1, data_shape[-2], data_shape[-1])
    data_labels = data_labels.reshape(-1, label_shape[-1])
    data_gender = data_gender.reshape(-1, gender_shape[-1])
    data_score = data_score.reshape(-1, score_shape[-1])

    if data_type == "train":
        count_1 = np.count_nonzero(data_labels)
        count_0 = data_labels.shape[0] - count_1
        return data_li, data_labels, data_gender, data_score, count_0, count_1
    return data_li, data_labels, data_gender, data_score
