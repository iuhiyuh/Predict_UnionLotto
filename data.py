# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import conf
import torch.utils.data as Data
import pandas as pd


def make_data(config):
    if os.path.exists(config.data_path):
        csv = pd.read_csv(config.data_path)
        dataset = []
        for i in range(csv.shape[0]):
            row = csv.loc[i, :]
            dataset.append(row[2].replace('|', ','))
        with open('./data/data.txt', 'w') as file:
            for i, data in enumerate(dataset):
                file.write(str(data))
                if i != len(dataset):
                    file.write('\n')


def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]


def tranform_data():
    red_1, red_2, red_3, red_4, red_5, red_6, blue = [], [], [], [], [], [], []
    lists = [red_1, red_2, red_3, red_4, red_5, red_6, blue]

    if not os.path.exists("./data/data.txt"):
        make_data(conf.Config)
    with open('./data/data.txt', 'r') as file:
        DATA = file.read().split('\n')

    for data in DATA:
        for i, lista in enumerate(lists):
            lista.append(data.split(',')[i])

    new_lists = []
    for lista in lists:
        lista = list(map(int, lista))
        data_list_len = len(lista)
        end_index = int(np.float(data_list_len / float(11)) * 11)
        final_list = []
        for index, _ in enumerate(lista[0: end_index - 11]):
            tmp_list = []
            for i in range(11):
                tmp_list.append(lista[index + i])
            final_list.append(tmp_list)
        new_lists.append(final_list)

    x_datas, y_datas = [], []
    for lista in new_lists:
        x_data = np.array(lista)[:, 0:10].reshape([-1, 10, 1])
        y_data = np.array(lista)[:, 10:].reshape([-1, 1]).ravel()
        x_datas.append(x_data)
        y_datas.append(y_data - 1)

    return x_datas, y_datas


def get_data(ball_index):
    x_datas, y_datas = tranform_data()
    if 7 >= ball_index >= 0:
        dataset = Data.TensorDataset(torch.Tensor(x_datas[ball_index].tolist())
                                     , torch.Tensor(y_datas[ball_index].tolist()))
        return dataset
    else:
        return


def get_dataloader(config):
    train_dataloaders = []
    val_dataloaders = []
    for i in range(7):
        dataset = get_data(i)

        validation_split = config.validation_split
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))

        if config.shufflw_dataset:
            np.random.seed(config.random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[: split]

        train_sampler = Data.SubsetRandomSampler(train_indices)
        valid_sampler = Data.SubsetRandomSampler(val_indices)

        train_loader = Data.DataLoader(dataset=dataset
                                       , batch_size=config.batch_size
                                       , num_workers=config.num_workers
                                       , sampler=train_sampler)
        valid_loader = Data.DataLoader(dataset=dataset
                                       , batch_size=config.batch_size
                                       , num_workers=config.num_workers
                                       , sampler=valid_sampler)

        train_dataloaders.append(train_loader)
        val_dataloaders.append(valid_loader)
    return train_dataloaders, val_dataloaders


if __name__ == "__main__":
    tranform_data()
