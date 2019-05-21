# -*- coding: utf-8 -*-
import torch


class Config:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_path = r'E:\workspace\work\two_color_ball\data\data.csv'
    model_prefix = 'checkpoints/ball'
    model_path = 'checkpoints/ball.pth'
    pickle_path = 'data/data.npz'

    validation_split = .2
    shufflw_dataset = True
    random_seed = 1

    max_epoch = 400
    num_layers = 2
    hidden_dim = 256
    input_size = 1
    class_num = 33
    batch_size = 128
    num_workers = 2

    LR = 1e-3
    weight_decay = 1e-4
    weight = torch.Tensor([0.00248, 0.0031, 0.00373, 0.00448
                              , 0.0055, 0.00657, 0.00833, 0.010204
                              , 0.010101, 0.01818, 0.02439, 0.02632
                              , 0.0333, 0.043478, 0.05, 0.07143
                              , 0.25, 0.14286, 0.25, 0.33333, 2000
                              , 0.5, 1, 0.33333, 0.5
                              , 0.33333, 2000, 1, 2000, 0.5, 2000, 0.33333, 2000])
