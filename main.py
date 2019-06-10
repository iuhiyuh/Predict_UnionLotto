# -*- coding: utf-8 -*-
import os

from conf import Config
from data import get_dataloader
from train import train
from validation import val
from model import Model
from torch import nn

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

if __name__ == "__main__":
    conf = Config()
    for i in range(1):
        datasets_train, datasets_val = get_dataloader(conf)
        model = Model(conf).to(conf.device)
        model = nn.DataParallel(model)
        opt = optim.Adam(model.parameters(), lr=conf.LR, weight_decay=conf.weight_decay)
        criterion = nn.CrossEntropyLoss(weight=conf.weight).to(conf.device)
        loss_logs1, loss_logs2 = [], []
        acc_logs1, acc_logs2 = [], []
        for epoch in range(conf.max_epoch):
            train(opt, criterion, datasets_train, model, i, epoch, loss_logs1, acc_logs1)
            val(criterion, datasets_val, model, i, epoch, loss_logs2, acc_logs2)
