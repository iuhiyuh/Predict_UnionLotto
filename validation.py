# -*- coding: utf-8 -*-
from conf import *
from data import get_dataloader
from torch import nn
import numpy as np
from show import show_loss_acc


def val(criterion, datasets, model, idx, epoch, loss_logs, acc_logs, **kwargs):
    conf = Config()
    model.eval()
    for k, v in kwargs.items():
        setattr(conf, k, v)
    loss_log = []
    acc_log = []
    for step, (batch_x, batch_y) in enumerate(datasets[idx]):
        batch_x = batch_x.to(conf.device)
        batch_y = batch_y.to(conf.device)
        output = model(batch_x)
        loss = criterion(output, batch_y.long())
        los_log = loss.cpu().detach().numpy()
        loss_log.append(los_log)
        _, prediction = output.topk(1, 1, True)
        correct = prediction.t().eq(batch_y.view(1, -1).long())
        prediction = prediction.detach().cpu().numpy()
        n_correc_elems = correct.float().sum().item() / conf.batch_size
        acc_log.append(np.array(n_correc_elems, dtype=np.float32))

        print('Model: ', idx + 1
              , '| Epoch: ', epoch + 1
              , '| validation loss: %.4f' % los_log
              , '| acc: %.4f' % n_correc_elems)
        if step == len(datasets[idx]) - 1:
            loss_logs.append(np.array(np.mean(loss_log)))
            acc_logs.append(np.array(np.mean(acc_log)))
        if (epoch + 1) % 100 == 0 and step == len(datasets[idx]) - 1:
            show_loss_acc(loss_logs, epoch + 1, "./val_result/{}号球".format(idx), "loss", "validation")
            show_loss_acc(acc_logs, epoch + 1, "./val_result/{}号球".format(idx), "acc", "validation")
            np.savetxt("./val_result/{}号球/{}.txt".format(idx, epoch + 1), prediction, fmt='%0.8f')


