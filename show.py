# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt


def show_loss_acc(log, epoch, path, flag, name):
    plt.clf()
    if name == "train":
        plt.plot(log, label=name)
    else:
        plt.plot(log, color='orangered', label=name)
    plt.legend(loc='best')
    plt.xlabel('epochs')
    plt.ylabel(flag)
    if flag == 'acc':
        plt.ylim((0, 1))
    else:
        plt.ylim((0, 8))
    plt.savefig("{}/{}_epoch{}".format(path, flag, epoch))


def show_data(x, y):
    plt.clf()
    plt.ioff()
    plt.scatter(x.numpy(), y.numpy())
