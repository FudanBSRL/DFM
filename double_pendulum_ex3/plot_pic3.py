import os
import torch
import time
import math
import random
import sys
import numpy as np
from torch.utils import data
from torch import nn
from torch.autograd import Variable as V
import matplotlib.pyplot as plt
from model import Net


def plot_pic(acc_data_t, acc_data, trained_net, net):
        
    train_time = torch.linspace(0, 10, 1001).reshape(-1, 1)
    train_data = trained_net(train_time.double())

    train_time = train_time.reshape(-1).detach().numpy()
    train_data = train_data.detach().numpy()


    predict_time = torch.linspace(10, 25, 1501).reshape(-1, 1)
    predict_data = net(predict_time.double())

    predict_time = predict_time.reshape(-1).detach().numpy()
    predict_data = predict_data.detach().numpy()

    # acc_data = np.loadtxt(open(f'./csvs/target25s.csv', "rb"), delimiter=",") 
    acc_t = acc_data_t
    acc_x = acc_data[0, :]
    acc_y = acc_data[1, :]
    acc_z = acc_data[2, :]

    color = ['red', 'green', 'blue']
    fig1, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=False)
    ax1.plot(acc_t, acc_x, c=color[0], linestyle='-')
    ax1.plot(train_time, train_data[:, 0], c=color[1], linestyle='-')
    ax1.plot(predict_time, predict_data[:, 0], c=color[2], linestyle='-')

    ax1.set_xlabel('$t$',fontdict={"family": "Times New Roman", "size": 14})
    ax1.set_ylabel('$x$',fontdict={"family": "Times New Roman", "size": 14})
    ax1.set_xlim((0, 25))
    ax1.set_ylim((-20, 20))

    ax2.plot(acc_t, acc_y, c=color[0], linestyle='-')
    ax2.plot(train_time, train_data[:, 1], c=color[1], linestyle='-')
    ax2.plot(predict_time, predict_data[:, 1], c=color[2], linestyle='-')

    ax2.set_xlabel('$t$', fontdict={"family": "Times New Roman", "size": 14})
    ax2.set_ylabel('$y$', fontdict={"family": "Times New Roman", "size": 14})
    ax2.set_xlim((0, 25))
    ax2.set_ylim((-25, 25))

    ax3.plot(acc_t, acc_z, c=color[0], linestyle='-')
    ax3.plot(train_time, train_data[:, 2], c=color[1], linestyle='-')
    ax3.plot(predict_time, predict_data[:, 2], c=color[2], linestyle='-')

    ax3.set_xlabel('$t$',fontdict={"family": "Times New Roman", "size": 14})
    ax3.set_ylabel('$z$',fontdict={"family": "Times New Roman", "size": 14})
    ax3.set_xlim((0, 25))
    ax3.set_ylim((0, 50))

    fig1.legend(['true', 'predicion', 'adjust'], loc='upper center', ncol=4)
    fig1.align_labels()
    fig1.savefig(f'img/cmp_pic.png')
    
    plt.close('all')

