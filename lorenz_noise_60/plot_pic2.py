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


def plot_pic(path, ex, acc_data_t, acc_data, net, predict_t=None, predict_data=None, ad_t=None, ad_data=None, 
             before_ad_T=None, before_ad_data=None):
        
    train_time = torch.linspace(0, 10, 1001).reshape(-1, 1)
    train_data = net(train_time.double())

    train_time = train_time.reshape(-1).detach().numpy()
    train_data = train_data.detach().numpy()

    # acc_data = np.loadtxt(open(f'./csvs/target25s.csv', "rb"), delimiter=",") 
    acc_t = acc_data_t
    acc_x = acc_data[0, :]
    acc_y = acc_data[1, :]
    acc_z = acc_data[2, :]

    if predict_data is not None:
        pre_t = predict_t
        pre_x = predict_data[:, 0]
        pre_y = predict_data[:, 1]
        pre_z = predict_data[:, 2]

    if ad_data is not None:
        ad_t = ad_t
        ad_x = ad_data[:, 0]
        ad_y = ad_data[:, 1]
        ad_z = ad_data[:, 2]

    if before_ad_data is not None:
        before_ad_t = before_ad_T
        before_ad_x = before_ad_data[:, 0]
        before_ad_y = before_ad_data[:, 1]
        before_ad_z = before_ad_data[:, 2]


    color = ['red', 'green', 'blue']
    fig1, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=False)
    ax1.plot(acc_t, acc_x, c=color[0], linestyle='-')  # 25s
    # ax1.plot(train_time, train_data[:, 0], c=color[1], linestyle='--') # trian data
    if predict_data is not None:
        ax1.plot(pre_t, pre_x, c=color[1], linestyle='dashed') # predict data
    if ad_data is not None:
        ax1.plot(ad_t, ad_x, c=color[2], linestyle='-.') # adjust data

    ax1.set_xlabel('$t$',fontdict={"family": "Times New Roman", "size": 14})
    ax1.set_ylabel('$x$',fontdict={"family": "Times New Roman", "size": 14})
    ax1.set_xlim((10, 25))
    ax1.set_ylim((-20, 20))

    ax2.plot(acc_t, acc_y, c=color[0], linestyle='-')  # 25s
    # ax2.plot(train_time, train_data[:, 1], c=color[1], linestyle='--') # trian data
    if predict_data is not None:
        ax2.plot(pre_t, pre_y, c=color[1], linestyle='dashed') # predict data
    if ad_data is not None:
        ax2.plot(ad_t, ad_y, c=color[2], linestyle='-.') # adjust data

    ax2.set_xlabel('$t$', fontdict={"family": "Times New Roman", "size": 14})
    ax2.set_ylabel('$y$', fontdict={"family": "Times New Roman", "size": 14})
    ax2.set_xlim((10, 25))
    ax2.set_ylim((-25, 25))

    ax3.plot(acc_t, acc_z, c=color[0], linestyle='-')  # 25s
    # ax3.plot(train_time, train_data[:, 2], c=color[1], linestyle='--') # trian data
    if predict_data is not None:
        ax3.plot(pre_t, pre_z, c=color[1], linestyle='dashed') # predict data
    if ad_data is not None:
        ax3.plot(ad_t, ad_z, c=color[2], linestyle='-.') # adjust data

    ax3.set_xlabel('$t$',fontdict={"family": "Times New Roman", "size": 14})
    ax3.set_ylabel('$z$',fontdict={"family": "Times New Roman", "size": 14})
    ax3.set_xlim((10, 25))
    ax3.set_ylim((0, 50))

    fig1.legend(['true', 'predicion', 'adjust'], loc='upper center', ncol=4)
    fig1.align_labels()
    fig1.savefig(f'{path}/{ex}_pic.png', dpi=500)

    
    # fig2
    color = ['red', 'green', 'blue']
    fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=False)
    ax1.plot(acc_t, acc_x, c=color[0], linestyle='-')  # 25s
    # ax1.plot(train_time, train_data[:, 0], c=color[1], linestyle='--') # trian data
    if before_ad_data is not None:
        ax1.plot(before_ad_t, before_ad_x, c=color[1], linestyle='dashed') # predict data
    if ad_data is not None:
        ax1.plot(ad_t, ad_x, c=color[2], linestyle='-.') # adjust data

    ax1.set_xlabel('t',fontdict={"family": "Times New Roman", "size": 14})
    ax1.set_ylabel('x',fontdict={"family": "Times New Roman", "size": 14})
    ax1.set_xlim((10, 25))
    ax1.set_ylim((-20, 20))

    ax2.plot(acc_t, acc_y, c=color[0], linestyle='-')  # 25s
    # ax2.plot(train_time, train_data[:, 1], c=color[1], linestyle='--') # trian data
    if before_ad_data is not None:
        ax2.plot(before_ad_t, before_ad_y, c=color[1], linestyle='dashed') # predict data
    if ad_data is not None:
        ax2.plot(ad_t, ad_y, c=color[2], linestyle='-.') # adjust data

    ax2.set_xlabel('t', fontdict={"family": "Times New Roman", "size": 14})
    ax2.set_ylabel('y', fontdict={"family": "Times New Roman", "size": 14})
    ax2.set_xlim((10, 25))
    ax2.set_ylim((-25, 25))

    ax3.plot(acc_t, acc_z, c=color[0], linestyle='-')  # 25s
    # ax3.plot(train_time, train_data[:, 2], c=color[1], linestyle='--') # trian data
    if before_ad_data is not None:
        ax3.plot(before_ad_t, before_ad_z, c=color[1], linestyle='dashed') # predict data
    if ad_data is not None:
        ax3.plot(ad_t, ad_z, c=color[2], linestyle='-.') # adjust data

    ax3.set_xlabel('t',fontdict={"family": "Times New Roman", "size": 14})
    ax3.set_ylabel('z',fontdict={"family": "Times New Roman", "size": 14})
    ax3.set_xlim((10, 25))
    ax3.set_ylim((0, 50))
    
    fig2.legend(['true', 'predict after adjust', 'adjust'], loc='upper center', ncol=4)
    fig2.align_labels()
    fig2.savefig(f'{path}/{ex}_pic_aj.png', dpi=500)

    plt.close('all')