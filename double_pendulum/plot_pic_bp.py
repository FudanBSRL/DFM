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


def plot_pic(ex, acc_data_t, acc_data, net, predict_t=None, predict_data=None, ad_t=None, ad_data=None, 
             before_ad_T=None, before_ad_data=None):
        
    train_time = torch.linspace(0, 10, 1001).reshape(-1, 1)
    train_data = net(train_time.double())

    train_time = train_time.reshape(-1).detach().numpy()
    train_data = train_data.detach().numpy()

    # acc_data = np.loadtxt(open(f'./csvs/target25s.csv', "rb"), delimiter=",") 
    r = 401
    acc_t = acc_data_t[0:r]
    acc_x = acc_data[0, 0:r]
    acc_y = acc_data[1, 0:r]
    acc_z = acc_data[2, 0:r]
    acc_a = acc_data[3, 0:r]

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


    color = ['purple', 'red', 'green', 'black']
    fig1, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=False)
    ax1.plot(acc_t, acc_x, c=color[0], linestyle='-')  # 25s
    ax1.plot(train_time, train_data[:, 0], c=color[1], linestyle='--') # trian data
    if predict_data is not None:
        ax1.plot(pre_t, pre_x, c=color[2], linestyle='-.') # predict data
    if ad_data is not None:
        ax1.plot(ad_t, ad_x, c=color[3], linestyle='-') # adjust data

    # ax1.legend(['true', 'regression', 'predicion', 'adjust'], loc='upper right')

    ax2.plot(acc_t, acc_y, c=color[0], linestyle='-')  # 25s
    ax2.plot(train_time, train_data[:, 1], c=color[1], linestyle='--') # trian data
    if predict_data is not None:
        ax2.plot(pre_t, pre_y, c=color[2], linestyle='-.') # predict data
    if ad_data is not None:
        ax2.plot(ad_t, ad_y, c=color[3], linestyle='-') # adjust data


    ax3.plot(acc_t, acc_z, c=color[0], linestyle='-')  # 25s
    ax3.plot(train_time, train_data[:, 2], c=color[1], linestyle='--') # trian data
    if predict_data is not None:
        ax3.plot(pre_t, pre_z, c=color[2], linestyle='-.') # predict data
    if ad_data is not None:
        ax3.plot(ad_t, ad_z, c=color[3], linestyle='-', linewidth=1) # adjust data


    ax4.plot(acc_t, acc_a, c=color[0], linestyle='-')  # 25s
    ax4.plot(train_time, train_data[:, 3], c=color[1], linestyle='--') # trian data
    if predict_data is not None:
        ax4.plot(pre_t, pre_z, c=color[2], linestyle='-.') # predict data
    if ad_data is not None:
        ax4.plot(ad_t, ad_z, c=color[3], linestyle='-', linewidth=1) # adjust data

    fig1.savefig(f'img/{ex}_pic.png', dpi=500)

    
    # fig2
    color = ['purple', 'red', 'green', 'black']
    fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=False)
    ax1.plot(acc_t, acc_x, c=color[0], linestyle='-', linewidth=0.5)  # 25s
    ax1.plot(train_time, train_data[:, 0], c=color[1], linestyle='--') # trian data
    if before_ad_data is not None:
        ax1.plot(before_ad_t, before_ad_x, c=color[2], linestyle='-.') # predict data
    if ad_data is not None:
        ax1.plot(ad_t, ad_x, c=color[3], linestyle='-', linewidth=1) # adjust data

    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    ax1.legend(['true', 'regression', 'pre_after_ad', 'adjust'], loc='upper right')

    ax2.plot(acc_t, acc_y, c=color[0], linestyle='-', linewidth=0.5)  # 25s
    ax2.plot(train_time, train_data[:, 1], c=color[1], linestyle='--') # trian data
    if before_ad_data is not None:
        ax2.plot(before_ad_t, before_ad_y, c=color[2], linestyle='-.') # predict data
    if ad_data is not None:
        ax2.plot(ad_t, ad_y, c=color[3], linestyle='-', linewidth=1) # adjust data

    ax2.set_xlabel('t')
    ax2.set_ylabel('y')

    ax3.plot(acc_t, acc_z, c=color[0], linestyle='-', linewidth=0.5)  # 25s
    ax3.plot(train_time, train_data[:, 2], c=color[1], linestyle='--') # trian data
    if before_ad_data is not None:
        ax3.plot(before_ad_t, before_ad_z, c=color[2], linestyle='-.') # predict data
    if ad_data is not None:
        ax3.plot(ad_t, ad_z, c=color[3], linestyle='-', linewidth=1) # adjust data

    ax3.set_xlabel('t')
    ax3.set_ylabel('z')
    fig2.savefig(f'img/{ex}_pic_aj.png')

    plt.close('all')