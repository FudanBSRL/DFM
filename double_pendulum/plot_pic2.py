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


def plot_pic(path, ex, acc_data_t, acc_data, net, xlim, predict_t=None, predict_data=None, ad_t=None, ad_data=None, 
             before_ad_T=None, before_ad_data=None):

    # acc_data = np.loadtxt(open(f'./csvs/target25s.csv', "rb"), delimiter=",") 
    acc_t = acc_data_t
    acc_th1 = acc_data[0, :]
    acc_th2 = acc_data[1, :]
    acc_w1 = acc_data[2, :]
    acc_w2 = acc_data[3, :]

    if predict_data is not None:
        pre_t = predict_t
        pre_th1 = predict_data[:, 0]
        pre_th2 = predict_data[:, 1]
        pre_w1 = predict_data[:, 2]
        pre_w2 = predict_data[:, 3]

    if ad_data is not None:
        ad_t = ad_t
        ad_th1 = ad_data[:, 0]
        ad_th2 = ad_data[:, 1]
        ad_w1 = ad_data[:, 2]
        ad_w2 = ad_data[:, 3]

    if before_ad_data is not None:
        before_ad_t = before_ad_T
        before_ad_th1 = before_ad_data[:, 0]
        before_ad_th2 = before_ad_data[:, 1]
        before_ad_w1 = before_ad_data[:, 2]
        before_ad_w2 = before_ad_data[:, 3]


    color = ['red', 'green', 'blue']
    fig1, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=False)
    ax1.plot(acc_t, acc_th1, c=color[0], linestyle='-')  # 25s
    if predict_data is not None:
        ax1.plot(pre_t, pre_th1, c=color[1], linestyle='dashed') # predict data
    if ad_data is not None:
        ax1.plot(ad_t, ad_th1, c=color[2], linestyle='-.') # adjust data

    ax1.set_xlim(xlim)

    ax2.plot(acc_t, acc_th2, c=color[0], linestyle='-')  # 25s
    if predict_data is not None:
        ax2.plot(pre_t, pre_th2, c=color[1], linestyle='dashed') # predict data
    if ad_data is not None:
        ax2.plot(ad_t, ad_th2, c=color[2], linestyle='-.') # adjust data

    ax2.set_xlim(xlim)

    ax3.plot(acc_t, acc_w1, c=color[0], linestyle='-')  # 25s
    if predict_data is not None:
        ax3.plot(pre_t, pre_w1, c=color[1], linestyle='dashed') # predict data
    if ad_data is not None:
        ax3.plot(ad_t, ad_w1, c=color[2], linestyle='-.') # adjust data

    ax3.set_xlim(xlim)
    
    ax4.plot(acc_t, acc_w2, c=color[0], linestyle='-')  # 25s
    if predict_data is not None:
        ax4.plot(pre_t, pre_w2, c=color[1], linestyle='dashed') # predict data
    if ad_data is not None:
        ax4.plot(ad_t, ad_w2, c=color[2], linestyle='-.') # adjust data

    ax4.set_xlim(xlim)

    fig1.legend(['true', 'predicion', 'adjust'], loc='upper center', ncol=4)
    fig1.align_labels()
    fig1.savefig(f'{path}/{ex}_pic.png', dpi=500)

    
    # fig2
    color = ['red', 'green', 'blue']
    fig2, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=False)
    ax1.plot(acc_t, acc_th1, c=color[0], linestyle='-')  # 25s
    if before_ad_data is not None:
        ax1.plot(before_ad_t, before_ad_th1, c=color[1], linestyle='dashed') # predict data
    if ad_data is not None:
        ax1.plot(ad_t, ad_th1, c=color[2], linestyle='-.') # adjust data

    ax1.set_xlim(xlim)

    ax2.plot(acc_t, acc_th2, c=color[0], linestyle='-')  # 25s
    if before_ad_data is not None:
        ax2.plot(before_ad_t, before_ad_th2, c=color[1], linestyle='dashed') # predict data
    if ad_data is not None:
        ax2.plot(ad_t, ad_th2, c=color[2], linestyle='-.') # adjust data

    ax2.set_xlim(xlim)

    ax3.plot(acc_t, acc_w1, c=color[0], linestyle='-')  # 25s
    if before_ad_data is not None:
        ax3.plot(before_ad_t, before_ad_w1, c=color[1], linestyle='dashed') # predict data
    if ad_data is not None:
        ax3.plot(ad_t, ad_w1, c=color[2], linestyle='-.') # adjust data

    ax3.set_xlim(xlim)

    ax4.plot(acc_t, acc_w2, c=color[0], linestyle='-')  # 25s
    if before_ad_data is not None:
        ax4.plot(before_ad_t, before_ad_w2, c=color[1], linestyle='dashed') # predict data
    if ad_data is not None:
        ax4.plot(ad_t, ad_w2, c=color[2], linestyle='-.') # adjust data
    
    ax4.set_xlim(xlim)

    fig2.legend(['true', 'predict after adjust', 'adjust'], loc='upper center', ncol=4)
    fig2.align_labels()
    fig2.savefig(f'{path}/{ex}_pic_aj.png', dpi=500)

    plt.close('all')