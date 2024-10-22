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


def plot_pic(ex, acc_data_t, acc_data, xlim, net=None, predict_t=None, predict_data=None, ad_t=None, ad_data=None, 
             before_ad_T=None, before_ad_data=None):
        
    # train_time = torch.linspace(0, 10, 1001).reshape(-1, 1)
    # train_data = net(train_time.double())

    # train_time = train_time.reshape(-1).detach().numpy()
    # train_data = train_data.detach().numpy()

    # acc_data = np.loadtxt(open(f'./csvs/target25s.csv', "rb"), delimiter=",") 
    acc_t = acc_data_t
    acc_th1 = acc_data[0, :]
    acc_th2 = acc_data[1, :]
    acc_w1 = acc_data[2, :]
    acc_w2 = acc_data[3, :]
    # acc_t = acc_data_t
    # acc_x1 = acc_data[0, :]
    # acc_y1 = acc_data[1, :]
    # acc_x2 = acc_data[2, :]
    # acc_y2 = acc_data[3, :]
    # acc_x1_dot = acc_data[4, :]
    # acc_y1_dot = acc_data[5, :]
    # acc_x2_dot = acc_data[6, :]
    # acc_y2_dot = acc_data[7, :]

    if predict_data is not None:
        pre_t = predict_t
        predict_data = predict_data.T
        pre_th1 = predict_data[0, :]
        pre_th2 = predict_data[1, :]
        pre_w1 = predict_data[2, :]
        pre_w2 = predict_data[3, :]
        # pre_t = predict_t
        # pre_x1 = predict_data[:, 0]
        # pre_y1 = predict_data[:, 1]
        # pre_x2 = predict_data[:, 2]
        # pre_y2 = predict_data[:, 3]
        # pre_x1_dot = predict_data[:, 4]
        # pre_y1_dot = predict_data[:, 5]
        # pre_x2_dot = predict_data[:, 6]
        # pre_y2_dot = predict_data[:, 7]

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


    color = ['blue', 'red', 'green', 'black']
    # fig1, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(nrows=8, ncols=1, sharex=True)
    fig1, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True)
    ax1.plot(acc_t, acc_th1, c=color[0], linestyle='-')  # 25s
    if predict_data is not None:
        ax1.plot(pre_t, pre_th1, c=color[2], linestyle='-.') # predict data
    ax1.set_xlim(xlim)
    ax1.set_ylim((-1, 1))
    fig1.legend(['true', 'RC predict'], loc='upper center', ncol=2)

    ax2.plot(acc_t, acc_th2, c=color[0], linestyle='-')  # 25s
    if predict_data is not None:
        ax2.plot(pre_t, pre_th2, c=color[2], linestyle='-.') # predict data
    ax2.set_xlim(xlim)
    ax2.set_ylim((-45, -25))

    ax3.plot(acc_t, acc_w1, c=color[0], linestyle='-')  # 25s
    if predict_data is not None:
        ax3.plot(pre_t, pre_w1, c=color[2], linestyle='-.') # predict data
    ax3.set_xlim(xlim)
    ax3.set_ylim((-10, 10))
        
    ax4.plot(acc_t, acc_w2, c=color[0], linestyle='-')  # 25s
    if predict_data is not None:
        ax4.plot(pre_t, pre_w2, c=color[2], linestyle='-.') # predict data
    ax4.set_xlim(xlim)
    ax4.set_ylim((-30, 30))

    # ax5.plot(acc_t, acc_x1_dot, c=color[0], linestyle='-')  # 25s
    # if predict_data is not None:
    #     ax5.plot(pre_t, pre_x1_dot, c=color[2], linestyle='-.') # predict data
    # ax5.set_xlim(xlim)
    # ax5.set_ylim((-1.5, 1.5))
    #
    # ax6.plot(acc_t, acc_y1_dot, c=color[0], linestyle='-')  # 25s
    # if predict_data is not None:
    #     ax6.plot(pre_t, pre_y1_dot, c=color[2], linestyle='-.') # predict data
    # ax6.set_xlim(xlim)
    # ax6.set_ylim((-0.6, 0.6))
    #
    # ax7.plot(acc_t, acc_x2_dot, c=color[0], linestyle='-')  # 25s
    # if predict_data is not None:
    #     ax7.plot(pre_t, pre_x2_dot, c=color[2], linestyle='-.') # predict data
    # ax7.set_xlim(xlim)
    # ax7.set_ylim((-2.6, 2.6))
    #
    # ax8.plot(acc_t, acc_y2_dot, c=color[0], linestyle='-')  # 25s
    # if predict_data is not None:
    #     ax8.plot(pre_t, pre_y2_dot, c=color[2], linestyle='-.') # predict data
    # ax8.set_xlim(xlim)
    # ax8.set_ylim((-2.6, 2.6))


    fig1.savefig(f'img/{ex}_pic.png', dpi=500)

    plt.close('all')