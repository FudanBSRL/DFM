import os
import torch
import math
import numpy as np
import gc
from torch import nn
from predict import predict, get_theta
from torch.autograd import Variable as V
from model import Net
from plot_pic_rc_predict import plot_pic
from data_util import read_data_ex
import matplotlib.pyplot as plt


if __name__ == '__main__':
    base_path = f'./'
    pre_train_ex = f'rb_pre_without_model_ex4'
    trained_net = torch.load(f'{base_path}/model/model_{pre_train_ex}.pth')

    # get system's Wout, predict next 1s
    dt = 0.01
    num_d = 15
    start_train = 0
    traintime = 20
    testtime = 5
    current_time = 21
    maxtime = current_time + testtime
    time_step = testtime

    ridge_param = 1e-4
    # ridge_param = 3e-4

    xlim = (current_time, maxtime)

    acc_t, acc_data, _, _ = read_data_ex('./csvs/ex_data1.CSV', rate=round(1/dt), filter=False)

    start_train_pt = round(start_train / dt)
    train_pts = round(traintime / dt)
    
    trian_T = V(torch.from_numpy(acc_t[: train_pts + 1]).double().reshape((-1, 1)))
    T = trian_T

    y_trained = V(torch.from_numpy(acc_data[:, start_train_pt:train_pts + 1]).double()).T
    theta = get_theta(y_train=y_trained, num_d=num_d, ridge_param=ridge_param)
    predict_Y = V(torch.from_numpy(acc_data[:, :train_pts + 1]).double()).T

    train_pts = round(current_time / dt)
    T = V(torch.from_numpy(acc_t[: round(current_time / dt) + 1]).double().reshape((-1, 1)))
    predict_Y = V(torch.from_numpy(acc_data[:, :round(current_time / dt) + 1]).double()).T

    # y_trained_reg = V(trained_net(torch.from_numpy(acc_t[start_train_pt:train_pts + 1]).float().reshape(-1, 1) - 15).double().detach())
    # theta_reg = get_theta(y_train=y_trained_reg, num_d=num_d, ridge_param=ridge_param)
    # theta = theta_reg
    # predict_Y = V(trained_net(torch.from_numpy(acc_t[:train_pts + 1]).float().reshape(-1, 1) - 15)).double()


    times = int(testtime / time_step)

    for i in range(times):

        pre_T = T[-num_d:]
        pre_data = predict_Y[-num_d:, :]
        pre_start = current_time + dt
        pre_step = int(time_step / dt)


        pre_T, predict_data = predict(theta=theta, num_d=num_d, dt=dt, pre_t=pre_T, predict_data=pre_data, 
                                    pre_start=pre_start, pre_step=pre_step)

        
        T = torch.cat((T, pre_T[num_d:, :]), dim=0)
        predict_Y = torch.cat((predict_Y, predict_data[num_d:, :]), dim=0)

        plot_pic(ex='test', acc_data_t=acc_t, acc_data=acc_data, net=None, 
                 predict_t=T[train_pts:].detach().numpy(), predict_data=predict_Y[train_pts:, :].detach().numpy(),
                 xlim=xlim)
        
        current_time += time_step

    torch.save(T, f'datas/oriRC_predict_T')
    torch.save(predict_Y, f'datas/oriRC_predict_Y')