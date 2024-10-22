import os
import torch
import math
import numpy as np
from torch import nn
from predict import predict, get_theta
from torch.autograd import Variable as V
from model import Net
from pde_lorenz import pde_loss_gpinn, pde_loss
from plot_pic_rc_predict import plot_pic
from lorenz_ode import lorenz_ode
import matplotlib.pyplot as plt
from data_util import read_data


if __name__ == '__main__':
    # train netword，get 0~10s y,z data, by x data
    pre_train_ex = f'rb_pre_without_model_ex6'
    trained_net = torch.load(f'./model/model_{pre_train_ex}.pth')

    # get system's Wout, predict next 1s
    dt = 0.05
    num_d = 3
    ridge_param = 2
    warmup = 0
    traintime = 10
    testtime = 15
    maxtime = warmup + traintime + testtime
    current_time = warmup + traintime
    time_step = 15

    # data = lorenz_ode(time_span=(0, 25), ic=[-8., 7., 27.], t_eval=np.linspace(0, 25, 501), method='RK45')
    # acc_t = data.t
    # print(acc_t)
    # acc_data = data.y
    acc_t, acc_data = read_data(f'./datas/lorenz_sparse.csv')

    warmup_pts = round(warmup / dt)
    train_pts = round(traintime / dt)
    warmuptrain_pts = round((warmup + traintime) / dt)
    trian_T = V(torch.from_numpy(acc_t[0: warmuptrain_pts + 1]).double().reshape((-1, 1)))
    y_trained = V(torch.from_numpy(acc_data[:, 0:warmuptrain_pts + 1]).double()).T
    print(trian_T)
    theta = get_theta(y_train=y_trained, num_d=num_d, ridge_param=ridge_param)
    old_theta = theta
    # theta[abs(theta) < 1e-1] = 0

    T = trian_T
    predict_Y = y_trained

    times = int(testtime / time_step)

    for i in range(times):
        pre_T = T[-num_d:]
        pre_data = predict_Y[-num_d:, :]
        pre_start = current_time + dt
        pre_step = int(time_step / dt)


        pre_T, predict_data = predict(theta=theta, num_d=num_d, dt=dt, pre_t=pre_T, predict_data=pre_data, 
                                    pre_start=pre_start, pre_step=pre_step)

        
        T = torch.cat((T, pre_T[num_d:, :]), axis=0)
        predict_Y = torch.cat((predict_Y, predict_data[num_d:, :]), axis=0)

        plot_pic(ex='test_ori_RC', acc_data_t=acc_t, acc_data=acc_data, net=trained_net, 
                 predict_t=T[train_pts:].detach().numpy(), predict_data=predict_Y[train_pts:, :].detach().numpy())
        
        
        current_time += time_step

    torch.save(T, f'datas/oriRC_predict_T')
    torch.save(predict_Y, f'datas/oriRC_predict_Y')