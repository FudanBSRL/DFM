import os
import torch
import math
import numpy as np
from torch import nn
from predict import predict, get_theta
from torch.autograd import Variable as V
from model import Net
from plot_pic_rc_predict import plot_pic
from data_util import read_data
import matplotlib.pyplot as plt


if __name__ == '__main__':
    base_path = f'./'
    pre_train_ex = f'rb_pre_without_model_ex9'
    # train netword，get 0~10s y,z data, by x data
    # trained_net = torch.load(f'{base_path}/model/model_{pre_train_ex}.pth')
    # params = torch.load(f'/home/cxz/rb_pre_without_model1/params/params_rb_pre_without_model_ex213.pt')

    # get system's Wout, predict next 1s
    dt = 0.025
    num_d = 2
    warmup = 0
    traintime = 20
    testtime = 20
    maxtime = 40
    current_time = 20
    time_step = testtime

    xlim = (current_time, maxtime)

    # get acc_data
    # data = lorenz_ode(time_span=(0, maxtime), ic=[-8, 7, 27], t_eval=torch.linspace(0, maxtime, int(maxtime / dt) + 1), method='RK45')
    # data = lorenz_ode(time_span=(0, 25), ic=[-8., 7., 27.], t_eval=torch.linspace(0, 25, 1001), method='RK45')
    # acc_t = data.t
    # print(acc_t)
    # acc_data = data.y
    acc_t, acc_data = read_data('./csvs/d_pendulum.csv')

    train_pts = round(traintime / dt)
    
    trian_T = V(torch.from_numpy(acc_t[0: train_pts + 1]).double().reshape((-1, 1)))
    y_trained = V(torch.from_numpy(acc_data[:, 0:train_pts + 1]).double()).T
    # print(trian_T)
    theta = get_theta(y_train=y_trained, num_d=num_d)
    old_theta = theta

    # y_trained_reg = trained_net(V(torch.from_numpy(acc_t[0: warmuptrain_pts + 1])).double().reshape(-1, 1))
    # theta_reg = get_theta(y_train=y_trained_reg, num_d=num_d)
    # theta[abs(theta) < 1e-5] = 0

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

        plot_pic(ex='test', acc_data_t=acc_t, acc_data=acc_data, net=None, 
                 predict_t=T[train_pts:].detach().numpy(), predict_data=predict_Y[train_pts:, :].detach().numpy(),
                 xlim=xlim)
        
        
        current_time += time_step

    torch.save(T, f'datas/oriRC_predict_T')
    torch.save(predict_Y, f'datas/oriRC_predict_Y')