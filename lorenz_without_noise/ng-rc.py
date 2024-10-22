import os
import torch
import math
import numpy as np
from torch import nn
from predict import predict, get_theta
from torch.autograd import Variable as V
from model import Net
from data_util import read_data
from plot_pic_rc_predict import plot_pic
from lorenz_ode import lorenz_ode
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # train netword，get 0~10s y,z data, by x data
    trained_net = torch.load(f'./model/model_rb_pre_without_model_ex1.pth')
    # params = torch.load(f'/home/cxz/rb_pre_without_model1/params/params_rb_pre_without_model_ex213.pt')

    # get system's Wout, predict next 1s
    dt = 0.025
    num_d = 3
    warmup = 0
    traintime = 10
    testtime = 15
    maxtime = 25
    current_time = maxtime - testtime
    time_step = testtime

    xlim = (current_time, maxtime)

    xlim = (current_time, maxtime)

    # get acc_data
    acc_t, acc_data = read_data(f'./datas/lorenz_25.csv')
    # data = lorenz_ode(time_span=(0, maxtime), ic=[-8, 7, 27], t_eval=np.linspace(0, maxtime, int(maxtime / dt) + 1), method='RK45')
    # acc_t = data.t
    # # print(acc_t)
    # # print(acc_t)
    # acc_data = data.y
    # # print(acc_data)

    warmup_pts = round(warmup / dt)
    train_pts = round(traintime / dt)
    known_pts = round(current_time / dt)
    # trian_t = np.linspace(start=0.00, stop=train_time, num=train_num)
    # trian_T = V(torch.from_numpy(trian_t).double().reshape((-1, 1)))
    # y_trained = trained_net(trian_T)
    trian_T = V(torch.from_numpy(acc_t[0: train_pts + 1]).double().reshape((-1, 1)))
    y_trained = V(torch.from_numpy(acc_data[:, 0:train_pts + 1]).double()).T
    # print(trian_T)
    theta = get_theta(y_train=y_trained, num_d=num_d)
    old_theta = theta
    # theta[abs(theta) < 1e-1] = 0

    T = V(torch.from_numpy(acc_t[0: known_pts + 1]).double().reshape((-1, 1)))
    predict_Y = V(torch.from_numpy(acc_data[:, 0:known_pts + 1]).double()).T
    # print(T)

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

        plot_pic(ex='test', acc_data_t=acc_t, acc_data=acc_data, net=trained_net, 
                 predict_t=T[known_pts:].detach().numpy(), predict_data=predict_Y[known_pts:, :].detach().numpy(),
                 xlim=xlim)
        
        
        current_time += time_step

    torch.save(T, f'datas/rc_predict_T')
    torch.save(predict_Y, f'datas/rc_predict_Y')

        # theta = get_theta(y_train=ad_Y, num_d=num_d)

    # # 绘制绝对误差
    # color = ['purple', 'red']
    # start = 400
    # keep = 200
    # fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=False)
    # ax1.plot(T[400:400+keep], np.abs(predict_Y[400:400+keep, 0] - acc_data[0, 400:400+keep]), c=color[1], linestyle='-')  # 25s

    # ax1.set_xlabel('t')
    # ax1.set_ylabel('x')
    # # ax1.legend(['true', 'regression', 'predicion', 'adjust'], loc='upper right')

    # ax2.plot(T[400:400+keep], np.abs(predict_Y[400:400+keep, 1] - acc_data[1, 400:400+keep]), c=color[1], linestyle='-') # trian data

    # ax2.set_xlabel('t')
    # ax2.set_ylabel('y')

    # ax3.plot(T[400:400+keep], np.abs(predict_Y[400:400+keep, 2] - acc_data[2, 400:400+keep]), c=color[1], linestyle='-') # trian data

    # ax3.set_xlabel('t')
    # ax3.set_ylabel('z')
    # # fig1.legend(['true', 'ODE45'], loc='upper center', ncol=2)
    # fig2.savefig('rc_error.png')