import os
import torch
import math
import gc
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from predict import predict, get_theta
from torch.autograd import Variable as V
from model import Net
from pde_lorenz import pde_loss_gpinn, pde_loss, jacobi
from dp_ode import dp_ode
from discover import discover, num_candidates
from data_util import read_data, read_data_ex


ex = 3

def plot(acc_data_t, acc_data, predict_t, predict_data):
    acc_t = acc_data_t
    acc_th1 = acc_data[:, 0]
    acc_th2 = acc_data[:, 1]
    acc_w1 = acc_data[:, 2]
    acc_w2 = acc_data[:, 3]

    pre_t = predict_t
    pre_th1 = predict_data[:, 0]
    pre_th2 = predict_data[:, 1]
    pre_w1 = predict_data[:, 2]
    pre_w2 = predict_data[:, 3]

    xlim = (20, 30)

    color = ['red', 'blue']
    fig1, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=False)
    ax1.plot(acc_t, acc_th1, c=color[0], linestyle='-')
    ax1.plot(pre_t, pre_th1, c=color[1], linestyle='--')

    ax1.set_xlim(xlim)

    ax2.plot(acc_t, acc_th2, c=color[0], linestyle='-')
    ax2.plot(pre_t, pre_th2, c=color[1], linestyle='--')

    ax2.set_xlim(xlim)

    ax3.plot(acc_t, acc_w1, c=color[0], linestyle='-')
    ax3.plot(pre_t, pre_w1, c=color[1], linestyle='--')

    ax3.set_xlim(xlim)

    ax4.plot(acc_t, acc_w2, c=color[0], linestyle='-')
    ax4.plot(pre_t, pre_w2, c=color[1], linestyle='--')

    ax4.set_xlim(xlim)

    fig1.legend(['true', 'predicion'], loc='upper center', ncol=4)
    fig1.align_labels()
    fig1.savefig(f'img0.5/pinn_pic{ex}.png', dpi=500)


if __name__ == '__main__':

    base_path = f'./'
    pre_train_ex = f'rb_pre_without_model_ex3'
    ksi = torch.load(f'{base_path}/ksis/ksi_{pre_train_ex}.pt').float()
    Lambda = torch.load(f'{base_path}/ksis/ksi_{pre_train_ex}_lambda.pt')

    # data = lorenz_ode(time_span=(0, 25), ic=[-8, 7, 27], t_eval=torch.linspace(0, 25, int(25 / 0.025) + 1), method='RK45')
    acc_t, acc_data, _, _ = read_data_ex('./csvs/ex_data1.CSV', rate=40)
    acc_t = acc_t.T
    acc_data = acc_data.T

    dt = 0.025
    testtime = 10
    time_step = 0.2
    current_time = 20

    pre_T = acc_t[0:801]
    pre_Y = acc_data[0:801]

    for i in range(int(testtime / time_step)):

        gc.collect()

        T_k = torch.from_numpy(pre_T[-1:]).reshape(-1, 1).float()
        Y_k = torch.from_numpy(pre_Y[-1:]).reshape(-1, 4).float()

        p_T_ori = torch.linspace(current_time, current_time + time_step, int(time_step / dt) + 1).reshape(-1, 1).float()
        p_T = p_T_ori - (p_T_ori[0, 0] + p_T_ori[-1, 0]) / 2

        train_T = torch.linspace(p_T[0, 0], p_T[-1, 0], round(time_step * 200)).reshape(-1, 1).float()

        T0 = T_k - (p_T_ori[0, 0] + p_T_ori[-1, 0]) / 2

        # net = Net(width=20, deep=10, in_num=1, out_num=4).float()
        net = Net(width=256, deep=5, in_num=1, out_num=4).float()
        for m in net.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

        mse_cost_function = torch.nn.MSELoss(reduction='mean')

        num_epoch = 15000
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

        for epoch in range(num_epoch):
            y0 = net(T0)
            mse0 = mse_cost_function(y0, Y_k)

            msed, u, u_t, _ = discover(net=net, T=train_T, ksi=ksi, Lambda=Lambda)

            loss =  1e2 * mse0 + 1 * msed

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f'{epoch + 1}, loss {loss}, mse0 {mse0}, msed {msed}')
        
        # 定义优化器
        optimizer = torch.optim.LBFGS(net.parameters(), 
                                    lr=1, 
                                    max_iter=20000, 
                                    max_eval=20000,
                                    tolerance_grad=1.0 * np.finfo(float).eps,
                                    tolerance_change=1.0 * np.finfo(float).eps,
                                    history_size=50)
        
        iteration = 0
        # 定义一个闭包函数，用于L-BFGS的线搜索
        def closure():
            global iteration
            optimizer.zero_grad()  # 清除之前的梯度

            y0 = net(T0)
            mse0 = mse_cost_function(y0, Y_k)

            msed, u, u_t, _ = discover(net=net, T=train_T, ksi=ksi, Lambda=Lambda)

            loss = 1e2 * mse0 + 1 * msed
            loss.backward()  # 进行反向传播
            iteration += 1
            if (iteration + 1) % 100 == 0:
                print(f'Iteration {iteration + 1}, mse0 {mse0:f}, msed {msed:f}')
            return loss
        
        # 执行优化步骤
        optimizer.step(closure)


        # print(pre_T.shape)
        pre_T = np.concatenate((pre_T, p_T_ori.detach().numpy()[1:].reshape(-1)))
        pre_Y = np.concatenate((pre_Y, net(p_T).detach().numpy()[1:]), axis=0)

        plot(acc_t, acc_data, pre_T, pre_Y)

        current_time += time_step

        torch.save(torch.from_numpy(pre_T).reshape(-1, 1).double(), f'./datas/T_PINN{ex}.pt')
        torch.save(torch.from_numpy(pre_Y).reshape(-1, 4).double(), f'./datas/Y_PINN{ex}.pt')
