import os
import torch
import math
import gc
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from predict import predict, get_theta
from torch.autograd import Variable as V
from model import Net, Net2
from pde_lorenz import pde_loss_gpinn, pde_loss, jacobi
from lorenz_ode import lorenz_ode
from discover import discover_gpinn, discover, num_candidates


ex = 1

def plot(acc_data_t, acc_data, predict_t, predict_data):
    acc_t = acc_data_t
    acc_x = acc_data[:, 0]
    acc_y = acc_data[:, 1]
    acc_z = acc_data[:, 2]

    pre_t = predict_t
    pre_x = predict_data[:, 0]
    pre_y = predict_data[:, 1]
    pre_z = predict_data[:, 2]

    # print(predict_t.shape)
    # print(predict_data.shape)

    color = ['red', 'blue']
    fig1, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=False)
    ax1.plot(acc_t, acc_x, c=color[0], linestyle='-')  # 25s
    ax1.plot(pre_t, pre_x, c=color[1], linestyle='--')  # 25s

    ax1.set_xlabel('$t$',fontdict={"family": "Times New Roman", "size": 14})
    ax1.set_ylabel('$x$',fontdict={"family": "Times New Roman", "size": 14})
    ax1.set_xlim((10, 25))
    ax1.set_ylim((-20, 20))

    ax2.plot(acc_t, acc_y, c=color[0], linestyle='-')  # 25s
    ax2.plot(pre_t, pre_y, c=color[1], linestyle='--')  # 25s

    ax2.set_xlabel('$t$', fontdict={"family": "Times New Roman", "size": 14})
    ax2.set_ylabel('$y$', fontdict={"family": "Times New Roman", "size": 14})
    ax2.set_xlim((10, 25))
    ax2.set_ylim((-25, 25))

    ax3.plot(acc_t, acc_z, c=color[0], linestyle='-')  # 25s
    ax3.plot(pre_t, pre_z, c=color[1], linestyle='--')  # 25s

    ax3.set_xlabel('$t$',fontdict={"family": "Times New Roman", "size": 14})
    ax3.set_ylabel('$z$',fontdict={"family": "Times New Roman", "size": 14})
    ax3.set_xlim((10, 25))
    ax3.set_ylim((0, 50))

    fig1.legend(['true', 'predicion'], loc='upper center', ncol=4)
    fig1.align_labels()
    fig1.savefig(f'img0.5/pinn_pic{ex}.png')


if __name__ == '__main__':

    base_path = f'./'
    pre_train_ex = f'rb_pre_without_model_ex6'
    ksi = torch.load(f'{base_path}/ksis/ksi_{pre_train_ex}.pt')
    Lambda = torch.load(f'{base_path}/ksis/ksi_{pre_train_ex}_lambda.pt')

    
    dt = 0.025

    data = lorenz_ode(time_span=(0, 25), ic=[-8, 7, 27], t_eval=torch.linspace(0, 25, int(25 / dt) + 1), method='RK45')
    acc_t = data.t.T
    acc_data = data.y.T

    testtime = 15
    time_step = 0.5
    current_time = 10

    pre_T = acc_t[0:401]
    pre_Y = acc_data[0:401]

    for i in range(int(testtime / time_step)):
        gc.collect()

        T_k = torch.from_numpy(pre_T[-1:]).reshape(-1, 1).double()
        Y_k = torch.from_numpy(pre_Y[-1:]).reshape(-1, 3).double()

        # train_T = torch.linspace(current_time, current_time + time_step, 51).reshape(-1, 1).double()
        # train_T = train_T - (train_T[0, 0] + train_T[-1, 0]) / 2

        p_T_ori = torch.linspace(current_time, current_time + time_step, int(time_step / dt) + 1).reshape(-1, 1).double()
        p_T = p_T_ori - (p_T_ori[0, 0] + p_T_ori[-1, 0]) / 2

        train_T = torch.linspace(p_T[0, 0], p_T[-1, 0], 101).reshape(-1, 1).double()

        T0 = T_k - (p_T_ori[0, 0] + p_T_ori[-1, 0]) / 2

        net = Net(width=80, deep=3, in_num=1, out_num=3).double()
        for m in net.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

        mse_cost_function = torch.nn.MSELoss(reduction='mean')

        num_epoch = 400000
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=num_epoch, T_mult=2)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=[20000, 50000])

        for epoch in range(num_epoch):
            y0 = net(T0)
            mse0 = mse_cost_function(y0, Y_k)
            
            # msed0, _, _ = discover(net=net, T=T_k, ksi=ksi, Lambda=Lambda)
            # train_T = torch.from_numpy(np.random.uniform(low=p_T[0, 0], high=p_T[-1, 0], size=(101, 1)))
            msed, _, _ = discover(net=net, T=train_T, ksi=ksi, Lambda=Lambda)

            loss = 1e2 * mse0 + 1 * msed

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 100 == 0:
                print(f'{epoch}, loss {loss}, mse0 {mse0}, msed {msed}')
        
        # print(pre_T.shape)
        pre_T = np.concatenate((pre_T, p_T_ori.detach().numpy()[1:].reshape(-1)))
        pre_Y = np.concatenate((pre_Y, net(p_T).detach().numpy()[1:]), axis=0)

        plot(acc_t, acc_data, pre_T, pre_Y)

        current_time += time_step

        torch.save(torch.from_numpy(pre_T).reshape(-1, 1).double(), f'./datas/T_PINN{ex}.pt')
        torch.save(torch.from_numpy(pre_Y).reshape(-1, 3).double(), f'./datas/Y_PINN{ex}.pt')
