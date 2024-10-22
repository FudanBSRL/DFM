import os
import gc
import torch
import math
import numpy as np
from torch import nn
from predict import predict, get_theta
from torch.autograd import Variable as V
from model import Net
from pde_lorenz import pde_loss_gpinn, pde_loss, jacobi
from plot_pic import plot_pic
from plot_pic2 import plot_pic as plot_pic2
from lorenz_ode import lorenz_ode
from discover import discover_gpinn, discover, num_candidates
from data_util import read_data


def ad_train(pd_net_path, net, num_d, alpha, num_epoch, input, label, ksi, Lambda):

    input = input.detach()
    input.requires_grad = False
    label = label.detach()
    label.requires_grad = False
    ksi = ksi.detach()
    ksi.requires_grad = False

    IC_t = input[0:num_d, :]
    IC_label = label[0:num_d, :]

    PT = torch.linspace(input[0, 0], input[-1, 0], 100).reshape(-1, 1).double()

    mse_cost_function = torch.nn.MSELoss(reduction='mean')
    
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2000, T_mult=2)

    for epoch in range(num_epoch):
        y0 = net(IC_t)
        mse0 = mse_cost_function(y0, IC_label)

        # msed, u, _ = discover(net=net, T=input, ksi=ksi, Lambda=Lambda)
        # mseu = mse_cost_function(u, label)
        # PT = torch.from_numpy(np.random.uniform(input[0, 0], input[-1, 0], (20, 1))).double()
        msed, u, _ = discover(net=net, T=PT, ksi=ksi, Lambda=Lambda)
        y_hat = net(input)
        mseu = mse_cost_function(y_hat, label)
        if epoch < 2000:
            loss = 0.5 * mseu + 0.5 * (msed)
        else:
            loss = (1 - alpha) * mseu + alpha * (msed)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        if (epoch + 1) % 100 == 0:
            print(f'epoch {epoch + 1}, mse0 {mse0:f}, mseu {mseu:f}, msed {msed:f}')

    # 定义优化器
    optimizer = torch.optim.LBFGS(net.parameters(), 
                                  lr=1, 
                                  max_iter=5000, 
                                  max_eval=5000,
                                  tolerance_grad=1.0 * np.finfo(float).eps,
                                  tolerance_change=1.0 * np.finfo(float).eps,
                                  history_size=50)
    
    iteration = 0
    # 定义一个闭包函数，用于L-BFGS的线搜索
    def closure():
        nonlocal iteration, alpha
        optimizer.zero_grad()  # 清除之前的梯度

        y0 = net(IC_t)
        mse0 = mse_cost_function(y0, IC_label)

        msed, u, _ = discover(net=net, T=PT, ksi=ksi, Lambda=Lambda)
        y_hat = net(input)
        mseu = mse_cost_function(y_hat, label)
        loss = (1 - alpha) * mseu + alpha * (msed)

        loss.backward()  # 进行反向传播
        iteration += 1
        if (iteration) % 100 == 0:
            print(f'Iteration {iteration}, mse0 {mse0:f}, mseu {mseu:f}, msed {msed:f}')
        return loss
    
    # 执行优化步骤
    optimizer.step(closure)
    
    return net


def adjust_data(pd_net_path, num_d, alpha, input, label, ksi, Lambda):
    if os.path.exists(pd_net_path):
        ad_net = torch.load(pd_net_path)
    else:
        ad_net = Net(width=256, deep=4, in_num=1, out_num=3).double()
        ad_net.double()
        for m in ad_net.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    input = input - (input[0, 0] + input[-1, 0]) / 2
    net = ad_train(pd_net_path=pd_net_path, net=ad_net, num_d=num_d, alpha=alpha, num_epoch=10000, input=input, label=label, ksi=ksi, Lambda=Lambda)

    adjusted_data = ad_net(input)
    return adjusted_data, net


if __name__ == '__main__':
    base_path = f'./'
    pre_train_ex = f'rb_pre_without_model_ex6'
    # acc_data_filename = f'./datas/acc_data_{pre_train_ex}.npy'
    # train netword，get 0~10s y,z data, by x data
    trained_net = torch.load(f'{base_path}/model/model_{pre_train_ex}.pth')
    # params = torch.load(f'/home/cxz/rb_pre_without_model1/params/rb_pre_without_model_ex213.pt')
    ksi = torch.load(f'{base_path}/ksis/ksi_{pre_train_ex}.pt')
    Lambda = torch.load(f'./ksis/ksi_{pre_train_ex}_lambda.pt')

    # get system's Wout, predict next 1s
    dt = 0.025
    num_d = 3
    ridge_param = 1e-3
    warmup = 0
    traintime = 10
    testtime = 15
    maxtime = warmup + traintime + testtime
    current_time = warmup + traintime
    time_step = 0.5

    # data = lorenz_ode(time_span=(0, maxtime), ic=[-8, 7, 27], t_eval=np.linspace(0, maxtime, int(maxtime / dt) + 1), method='RK45')
    # acc_t = data.t
    # acc_data = data.y
    acc_t, acc_data = read_data(f'./datas/lorenz_sparse.csv')

    warmup_pts = round(warmup / dt)
    train_pts = round(traintime / dt)
    warmuptrain_pts = round((warmup + traintime) / dt)

    # trian_T = V(torch.from_numpy(acc_t[0: warmuptrain_pts + 1]).double().reshape((-1, 1)))
    # y_trained = V(torch.from_numpy(acc_data[:, 0:warmuptrain_pts + 1]).double()).T
    # theta = get_theta(y_train=y_trained, num_d=num_d)
    # old_theta = theta
    # theta[abs(theta) < 1e-3] = 0
    train_T_reg = V(torch.from_numpy(np.linspace(0, 10, 401)).reshape(-1, 1)).double()
    y_trained_reg = trained_net(train_T_reg)
    theta_reg = get_theta(y_train=y_trained_reg, num_d=num_d, ridge_param=ridge_param)

    T = train_T_reg
    predict_Y = y_trained_reg
    ad_Y = y_trained_reg
    before_ad_Y = y_trained_reg

    img_path = f'img{time_step}/'

    ex = 152
    alpha = 0.2
    times = int(15 / time_step)

    dydt = torch.tensor([])

    for i in range(times):
        gc.collect()

        pre_T = T[-num_d:]
        ad_T = T[-num_d:]
        pre_data = predict_Y[-num_d:, :]
        ad_data = ad_Y[-num_d:, :]
        pre_start = current_time + dt
        pre_step = int(time_step / dt)


        pre_T, predict_data = predict(theta=theta_reg, num_d=num_d, dt=dt, pre_t=pre_T, predict_data=pre_data, 
                                    pre_start=pre_start, pre_step=pre_step)
        
        pre_T, ad_data = predict(theta=theta_reg, num_d=num_d, dt=dt, pre_t=ad_T, predict_data=ad_data, 
                                    pre_start=pre_start, pre_step=pre_step)

        
        T = torch.cat((T, pre_T[num_d:, :]), axis=0)
        predict_Y = torch.cat((predict_Y, predict_data[num_d:, :]), axis=0)
        before_ad_Y = torch.cat((before_ad_Y, ad_data[num_d:, :]), axis=0)

        plot_pic2(path=img_path, ex=ex, acc_data_t=acc_t, acc_data=acc_data, net=trained_net, 
                 predict_t=T[train_pts:].detach().numpy(), predict_data=predict_Y[train_pts:, :].detach().numpy(), 
                 ad_t=T[train_pts: -pre_step].detach().numpy(), ad_data=ad_Y[train_pts:, :].detach().numpy(), 
                 before_ad_T=T[train_pts:].detach().numpy(), before_ad_data=before_ad_Y[train_pts:, :].detach().numpy())
        
        
        # use network adjust data
        pd_net_path = f'./model/model_predict_{ex}_{(current_time)}.pth'
        adjusted_data, net = adjust_data(pd_net_path=pd_net_path, num_d=num_d, alpha=alpha, input=pre_T, label=ad_data, ksi=ksi, Lambda=Lambda)

        u, u_t = jacobi(net, pre_T)
        dydt = torch.cat((dydt, u_t[num_d:]), axis=0)

        # update
        ad_Y = torch.cat((ad_Y, adjusted_data[num_d:, :]), axis=0)

        plot_pic2(path=img_path, ex=ex, acc_data_t=acc_t, acc_data=acc_data, net=trained_net, 
                 predict_t=T[train_pts:].detach().numpy(), predict_data=predict_Y[train_pts:, :].detach().numpy(), 
                 ad_t=T[train_pts:].detach().numpy(), ad_data=ad_Y[train_pts:, :].detach().numpy(), 
                 before_ad_T=T[train_pts:].detach().numpy(), before_ad_data=before_ad_Y[train_pts:, :].detach().numpy())
        
        current_time += time_step

        # theta = get_theta(y_train=ad_Y, num_d=num_d)

        torch.save(T, f'./datas/T_{ex}.pt')
        torch.save(predict_Y, f'./datas/predict_Y_{ex}.pt')
        torch.save(before_ad_Y, f'./datas/before_ad_Y_{ex}.pt')
        torch.save(ad_Y, f'./datas/ad_Y_{ex}.pt')
        torch.save(dydt, f'./datas/dydt_{ex}.pt')
        

