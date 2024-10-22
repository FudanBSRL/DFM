import os
import gc
import torch
import time
import numpy as np
from torch import nn
from predict import predict, get_theta
from torch.autograd import Variable as V
from model import Net
from plot_pic2 import plot_pic as plot_pic2
from data_util import read_data_ex
from discover import discover


def ad_train(pd_net_path, net, num_d, num_epoch, input, label, ksi, Lambda, device='cpu'):

    PT = torch.linspace(input[num_d - 1, 0], input[-1, 0], round((input[-1, 0].numpy() - input[num_d - 1, 0].numpy()) * 500)).reshape(-1, 1).double().to(device)

    input = input.detach().double().to(device)
    input.requires_grad = False
    label = label.detach().double().to(device)
    label.requires_grad = False
    ksi = ksi.detach().double().to(device)
    ksi.requires_grad = False

    IC_t = input[num_d - 1, :]
    IC_label = label[num_d - 1, :]

    t_input = input[num_d:, :]
    t_label = label[num_d:, :]

    mse_cost_function = torch.nn.MSELoss(reduction='mean')
    
    # optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-6, momentum=0.99)

    alpha = 0.005
    w_mse0 = 1e3
    w_msedu = 1e3
    # max_alpha = 0.500
    start_time = time.time()
    for epoch in range(num_epoch):

        # msed, u, _, _ = discover(net=net, T=PT, ksi=ksi, Lambda=Lambda)
        # mse0 = 0
        msed = 0
        msedu = 0
        y_hat = net(t_input)
        mseu = mse_cost_function(y_hat, t_label)

        
        y0 = net(IC_t)
        mse0 = mse_cost_function(y0, IC_label)
        # if epoch == 20000:
        #     optimizer.param_groups[0]['lr'] = 1e-5

        if epoch >= 20000:
            msed, u, u_t, _ = discover(net=net, T=PT, ksi=ksi, Lambda=Lambda, device=device)
            msedu = mse_cost_function(u[:, 2:], u_t[:, 0:2])
            # if 0 == (epoch + 1) % 200:
            #     alpha = min(alpha + 0.02, max_alpha)
            loss = (1 - alpha) * (mseu) + alpha * msed + w_mse0 * mse0 + w_msedu * msedu
        else:
            loss = mseu + mse0
            # loss = mseu + w_mse0 * mse0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # msed, u, u_t, _ = discover(net=net, T=PT, ksi=ksi, Lambda=Lambda, device=device)
        # msedu = mse_cost_function(u[:, 2:], u_t[:, 0:2])
        # loss = (1 - alpha) * mseu + alpha * msed + w_mse0 * mse0 + w_msedu * msedu
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        
        # loss = (1 - alpha) * mseu + alpha * msed + msedu
        # loss = (1 - alpha) * mseu + alpha * msed + msedu + 1e1 * mse0
        # loss = (1 - alpha) * mseu


        if (epoch + 1) % 100 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            start_time = time.time()
            # print(f'epoch {epoch + 1}, loss {loss:f}, mse0 {mse0:f}, mseu {mseu:f}, msed {msed:f}')
            print(f'epoch {epoch + 1}, loss {loss:f}, mse0 {mse0:f}, mseu {mseu:f}, msed {msed:f}, msedu {msedu:f}, time {elapsed_time}')

    # 定义优化器
    optimizer = torch.optim.LBFGS(net.parameters(), 
                                  lr=1e-1, 
                                  max_iter=4000,
                                  max_eval=4000,
                                  tolerance_grad=1.0 * np.finfo(float).eps,
                                  tolerance_change=1.0 * np.finfo(float).eps,
                                  history_size=50)
    
    iteration = 0
    # 定义一个闭包函数，用于L-BFGS的线搜索
    def closure():
        nonlocal iteration, alpha, start_time
        optimizer.zero_grad()  # 清除之前的梯度
        y0 = net(IC_t)
        mse0 = mse_cost_function(y0, IC_label)

        msed, u, u_t, _ = discover(net=net, T=PT, ksi=ksi, Lambda=Lambda, device=device)
        msedu = mse_cost_function(u[:, 2:], u_t[:, 0:2])
        # msed = 0
        y_hat = net(t_input)
        mseu = mse_cost_function(y_hat, t_label)
        # loss = (1 - alpha) * mseu + alpha * msed + msedu
        # loss = (1 - alpha) * mseu + alpha * msed + msedu + 1e1 * mse0
        loss = (1 - alpha) * mseu + alpha * msed + w_mse0 * mse0 + w_msedu * msedu
        loss.backward()  # 进行反向传播
        iteration += 1
        if (iteration) % 100 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            start_time = time.time()
            # print(f'Iteration {iteration}, mseu {mseu:f}, msed {msed:f}')
            print(f'Iteration {iteration}, mse0 {mse0:f}, mseu {mseu:f}, msed {msed:f}, msedu {msedu:f}, time {elapsed_time}')
        return loss
    
    # 执行优化步骤
    optimizer.step(closure)

    return net


def adjust_data(pd_net_path, num_d, input, label, ksi, Lambda):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'

    if os.path.exists(pd_net_path):
        ad_net = torch.load(pd_net_path)
    else:
        ad_net = Net(width=256, deep=5, in_num=1, out_num=4).double().to(device)
        for m in ad_net.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)


    # print(round((input[-1, 0].numpy() - input[0, 0].numpy()) * 200))
    # print(input)

    input = input - (input[0, 0] + input[-1, 0]) / 2
    net = ad_train(pd_net_path=pd_net_path, net=ad_net, num_d=num_d, num_epoch=21000, input=input, label=label, ksi=ksi,
                   Lambda=Lambda, device=device)

    adjusted_data = ad_net(input).double()
    return adjusted_data, net


if __name__ == '__main__':
    base_path = f'./'
    pre_train_ex = f'rb_pre_without_model_ex4'
    # acc_data_filename = f'./datas/acc_data_{pre_train_ex}.npy'
    # train netword，get 0~10s y,z data, by x data
    trained_net = torch.load(f'{base_path}/model/model_{pre_train_ex}.pth')
    # params = torch.load(f'/home/cxz/rb_pre_without_model1/params/rb_pre_without_model_ex213.pt')
    ksi = torch.load(f'{base_path}/ksis/ksi_{pre_train_ex}.pt')
    Lambda = torch.load(f'./ksis/ksi_{pre_train_ex}_lambda.pt')

    # get system's Wout, predict next 1s
    dt = 0.01
    num_d = 15
    start_train = 0
    traintime = 20
    testtime = 5
    current_time = 21
    maxtime = current_time + testtime
    time_step = 0.1

    ridge_param = 1e-4
    
    xlim = (current_time, maxtime)

    acc_t, acc_data, _, _ = read_data_ex('./csvs/ex_data1.CSV', rate=round(1/dt), filter=False)

    start_train_pt = round(start_train / dt)
    train_pts = round(traintime / dt)

    trian_T = V(torch.from_numpy(acc_t[: train_pts + 1]).double().reshape((-1, 1)))
    T = trian_T

    y_trained = V(torch.from_numpy(acc_data[:, start_train_pt:train_pts + 1]).double()).T
    theta = get_theta(y_train=y_trained, num_d=num_d, ridge_param=ridge_param)
    predict_Y = V(torch.from_numpy(acc_data[:, :train_pts + 1]).double()).T

    start_predict_pts = round(current_time / dt)
    T = V(torch.from_numpy(acc_t[: start_predict_pts + 1]).double().reshape((-1, 1)))
    predict_Y = V(torch.from_numpy(acc_data[:, :start_predict_pts + 1]).double()).T

    # y_trained_reg = V(trained_net(torch.from_numpy(acc_t[start_train_pt:train_pts + 1]).float().reshape(-1, 1) - 15).double().detach())
    # theta_reg = get_theta(y_train=y_trained_reg, num_d=num_d, ridge_param=ridge_param)
    # theta = theta_reg
    # predict_Y = V(trained_net(torch.from_numpy(acc_t[:train_pts + 1]).float().reshape(-1, 1) - 15)).double()


    # T = trian_T
    # predict_Y = trained_Y
    ad_Y = predict_Y
    before_ad_Y = predict_Y

    img_path = f'img{time_step}/'
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    ex = 25_01_0005
    times = int(testtime / time_step)

    # dydt = torch.tensor([])

    for i in range(times):
        gc.collect()

        pre_T = T[-num_d:]
        ad_T = T[-num_d:]
        pre_data = predict_Y[-num_d:, :]
        ad_data = ad_Y[-num_d:, :]
        pre_start = current_time + dt
        pre_step = int(time_step / dt)


        pre_T, predict_data = predict(theta=theta, num_d=num_d, dt=dt, pre_t=pre_T, predict_data=pre_data, 
                                    pre_start=pre_start, pre_step=pre_step)
        
        pre_T, ad_data = predict(theta=theta, num_d=num_d, dt=dt, pre_t=ad_T, predict_data=ad_data, 
                                    pre_start=pre_start, pre_step=pre_step)

        
        T = torch.cat((T, pre_T[num_d:, :]), dim=0)
        predict_Y = torch.cat((predict_Y, predict_data[num_d:, :]), dim=0)
        before_ad_Y = torch.cat((before_ad_Y, ad_data[num_d:, :]), dim=0)

        plot_pic2(path=img_path, ex=ex, acc_data_t=acc_t, acc_data=acc_data, net=trained_net, 
                 predict_t=T[start_predict_pts:].detach().numpy(), predict_data=predict_Y[start_predict_pts:, :].detach().numpy(), 
                 ad_t=T[start_predict_pts: -pre_step].detach().numpy(), ad_data=ad_Y[start_predict_pts:, :].detach().numpy(), 
                 before_ad_T=T[start_predict_pts:].detach().numpy(), before_ad_data=before_ad_Y[start_predict_pts:, :].detach().numpy(),
                 xlim=xlim)
        
        
        # use network adjust data
        pd_net_path = f'./model/model_predict_{ex}_{current_time}.pth'
        adjusted_data, net = adjust_data(pd_net_path=pd_net_path, num_d=num_d, input=pre_T, label=ad_data, ksi=ksi, Lambda=Lambda)

        # u, u_t = jacobi(net, pre_T)
        # dydt = torch.cat((dydt, u_t[num_d:]), axis=0)

        # update
        ad_Y = torch.cat((ad_Y, adjusted_data[num_d:, :]), dim=0)

        plot_pic2(path=img_path, ex=ex, acc_data_t=acc_t, acc_data=acc_data, net=trained_net, 
                 predict_t=T[train_pts:].detach().numpy(), predict_data=predict_Y[train_pts:, :].detach().numpy(), 
                 ad_t=T[train_pts:].detach().numpy(), ad_data=ad_Y[train_pts:, :].detach().numpy(), 
                 before_ad_T=T[train_pts:].detach().numpy(), before_ad_data=before_ad_Y[train_pts:, :].detach().numpy(),
                 xlim=xlim)
        
        current_time += time_step

        # theta = get_theta(y_train=ad_Y, num_d=num_d)

        torch.save(T, f'./datas/T_{ex}.pt')
        torch.save(predict_Y, f'./datas/predict_Y_{ex}.pt')
        torch.save(before_ad_Y, f'./datas/before_ad_Y_{ex}.pt')
        torch.save(ad_Y, f'./datas/ad_Y_{ex}.pt')
        # torch.save(dydt, f'./datas/dydt_{ex}.pt')
        

