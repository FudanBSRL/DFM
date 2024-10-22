import os
import torch
import math
import sys
import time
import numpy as np
from torch import nn
from torch.autograd import Variable as V
from model import Net
from plot_pic import plot_pic
from data_util import read_data_ex
from discover import discover, num_candidates, init_ksi_data, jacobi

torch.pi = math.pi


class pinn:
    def __init__(self, device='cpu', reset=False) -> None:

        torch.set_printoptions(precision=4, sci_mode=False)

        self.gpinn = False
        self.last_epoch = 0
        self.ex = 'rb_pre_without_model_ex4'

        self.dof = 4

        self.e = 5e-3

        self.num_epochs = 50000000
        self.device = device
        
        self.net_width = 256
        self.net_deep = 5

        self.model_filename = f'./model/model_{self.ex}.pth'
        self.data_filename = f'./csvs/ex_data1.CSV'
        self.ksi_filename = f'./ksis/ksi_{self.ex}.pt'
        self.acc_data_filename = f'./datas/acc_data_{self.ex}.npy'
        self.log_dirname = f'./logs3'
        self.Lambda_filename = f'./ksis/ksi_{self.ex}_lambda.pt'

        self.plot_epoch = 100
        self.decay_epoch = 120000
        
        self.para_w = 0.1
        self.milestones = []

        self.mse_cost_function = torch.nn.MSELoss(reduction='mean')
        
        if not reset and os.path.exists(self.model_filename):
            self.net = torch.load(self.model_filename)
        else:
            self.net = Net(width=self.net_width, deep=self.net_deep, in_num=1, out_num=self.dof).float()
            for m in self.net.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)
                    
            torch.save(self.net, self.model_filename)
        
        self.net.to(device)
        self.ksi, self.Lambda = self.init_ksi(reset)

        self.init_data()

    
    def update_lambda(self, th):
        self.Lambda[torch.abs(self.ksi) < th] = 0
        self.ksi.data = self.ksi * self.Lambda
        torch.save(self.ksi, self.ksi_filename)
        torch.save(self.Lambda, self.Lambda_filename)
        print(self.Lambda)
        print(self.ksi)
    

    def init_data(self):
        self.t, self.acc_data = read_data_ex(self.data_filename, rate=100)
        start = 1000
        r = 2001
        train_time = self.t[start:r]
        train_time = train_time - 15
        # self.train_time = V(torch.from_numpy(train_time - (train_time[0] + train_time[-1]) / 2).double()).reshape(-1, 1)
        self.train_time = V(torch.from_numpy(train_time).float()).reshape(-1, 1).to(self.device)
        self.train_label = V(torch.from_numpy(self.acc_data[:, start:r]).float()).T.to(self.device)

    
    def init_ksi(self, reset=False):
        if not reset and os.path.exists(self.ksi_filename):
            ksi = torch.load(self.ksi_filename)
        else:
            ksi = torch.zeros((num_candidates(self.dof), self.dof)).float()
            torch.save(ksi, self.ksi_filename)

        ksi = ksi.detach()
        ksi = ksi.to(self.device)
        ksi.requires_grad = True

        if not reset and os.path.exists(self.Lambda_filename):
            Lambda = torch.load(self.Lambda_filename)
        else:
            Lambda = torch.ones(ksi.shape)
            torch.save(Lambda, self.Lambda_filename)

        Lambda = Lambda.detach()
        Lambda = Lambda.to(self.device)
        Lambda.requires_grad = False
        return ksi, Lambda


    def init_ksi_LS(self):
        self.ksi.data = init_ksi_data(self.net, self.train_time)
        print(self.ksi)
        torch.save(self.ksi, self.ksi_filename)


    def train_ksi(self, lr, epoch_num, cut, th, tn_lb_num=1000):
        
        optimizer = torch.optim.Adam([{'params': self.ksi, 'lr': lr}])

        msed, u, u_t, cand = discover(net=self.net, T=self.train_time, ksi = self.ksi, Lambda=self.Lambda, train=True)
        u = u.detach()
        u_t = u_t.detach()
        cand = cand.detach()

        for epoch in range(epoch_num):

            optimizer.zero_grad()
            
            # msed, _, _ = discover(net=self.net, T=self.train_time, ksi = self.ksi, Lambda=self.Lambda, train=True)
            msed = torch.mean((u_t - cand @ (self.ksi)) ** 2)
            l1 = torch.mean(torch.abs(self.ksi))
            loss = msed + 1e-5 * l1

            loss.backward()
            optimizer.step()
            
            if cut:
                with torch.no_grad():
                    self.ksi[torch.abs(self.ksi) < th] = 0

            if (epoch + 1) % self.plot_epoch == 0:
                print(f'epoch {epoch + 1}, loss {loss:f}, msed {msed:f}')

            if (epoch + 1) % (10 * self.plot_epoch) == 0:
                print(self.ksi)
                torch.save(self.ksi, self.ksi_filename)

        
        if (epoch_num) % (10 * self.plot_epoch) != 0:
            print(self.ksi)
            torch.save(self.ksi, self.ksi_filename)

        # # 定义优化器
        # optimizer = torch.optim.LBFGS([self.ksi],
        #                               lr=1,
        #                               max_iter=tn_lb_num,
        #                               max_eval=tn_lb_num,
        #                               tolerance_grad=1.0 * np.finfo(float).eps,
        #                               tolerance_change=1.0 * np.finfo(float).eps,
        #                               history_size=50)
        #
        # iteration = 0
        #
        # # 定义一个闭包函数，用于L-BFGS的线搜索
        # def closure():
        #     nonlocal iteration
        #     optimizer.zero_grad()  # 清除之前的梯度
        #     msed = torch.mean(1 / 2 * (u_t - cand @ (self.ksi)) ** 2)
        #     l1 = torch.mean(torch.abs(self.ksi))
        #     loss = msed + 1e-5 * l1
        #     loss.backward()  # 进行反向传播
        #     iteration += 1
        #     if (iteration) % 100 == 0:
        #         print(f'Iteration {iteration}, loss {loss:f}, msed {msed:f}')
        #     return loss
        #
        # # 执行优化步骤
        # optimizer.step(closure)
        # if cut:
        #     with torch.no_grad():
        #         self.ksi[torch.abs(self.ksi) < th] = 0
        # torch.save(self.ksi, self.ksi_filename)


    def train_net(self, lr, epoch_num, tn_lb_num, alpha1, alpha2):
        
        optimizer = torch.optim.Adam([{'params': self.net.parameters(), 'lr': lr}])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2000, T_mult=2)
        
        for epoch in range(epoch_num):

            optimizer.zero_grad()
            
            # PT = np.random.uniform(low=0, high=10, size=(32, 1))
            # PT = V(torch.from_numpy(PT).double().reshape(-1, 1)).to(self.device)
            msed, u, u_t, _ = discover(net=self.net, T=self.train_time, ksi=self.ksi, Lambda=self.Lambda, train=True)
            # u = self.net(T)
            # mseu1 = self.mse_cost_function(u[:, 2:], self.train_label[:, 2:])
            # mseu2 = self.mse_cost_function(u[:, :2], self.train_label[:, :2])
            # mseu = 0.999 * mseu1 + 0.001 * mseu2
            mseu = self.mse_cost_function(u, self.train_label)
            msedu = self.mse_cost_function(u[:, 2:], u_t[:, :2])
            loss = mseu + alpha2 * msedu + alpha1 * msed

            loss.backward()
            optimizer.step()
            # scheduler.step()

            if (epoch + 1) % self.plot_epoch == 0:
                print(f'epoch {epoch + 1}, loss {loss:f}, mseu {mseu:f}, msedu {msedu:f}, msed {msed:f}')

            if (epoch + 1) % (10 * self.plot_epoch) == 0:
                plot_pic(ex='pretrain', acc_data_t=self.train_time, acc_data=self.train_label, net=self.net)
                torch.save(self.net, self.model_filename)

        if (epoch_num) % (10 * self.plot_epoch) != 0:
            torch.save(self.net, self.model_filename)

        # 定义优化器
        optimizer = torch.optim.LBFGS(self.net.parameters(), 
                                    lr=1, 
                                    max_iter=tn_lb_num, 
                                    max_eval=tn_lb_num,
                                    tolerance_grad=1.0 * np.finfo(float).eps,
                                    tolerance_change=1.0 * np.finfo(float).eps,
                                    history_size=50)
        
        iteration = 0
        # 定义一个闭包函数，用于L-BFGS的线搜索
        def closure():
            nonlocal iteration, alpha1, alpha2
            optimizer.zero_grad()  # 清除之前的梯度
            msed, u, u_t, _ = discover(net=self.net, T=self.train_time, ksi=self.ksi, Lambda=self.Lambda, train=True)
            # u = self.net(T)
            # mseu1 = self.mse_cost_function(u[:, 2:], self.train_label[:, 2:])
            # mseu2 = self.mse_cost_function(u[:, :2], self.train_label[:, :2])
            # mseu = 0.999 * mseu1 + 0.001 * mseu2
            mseu = self.mse_cost_function(u, self.train_label)
            msedu = self.mse_cost_function(u[:, 2:], u_t[:, :2])
            loss = mseu + alpha2 * msedu + alpha1 * msed
            loss.backward()  # 进行反向传播
            iteration += 1
            if (iteration) % 100 == 0:
                print(f'Iteration {iteration}, mseu {mseu:f}, msedu {msedu:f}, msed {msed:f}')
            return loss
        
        # 执行优化步骤
        optimizer.step(closure)
        plot_pic(ex='pretrain', acc_data_t=self.train_time, acc_data=self.train_label, net=self.net)
        torch.save(self.net, self.model_filename)


    def train_regression(self, lr, epoch_num, lb_num):
        optimizer = torch.optim.Adam([{'params': self.net.parameters(), 'lr': lr}])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epoch_num, T_mult=2)

        for epoch in range(epoch_num):

            optimizer.zero_grad()
            
            # y_hat = self.net(self.train_time)
            u, u_t = jacobi(self.net, self.train_time, device=device)
            mseu = self.mse_cost_function(u, self.train_label)
            msedu = self.mse_cost_function(u[:, 2:], u_t[:, :2])
            
            loss = mseu + msedu

            loss.backward()
            optimizer.step()
            # scheduler.step()

            if (epoch + 1) % self.plot_epoch == 0:
                print(f'epoch {epoch + 1}, loss {loss}, mseu {mseu}, msedu {msedu}')
                torch.save(self.net, self.model_filename)

            if (epoch + 1) % (10 * self.plot_epoch) == 0:
                plot_pic(ex='pretrain', acc_data_t=self.train_time, acc_data=self.train_label, net=self.net)

        # 定义优化器
        optimizer = torch.optim.LBFGS(self.net.parameters(), 
                                    lr=1, 
                                    max_iter=lb_num, 
                                    max_eval=lb_num,
                                    tolerance_grad=1.0 * np.finfo(float).eps,
                                    tolerance_change=1.0 * np.finfo(float).eps,
                                    history_size=50)
        
        iteration = 0
        # 定义一个闭包函数，用于L-BFGS的线搜索
        def closure():
            nonlocal iteration
            optimizer.zero_grad()  # 清除之前的梯度
            u, u_t = jacobi(self.net, self.train_time, device=device)
            mseu = self.mse_cost_function(u, self.train_label)
            msedu = self.mse_cost_function(u[:, 2:], u_t[:, :2])
            
            loss = mseu + msedu
            loss.backward()  # 进行反向传播
            iteration += 1
            if (iteration) % 100 == 0:
                print(f'Iteration {iteration}, loss {loss}, mseu {mseu}, msedu {msedu}')
            return loss
        
        # 执行优化步骤
        optimizer.step(closure)
        plot_pic(ex='pretrain', acc_data_t=self.train_time, acc_data=self.train_label, net=self.net)
        torch.save(self.net, self.model_filename)


    def train(self, lr1, lr2, epoch_num, w1, w2, w3, w4, cut, th):
        optimizer = torch.optim.Adam([{'params': self.net.parameters(), 'lr': lr1},
                                      {'params': self.ksi, 'lr': lr2}])

        start_time = time.time()
        for epoch in range(epoch_num):

            optimizer.zero_grad()
            
            msed, u, u_t, _ = discover(net=self.net, T=self.train_time, ksi=self.ksi, Lambda=self.Lambda, train=True, device=self.device)
            msedu = self.mse_cost_function(u[:, 2:], u_t[:, 0:2])
            mseu = self.mse_cost_function(u, self.train_label)

            l1 = torch.mean(torch.abs(self.ksi))
            loss = w1 * msed + w2 * mseu + w3 * msedu + w4 * l1

            loss.backward()
            optimizer.step()

            if (epoch + 1) % self.plot_epoch == 0:
                end_time = time.time()
                elapsed_time = end_time - start_time
                start_time = time.time()
                print(f'epoch {epoch + 1}, loss {loss:f}, msed {msed}, mseu {mseu}, msedu {msedu}, time {elapsed_time}')
                torch.save(self.net, self.model_filename)

            if (epoch + 1) % (10 * self.plot_epoch) == 0:
                plot_pic(ex='pretrain', acc_data_t=self.train_time, acc_data=self.train_label, net=self.net)
                if cut:
                    with torch.no_grad():
                        self.ksi[torch.abs(self.ksi) < th] = 0
                print(self.ksi)
                torch.save(self.ksi, self.ksi_filename)


    def pre_train(self, lr1, lr2, epoch_num, lb_num=1000, w1=1e-3, w2=1, w3=1e-5, w4=1e-5, cut=False, th=1e-4):
        train.train(lr1=lr1, lr2=lr2, epoch_num=epoch_num, w1=w1, w2=w2, w3=w3, w4=w4, cut=cut, th=th)
        # train.train_regression(lr=lr, epoch_num=epoch_num, lb_num=lb_num)
        # train.init_ksi_LS()


    def ado(self, pos, ado_num, tk_lr, tn_lr, tk_num, tn_num, tn_lb_num, cut, th, alpha1, alpha2):
        for i in range(ado_num):
            train.train_ksi(lr=tk_lr, epoch_num=tk_num, cut=cut, th=th, tn_lb_num=10000)
            print(f'==={pos}, {i} ===')
            train.train_net(lr=tn_lr, epoch_num=tn_num, tn_lb_num=tn_lb_num, alpha1=alpha1, alpha2=alpha2)
                

if __name__ == '__main__':
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print(f'run on {device}')

    if len(sys.argv) >= 2 and sys.argv[1] == 'new':
        train = pinn(device, reset=True)
    else:
        train = pinn(device)


    # train.train_regression(lr=1e-3, epoch_num=10000, lb_num=0)
    
    # train.train_ksi(lr=1e-6, epoch_num=300000, cut=False, th=1e-4, tn_lb_num=10000)
    # #
    # train.pre_train(lr1=1e-4, lr2=1e-5, epoch_num=20000, lb_num=10000, w1=1e-5, w3=1e-2)
    # train.pre_train(lr1=1e-4, lr2=1e-5, epoch_num=20000, lb_num=10000, w1=1e-5, w3=1e-1)

    
    ado_num = 10
    tk_num = 100000
    tn_num = 1000
    tn_lb_num = 1000
    # train.ado(pos=1, ado_num=ado_num, tk_lr=1e-6, tn_lr=1e-6, tk_num=tk_num, tn_num=tn_num, tn_lb_num=tn_lb_num,
    #           cut=False, th=1e-6, alpha1=1e-5, alpha2=1)
    # train.ado(pos=2, ado_num=ado_num, tk_lr=1e-6, tn_lr=1e-6, tk_num=tk_num, tn_num=tn_num, tn_lb_num=tn_lb_num,
    #           cut=True, th=1e-3, alpha1=1e-2, alpha2=1)
    # train.ado(pos=3, ado_num=ado_num, tk_lr=1e-6, tn_lr=1e-6, tk_num=tk_num, tn_num=tn_num, tn_lb_num=tn_lb_num,
            #   cut=True, th=1e-3, alpha1=1e-1, alpha2=1)
    # train.ado(pos=4, ado_num=ado_num, tk_lr=1e-6, tn_lr=1e-6, tk_num=tk_num, tn_num=tn_num, tn_lb_num=tn_lb_num,
    #           cut=True, th=1e-3, alpha1=1e0, alpha2=1)
    
    ado_num = 10
    tk_num = 100000
    tn_num = 1000
    tn_lb_num = 1000
    # train.ado(pos=5, ado_num=ado_num, tk_lr=1e-6, tn_lr=1e-6, tk_num=tk_num, tn_num=tn_num, tn_lb_num=tn_lb_num,
    #           cut=True, th=1e-3, alpha1=1e0, alpha2=1e1)
    # train.ado(pos=6, ado_num=ado_num, tk_lr=1e-6, tn_lr=1e-6, tk_num=tk_num, tn_num=tn_num, tn_lb_num=tn_lb_num,
    #           cut=True, th=1e-3, alpha1=1e1, alpha2=1e1)

    # train.ado(pos=7, ado_num=ado_num, tk_lr=1e-6, tn_lr=1e-6, tk_num=tk_num, tn_num=tn_num, tn_lb_num=tn_lb_num,
    #           cut=False, th=1e-3, alpha1=1e1, alpha2=1e1)

    train.ado(pos=8, ado_num=ado_num, tk_lr=1e-6, tn_lr=1e-6, tk_num=tk_num, tn_num=tn_num, tn_lb_num=tn_lb_num,
              cut=True, th=1e-3, alpha1=1e1, alpha2=1e1)
    