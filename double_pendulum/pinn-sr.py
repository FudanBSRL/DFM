import os
import torch
import math
import random
import sys
import time
import numpy as np
from torch import nn
from torch.autograd import Variable as V
from model import Net
from plot_pic import plot_pic
from data_util import read_data
from discover import discover_gpinn, discover, num_candidates, init_ksi_data

torch.pi = math.pi


class pinn:
    def __init__(self, device='cpu', reset=False) -> None:

        torch.set_printoptions(precision=4, sci_mode=False)

        self.gpinn = False
        self.last_epoch = 0
        self.ex = 'rb_pre_without_model_ex1'

        self.dof = 4

        self.e = 5e-3

        self.num_epochs = 50000000
        self.device = device
        
        self.net_width = 256
        self.net_deep = 5

        self.model_filename = f'./model/model_{self.ex}.pth'
        self.data_filename = f'./csvs/d_pendulum.csv'
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
            self.net = Net(width=self.net_width, deep=self.net_deep, in_num=1, out_num=self.dof)
            self.net.double()
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
        self.t, self.acc_data = read_data(self.data_filename)
        r = 801
        train_time = self.t[0:r]
        train_time = train_time - 10
        # self.train_time = V(torch.from_numpy(train_time - (train_time[0] + train_time[-1]) / 2).double()).reshape(-1, 1)
        self.train_time = V(torch.from_numpy(train_time).double()).reshape(-1, 1).to(self.device)
        self.train_label = V(torch.from_numpy(self.acc_data[:, 0:r]).double()).T.to(self.device)

    
    def init_ksi(self, reset=False):
        if not reset and os.path.exists(self.ksi_filename):
            ksi = torch.load(self.ksi_filename)
        else:
            ksi = torch.zeros((num_candidates(self.dof), self.dof)).double()
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


    def train_ksi(self, lr, epoch_num, cut, th):
        
        optimizer = torch.optim.Adam([{'params': self.ksi, 'lr': lr}])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2000, T_mult=2)

        for epoch in range(epoch_num):

            optimizer.zero_grad()
            
            msed, _, _ = discover(net=self.net, T=self.train_time, ksi = self.ksi, Lambda=self.Lambda)

            l1 = torch.mean(torch.abs(self.ksi))
            loss = msed + 1e-5 * l1

            loss.backward()
            optimizer.step()
            # scheduler.step()
            
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


    def train_net(self, lr, epoch_num, tn_lb_num, alpha):
        
        optimizer = torch.optim.Adam([{'params': self.net.parameters(), 'lr': lr}])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2000, T_mult=2)
        
        for epoch in range(epoch_num):

            optimizer.zero_grad()
            
            # PT = np.random.uniform(low=0, high=10, size=(32, 1))
            # PT = V(torch.from_numpy(PT).double().reshape(-1, 1)).to(self.device)
            msed, u, _ = discover(net=self.net, T=self.train_time, ksi=self.ksi, Lambda=self.Lambda)
            # u = self.net(T)
            mseu = self.mse_cost_function(u, self.train_label)
            loss = mseu + alpha * msed

            loss.backward()
            optimizer.step()
            # scheduler.step()

            if (epoch + 1) % self.plot_epoch == 0:
                print(f'epoch {epoch + 1}, loss {loss:f}, mseu {mseu:f}, msed {msed:f}')

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
            nonlocal iteration, alpha
            optimizer.zero_grad()  # 清除之前的梯度
            msed, u, _ = discover(net=self.net, T=self.train_time, ksi=self.ksi, Lambda=self.Lambda)
            # u = self.net(T)
            mseu = self.mse_cost_function(u, self.train_label)
            loss = mseu + alpha * msed
            loss.backward()  # 进行反向传播
            iteration += 1
            if (iteration) % 100 == 0:
                print(f'Iteration {iteration}, mseu {mseu:f}, msed {msed:f}')
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
            
            y_hat = self.net(self.train_time)
            loss = self.mse_cost_function(y_hat, self.train_label)

            loss.backward()
            optimizer.step()
            # scheduler.step()

            if (epoch + 1) % self.plot_epoch == 0:
                print(f'epoch {epoch + 1}, loss {loss}')
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
            y_hat = self.net(self.train_time)
            mseu = self.mse_cost_function(y_hat, self.train_label)
            loss = 1 * mseu
            loss.backward()  # 进行反向传播
            iteration += 1
            if (iteration) % 100 == 0:
                print(f'Iteration {iteration}, mseu {mseu:f}')
            return loss
        
        # 执行优化步骤
        optimizer.step(closure)
        plot_pic(ex='pretrain', acc_data_t=self.train_time, acc_data=self.train_label, net=self.net)
        torch.save(self.net, self.model_filename)


    def train(self, pos, lr1, lr2, epoch_num, w1, w2, w3, cut, th):
        optimizer = torch.optim.Adam([{'params': self.net.parameters(), 'lr': lr1},
                                      {'params': self.ksi, 'lr': lr2}])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2000, T_mult=2)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.para_w, last_epoch=-1)

        XYZ = V(torch.from_numpy(self.acc_data[:, 0:401]).double()).to(self.device).T
        PT = V(torch.from_numpy(self.t[0:401]).double()).reshape(-1, 1).to(self.device)

        for epoch in range(epoch_num):

            # PT = np.random.uniform(low=0, high=self.current_train, size=(16, 1))
            # PT = V(torch.from_numpy(PT).double().reshape((-1, 1))).to(self.device)

            optimizer.zero_grad()
            
            msed, u, u_t = discover(net=self.net, T=PT, ksi=self.ksi, Lambda=self.Lambda)
            # mseu = self.u_loss()
            mseu = self.mse_cost_function(u, XYZ)

            l1 = torch.mean(torch.abs(self.ksi))
            loss = w1 * msed + w2 * mseu + w3 * l1

            loss.backward()
            optimizer.step()
            # scheduler.step()

            if (epoch) % self.plot_epoch == 0:
                print(f'pos {pos}, epoch {epoch}, loss {loss:f}, msed {msed}, mseu {mseu}')
                torch.save(self.net, self.model_filename)

            if (epoch - 1) % (10 * self.plot_epoch) == 0:
                plot_pic(ex='pretrain', acc_data_t=self.t, acc_data=self.acc_data, net=self.net)
                if cut:
                    with torch.no_grad():
                        self.ksi[torch.abs(self.ksi) < th] = 0
                print(self.ksi)
                torch.save(self.ksi, self.ksi_filename)


    def pre_train(self, lr, epoch_num, lb_num):
        train.train_regression(lr=lr, epoch_num=epoch_num, lb_num=lb_num)
        # train.init_ksi_LS()


    def ado(self, pos, ado_num, tk_lr, tn_lr, tk_num, tn_num, tn_lb_num, cut, th, alpha):
        for i in range(ado_num):
            print(f'==={pos}, {i} ===')
            train.train_ksi(lr=tk_lr, epoch_num=tk_num, cut=cut, th=th)
            train.train_net(lr=tn_lr, epoch_num=tn_num, tn_lb_num=tn_lb_num, alpha=alpha)
                

if __name__ == '__main__':
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print(f'run on {device}')

    if len(sys.argv) >= 2 and sys.argv[1] == 'new':
        train = pinn(device, reset=True)
    else:
        train = pinn(device)


    # train.pre_train(lr=1e-3, epoch_num=5000, lb_num=5000)
    # train.ado(pos=1, ado_num=10, tk_lr=1e-2, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=False, th=1e-2, alpha=0.1)
    # train.ado(pos=2, ado_num=100, tk_lr=1e-2, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=1e-2, alpha=0.1)
    # train.ado(pos=3, ado_num=100, tk_lr=1e-2, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=5e-2, alpha=0.1)
    # train.ado(pos=4, ado_num=100, tk_lr=1e-2, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=1e-1, alpha=0.1)
    # train.ado(pos=5, ado_num=100, tk_lr=1e-2, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=2e-1, alpha=0.1)
    # train.ado(pos=6, ado_num=100, tk_lr=1e-2, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=3e-1, alpha=0.1)
    # train.ado(pos=7, ado_num=100, tk_lr=1e-2, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=4e-1, alpha=0.1)
    # train.ado(pos=8, ado_num=100, tk_lr=1e-2, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=5e-1, alpha=0.1)
    # train.ado(pos=9, ado_num=100, tk_lr=1e-2, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=False, th=1e-4, alpha=0.1)
    # train.ado(pos=10, ado_num=100, tk_lr=1e-2, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=1e-3, alpha=0.1)
    # train.ado(pos=11, ado_num=100, tk_lr=1e-2, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=1e-2, alpha=0.1)

    # train.pre_train(lr=1e-3, epoch_num=5000, lb_num=5000)
    # train.ado(pos=12, ado_num=50, tk_lr=1e-2, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=False, th=1e-4, alpha=0.1)
    # train.ado(pos=13, ado_num=100, tk_lr=1e-2, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=1e-3, alpha=0.1)
    # train.ado(pos=14, ado_num=100, tk_lr=1e-2, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=1e-2, alpha=0.1)
    # train.ado(pos=15, ado_num=100, tk_lr=1e-2, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=2e-2, alpha=0.1)
    # train.ado(pos=16, ado_num=100, tk_lr=1e-2, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=3e-2, alpha=0.1)
    # train.ado(pos=17, ado_num=100, tk_lr=1e-2, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=5e-2, alpha=0.1)
    # train.ado(pos=18, ado_num=100, tk_lr=1e-2, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=1e-1, alpha=0.1)
    # train.ado(pos=19, ado_num=100, tk_lr=1e-2, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=5e-1, alpha=0.1)
    # train.ado(pos=20, ado_num=100, tk_lr=1e-3, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=False, th=1e-4, alpha=0.1)
    # train.ado(pos=21, ado_num=100, tk_lr=1e-3, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=1e-3, alpha=0.1)
    # train.ado(pos=22, ado_num=100, tk_lr=1e-3, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=1e-2, alpha=0.1)
    
    # train.pre_train(lr=1e-3, epoch_num=5000, lb_num=5000)
    # train.ado(pos=23, ado_num=50, tk_lr=1e-3, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=False, th=1e-4, alpha=0.1)
    # train.ado(pos=24, ado_num=100, tk_lr=1e-3, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=1e-3, alpha=0.1)
    # train.ado(pos=25, ado_num=100, tk_lr=1e-3, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=1e-2, alpha=0.1)
    # train.ado(pos=26, ado_num=100, tk_lr=1e-3, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=5e-2, alpha=0.1)
    # train.ado(pos=27, ado_num=100, tk_lr=1e-3, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=1e-1, alpha=0.1)
    # train.ado(pos=28, ado_num=100, tk_lr=1e-3, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=2e-1, alpha=0.1)
    # train.ado(pos=29, ado_num=100, tk_lr=1e-3, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=2e-1, alpha=0.5)
    # train.ado(pos=30, ado_num=20, tk_lr=1e-3, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=False, th=2e-1, alpha=0.5)
    # train.ado(pos=31, ado_num=100, tk_lr=1e-3, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=1e-1, alpha=0.5)
    # train.ado(pos=32, ado_num=20, tk_lr=1e-3, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=False, th=1e-1, alpha=0.5)
    # train.ado(pos=33, ado_num=100, tk_lr=1e-3, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=5e-2, alpha=0.5)
    # train.ado(pos=34, ado_num=20, tk_lr=1e-3, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=False, th=1e-1, alpha=0.5)
    # train.ado(pos=35, ado_num=100, tk_lr=1e-3, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=5e-2, alpha=0.5)
    # train.ado(pos=36, ado_num=20, tk_lr=1e-3, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=False, th=1e-1, alpha=0.5)
    # train.ado(pos=37, ado_num=100, tk_lr=1e-3, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=5e-2, alpha=0.5)
    # train.ado(pos=38, ado_num=20, tk_lr=1e-3, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=False, th=1e-1, alpha=0.5)
    # train.ado(pos=39, ado_num=100, tk_lr=1e-3, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=1e-2, alpha=0.5)
    # train.ado(pos=40, ado_num=20, tk_lr=1e-3, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=False, th=1e-1, alpha=0.5)
    # train.ado(pos=41, ado_num=100, tk_lr=1e-3, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=1e-2, alpha=0.5)
    # train.ado(pos=42, ado_num=20, tk_lr=1e-3, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=False, th=1e-1, alpha=0.5)
    # train.ado(pos=43, ado_num=100, tk_lr=1e-3, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=5e-2, alpha=0.5)
    # train.ado(pos=44, ado_num=20, tk_lr=1e-3, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=False, th=1e-1, alpha=0.5)
    # train.ado(pos=45, ado_num=100, tk_lr=1e-3, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=5e-2, alpha=0.5)
    train.ado(pos=46, ado_num=20, tk_lr=1e-3, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=False, th=1e-1, alpha=0.5)
    train.ado(pos=47, ado_num=100, tk_lr=1e-3, tn_lr=1e-5, tk_num=1000, tn_num=1000, tn_lb_num=1000, cut=True, th=1e-1, alpha=0.5)
