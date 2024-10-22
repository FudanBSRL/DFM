import os
import torch
import math
import random
import sys
import numpy as np
from torch import nn
from torch.autograd import Variable as V
from model import Net
from lorenz_ode import lorenz_ode
from plot_pic import plot_pic
from discover import discover_gpinn, discover, num_candidates, init_ksi_data 

torch.pi = math.pi


class pinn:
    def __init__(self, device='cpu', reset=False) -> None:

        torch.set_printoptions(precision=6, sci_mode=False)

        self.gpinn = False
        self.last_epoch = 0
        self.ex = 'rb_pre_without_model_ex4'

        self.dof = 3

        self.e = 5e-3

        self.num_epochs = 50000000
        self.device = device
        

        self.net_width = 256
        self.net_deep = 4

        self.model_filename = f'./model/model_{self.ex}.pth'
        self.params_filename = f'./params/params_{self.ex}.pt'
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
            self.net = Net(width=self.net_width, deep=self.net_deep, in_num=1, out_num=3)
            self.net.double()
            for m in self.net.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)
                    
            torch.save(self.net, self.model_filename)
        
        self.net.to(device)
        self.params = self.init_params(reset)
        self.ksi, self.Lambda = self.init_ksi(reset)
        # self.Lambda = torch.ones(self.ksi.shape)
        # self.Lambda = torch.tensor([[1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 0, 0]])

        self.train_time = 3
        self.current_train = 10
        self.init_data()

    
    def update_lambda(self, th):
        self.Lambda[torch.abs(self.ksi) < th] = 0
        self.ksi.data = self.ksi * self.Lambda
        torch.save(self.ksi, self.ksi_filename)
        torch.save(self.Lambda, self.Lambda_filename)
        print(self.Lambda)
        print(self.ksi)


    def init_params(self, reset=False):
        # init m, k1, k3, c
        if not reset and os.path.exists(self.params_filename):
            params = torch.load(self.params_filename)
        else:
            params = torch.randn(3)
        params = params.detach()
        params = params.to(self.device)
        params.requires_grad = True
        return params
    

    def init_data(self):
        data = lorenz_ode(time_span=(0, 25), ic=[-8, 7, 27], t_eval=torch.linspace(0, 25, 1001), method='RK45')
        self.t = data.t
        self.acc_data = data.y
        self.x = data.y[0, :]

    
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
    

    def loss_gpinn(self, PT):
        # mse_ac_p, mse_ac_p_t, u, u_t = pde_loss_gpinn(self.net, PT, self.params)
        msed, msed_t = discover_gpinn(net=self.net, T=PT, Lambda=self.Lambda, ksi = self.ksi)

        mseu = self.u_loss()

        L1_ksi = torch.mean(torch.abs(self.ksi))

        loss = (self.mse_weight[0] * msed + self.mse_weight[1] * msed_t + self.mse_weight[2] * mseu + self.mse_weight[3] * L1_ksi)
        info = [msed, mseu, msed_t]
        return loss, info
    

    def loss(self, PT):
        # mse_ac_p, u, u_t = pde_loss(self.net, PT, self.params)
        msed = discover(net=self.net, T=PT, ksi = self.ksi)

        mseu = self.u_loss()

        L1_ksi = torch.mean(torch.abs(self.ksi))

        loss = (self.mse_weight[0] * msed + self.mse_weight[2] * mseu + self.mse_weight[3] * L1_ksi)
        info = [msed, mseu]
        return loss, info


    def init_ksi_LS(self):
        PT = V(torch.from_numpy(self.t[0:400]).double()).reshape(-1, 1).to(self.device)
        self.ksi.data = init_ksi_data(self.net, PT)
        print(self.ksi)
        torch.save(self.ksi, self.ksi_filename)


    def train_ksi(self, lr, epoch_num, cut, th):
        
        
        optimizer = torch.optim.Adam([{'params': self.ksi, 'lr': lr}])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2000, T_mult=2)

        for epoch in range(epoch_num):

            optimizer.zero_grad()
            
            # PT = np.random.uniform(low=0, high=10, size=(32, 1))
            # PT = V(torch.from_numpy(PT).double().reshape(-1, 1)).to(self.device)
            PT = V(torch.from_numpy(self.t[0:401]).double()).reshape(-1, 1).to(self.device)
            msed, _, _ = discover(net=self.net, T=PT, ksi = self.ksi)

            l1 = torch.mean(torch.abs(self.ksi))
            loss = msed + 1e-5 * l1

            loss.backward()
            optimizer.step()
            # scheduler.step()

            if (epoch) % self.plot_epoch == 0:
                print(f'epoch {epoch}, loss {loss:f}, msed {msed:f}')

            if (epoch) % (10 * self.plot_epoch) == 0:
                if cut:
                    with torch.no_grad():
                        self.ksi[torch.abs(self.ksi) < th] = 0
                print(self.ksi)
                torch.save(self.ksi, self.ksi_filename)

    

    def u_loss(self):
        T = V(torch.from_numpy(self.t[0:401]).double()).reshape(-1, 1).to(self.device)
        XYZ = V(torch.from_numpy(self.acc_data[:, 0:401]).double()).to(self.device).T
        y_hat = self.net(T)
        mseu = self.mse_cost_function(y_hat, XYZ)
        return mseu


    def train_regression(self, lr, epoch_num):
        optimizer = torch.optim.Adam([{'params': self.net.parameters(), 'lr': lr}])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epoch_num, T_mult=2)

        for epoch in range(epoch_num):

            optimizer.zero_grad()
            
            loss = self.u_loss()

            loss.backward()
            optimizer.step()
            # scheduler.step()

            if (epoch) % self.plot_epoch == 0:
                print(f'epoch {epoch}, loss {loss}')
                torch.save(self.net, self.model_filename)

            if (epoch) % (10 * self.plot_epoch) == 0:
                plot_pic(ex='pretrain', acc_data_t=self.t, acc_data=self.acc_data, net=self.net)


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

            if (epoch) % (10 * self.plot_epoch) == 0:
                plot_pic(ex='pretrain', acc_data_t=self.t, acc_data=self.acc_data, net=self.net)
                if cut:
                    with torch.no_grad():
                        self.ksi[torch.abs(self.ksi) < th] = 0
                print(self.ksi)
                torch.save(self.ksi, self.ksi_filename)
                

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print(f'run on {device}')

    if len(sys.argv) >= 2 and sys.argv[1] == 'new':
        train = pinn(device, reset=True)
    else:
        train = pinn(device)
    
    # train.train_regression(lr=1e-3, epoch_num=5000)
    # train.train_regression(lr=1e-4, epoch_num=5000)
    # train.init_ksi_LS()
    # train.update_lambda(th=1e-3)
    
    # train.train(pos=1, lr1=1e-4, lr2=1e-3, epoch_num=50000, w1=1e-4, w2=1, w3=1e-5, cut=True, th=1e-2)
    # train.update_lambda(th=1e-2)
    # train.train(pos=2, lr1=1e-4, lr2=1e-3, epoch_num=50000, w1=1e-4, w2=1, w3=1e-5, cut=True, th=1e-2)
    # train.update_lambda(th=1e-2)
    # train.train(pos=3, lr1=1e-4, lr2=1e-3, epoch_num=50000, w1=1e-4, w2=1, w3=1e-5, cut=True, th=1e-2)
    # train.update_lambda(th=1e-2)
    # train.train(pos=4, lr1=1e-4, lr2=1e-3, epoch_num=50000, w1=1e-4, w2=1, w3=1e-5, cut=True, th=1e-2)
    # train.update_lambda(th=1e-2)
    # train.train(pos=5, lr1=1e-4, lr2=1e-3, epoch_num=50000, w1=1e-4, w2=1, w3=1e-5, cut=True, th=1e-2)
    # train.update_lambda(th=1e-2)
    # train.train(pos=6, lr1=1e-5, lr2=1e-5, epoch_num=100000, w1=1e-4, w2=1, w3=0, cut=True, th=1e-2)
    # train.update_lambda(th=1e-2)
    # train.train(pos=7, lr1=1e-6, lr2=1e-5, epoch_num=100000, w1=1, w2=1, w3=0, cut=True, th=1e-2)
    # train.update_lambda(th=1e-2)
    # train.train(pos=8, lr1=1e-6, lr2=1e-5, epoch_num=1000000, w1=1, w2=1, w3=0, cut=True, th=1e-2)
    # train.update_lambda(th=1e-2)
    train.train(pos=9, lr1=1e-7, lr2=1e-5, epoch_num=1000000, w1=1e-5, w2=1, w3=0, cut=True, th=1e-2)
    train.update_lambda(th=1e-2)
    
    
    