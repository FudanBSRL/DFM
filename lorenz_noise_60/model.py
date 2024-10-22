from torch import nn


class Net(nn.Module):
    def __init__(self, width, deep, in_num, out_num):
        super(Net, self).__init__()
        self.Pinn = nn.Sequential()
        self.Pinn.add_module('in', nn.Linear(in_num, width))
        self.Pinn.add_module('activate1', nn.Tanh())

        for i in range(deep - 2):
            self.Pinn.add_module(f'hidden{i}', nn.Linear(width, width))
            self.Pinn.add_module(f'activate{i + 2}', nn.Tanh())
            
        self.Pinn.add_module(f'out', nn.Linear(width, out_num))

    def forward(self, x):
        out = self.Pinn(x)
        return out
    

class Net2(nn.Module):
    def __init__(self, width, deep, in_num, out_num):
        super(Net2, self).__init__()
        self.Pinn = nn.Sequential()
        self.Pinn.add_module('in', nn.Linear(in_num, width))
        self.Pinn.add_module('noemalize1', nn.LayerNorm(width))
        self.Pinn.add_module('activate1', nn.Tanh())

        for i in range(deep - 2):
            self.Pinn.add_module(f'hidden{i}', nn.Linear(width, width))
            self.Pinn.add_module(f'noemalize{i+2}', nn.LayerNorm(width))
            self.Pinn.add_module(f'activate{i + 2}', nn.Tanh())
            
        self.Pinn.add_module(f'out', nn.Linear(width, out_num))

    def forward(self, x):
        out = self.Pinn(x)
        return out