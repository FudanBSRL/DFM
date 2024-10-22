import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
from data_util import read_data

ex = 'rb_pre_without_model_ex6'
Ksi = None
y_0 = None


def lorenz_id(t, y):
  dy0 = Ksi[0, 0] + Ksi[1, 0] * y[0] + Ksi[2, 0] * y[1] + Ksi[3, 0] * y[2] + Ksi[4, 0] * y[0] * y[0] + Ksi[5, 0] * y[0] * y[1] + Ksi[6, 0] * y[0] * y[2] + Ksi[7, 0] * y[1] * y[1] + Ksi[8, 0] * y[1] * y[2] + Ksi[9, 0] * y[2] * y[2]
  dy1 = Ksi[0, 1] + Ksi[1, 1] * y[0] + Ksi[2, 1] * y[1] + Ksi[3, 1] * y[2] + Ksi[4, 1] * y[0] * y[0] + Ksi[5, 1] * y[0] * y[1] + Ksi[6, 1] * y[0] * y[2] + Ksi[7, 1] * y[1] * y[1] + Ksi[8, 1] * y[1] * y[2] + Ksi[9, 1] * y[2] * y[2]
  dy2 = Ksi[0, 2] + Ksi[1, 2] * y[0] + Ksi[2, 2] * y[1] + Ksi[3, 2] * y[2] + Ksi[4, 2] * y[0] * y[0] + Ksi[5, 2] * y[0] * y[1] + Ksi[6, 2] * y[0] * y[2] + Ksi[7, 2] * y[1] * y[1] + Ksi[8, 2] * y[1] * y[2] + Ksi[9, 2] * y[2] * y[2]
  
  # since lorenz is 3-dimensional, dy/dt should be an array of 3 values
  return [dy0, dy1, dy2]


def lorenz_ode_id(time_span, ic, t_eval, method):
  lorenz_soln = solve_ivp(lorenz_id, time_span, ic, t_eval=t_eval, method=method, rtol=1e-10, atol=1e-10)
  return lorenz_soln


if __name__ == '__main__':
  Ksi = torch.load(f'./ksis/ksi_{ex}.pt').detach().numpy()
  print(Ksi)
  
  current_train = 25
  # target_data = lorenz_ode(time_span=(0, current_train), ic=[-8., 7., 27.], t_eval=np.linspace(0, current_train, int(40 * current_train + 1)), method='RK45')
  # ic = target_data.y[:, 400]
  acc_t, acc_data = read_data(f'./datas/lorenz_sparse.csv')
  ic = acc_data[:, 200]
  # print(target_data.t[400])

  id_data = lorenz_ode_id(time_span=(0, 15), ic=ic, t_eval=np.linspace(0, 15, round(20 * 15 + 1)), method='RK45')
  id_data.t = id_data.t + 10

  color = ['purple', 'red']
  fig1, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=False)
  ax1.plot(acc_t, acc_data[0, :], c=color[0], linestyle='-')  # 25s
  ax1.plot(id_data.t, id_data.y[0, :], c=color[1], linestyle='--') # trian data

  ax1.set_xlabel('t')
  ax1.set_ylabel('x')
  ax1.set_xlim((10, 25))
  # ax1.legend(['true', 'regression', 'predicion', 'adjust'], loc='upper right')

  ax2.plot(acc_t, acc_data[1, :], c=color[0], linestyle='-')  # 25s
  ax2.plot(id_data.t, id_data.y[1, :], c=color[1], linestyle='--') # trian data

  ax2.set_xlabel('t')
  ax2.set_ylabel('y')
  ax2.set_xlim((10, 25))

  ax3.plot(acc_t, acc_data[2, :], c=color[0], linestyle='-')  # 25s
  ax3.plot(id_data.t, id_data.y[2, :], c=color[1], linestyle='--') # trian data

  ax3.set_xlabel('t')
  ax3.set_ylabel('z')
  ax3.set_xlim((10, 25))
  fig1.legend(['true', 'ODE45'], loc='upper center', ncol=2)
  fig1.savefig('cmp.png', dpi=500)

  td_T = torch.from_numpy(acc_t).double()
  td_Y = torch.from_numpy(acc_data).double()
  
  # id_T = torch.from_numpy(id_data.t).double()
  id_T = td_T
  id_Y = torch.from_numpy(id_data.y).double()

  id_Y = torch.cat((td_Y[:, 0:201], id_Y[:, 1:]), axis=1)
  
  print(id_Y.shape)
  torch.save(id_T, "./datas/id_T.pt")
  torch.save(id_Y, "./datas/id_Y.pt")