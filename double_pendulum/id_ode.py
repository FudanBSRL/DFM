import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
from data_util import read_data

ex = 'rb_pre_without_model_ex1'
Ksi = None
y_0 = None


def lorenz_id(t, y):
  feature_list = [1.0]
  
  for i in range(4):
    feature_list.append(y[i])

  for i in range(4):
    feature_list.append(np.sin(y[i]))
    feature_list.append(np.cos(y[i]))

  
  for i in range(4):
    for j in range(i, 4):
      feature_list.append(y[i] * y[j])
      feature_list.append(np.sin(y[i] + y[j]))
      feature_list.append(np.cos(y[i] + y[j]))
      if i != j :
        feature_list.append(np.sin(y[i] - y[j]))
        feature_list.append(np.cos(y[i] - y[j]))

  for i in range(4):
    for j in range(i, 4):
      for k in range(j, 4):
        feature_list.append(y[i] * y[j] * y[k])

  th1 = 0
  th2 = 0
  w1 = 0
  w2 = 0

  for i in range(len(feature_list)):
    th1 += Ksi[i, 0] * feature_list[i]
    th2 += Ksi[i, 1] * feature_list[i]
    w1 += Ksi[i, 2] * feature_list[i]
    w2 += Ksi[i, 3] * feature_list[i]

  # since lorenz is 3-dimensional, dy/dt should be an array of 3 values
  return [th1, th2, w1, w2]


def lorenz_ode_id(time_span, ic, t_eval, method):
  lorenz_soln = solve_ivp(lorenz_id, time_span, ic, t_eval=t_eval, method=method, rtol=1e-10, atol=1e-10)
  return lorenz_soln


if __name__ == '__main__':
  Ksi = torch.load(f'./ksis/ksi_{ex}.pt').detach().numpy()
  print(Ksi)
  
  acc_t, acc_data = read_data(f'./csvs/d_pendulum.csv')
  ic = acc_data[:, 800]

  train_time = 20
  test_time = 20

  id_data = lorenz_ode_id(time_span=(0, test_time), ic=ic, t_eval=np.linspace(0, test_time, round(test_time * 40 + 1)), method='RK45')
  id_data.t = id_data.t + train_time

  xlim = (train_time, train_time + test_time)
  # 结果对比
  color = ['blue', 'red']
  fig1, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=False)
  ax1.plot(acc_t, acc_data[0, :], c=color[0], linestyle='-')  # 25s
  ax1.plot(id_data.t, id_data.y[0, :], c=color[1], linestyle='--') # trian data

  ax1.set_xlim(xlim)

  ax2.plot(acc_t, acc_data[1, :], c=color[0], linestyle='-')  # 25s
  ax2.plot(id_data.t, id_data.y[1, :], c=color[1], linestyle='--') # trian data

  ax2.set_xlim(xlim)

  ax3.plot(acc_t, acc_data[2, :], c=color[0], linestyle='-')  # 25s
  ax3.plot(id_data.t, id_data.y[2, :], c=color[1], linestyle='--') # trian data

  ax3.set_xlim(xlim)

  ax4.plot(acc_t, acc_data[3, :], c=color[0], linestyle='-')  # 25s
  ax4.plot(id_data.t, id_data.y[3, :], c=color[1], linestyle='--') # trian data

  ax4.set_xlim(xlim)


  fig1.legend(['true', 'ODE45'], loc='upper center', ncol=2)
  fig1.savefig('res.png', dpi=500)

  td_T = torch.from_numpy(acc_t).double()
  td_Y = torch.from_numpy(acc_data).double()
  
  # id_T = torch.from_numpy(id_data.t).double()
  id_T = td_T
  id_Y = torch.from_numpy(id_data.y).double()

  id_Y = torch.cat((td_Y[:, 0:801], id_Y[:, 1:]), axis=1)
  
  print(id_Y.shape)
  torch.save(id_T, "./datas/id_T.pt")
  torch.save(id_Y, "./datas/id_Y.pt")


  err_Y = torch.sum((id_Y - acc_data) ** 2, axis=0)
  
  # 误差
  fig1, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=False)
  ax1.plot(id_T, err_Y, c=color[1], linestyle='--') # trian data

  ax1.set_xlim(xlim)

  # ax2.plot(id_T, id_Y[1, :] - acc_data[1, :], c=color[1], linestyle='--') # trian data

  # ax2.set_xlim(xlim)

  # ax3.plot(id_T, id_Y[2, :] - acc_data[2, :], c=color[1], linestyle='--') # trian data

  # ax3.set_xlim(xlim)

  # ax4.plot(id_T, id_Y[3, :] - acc_data[3, :], c=color[1], linestyle='--') # trian data

  # ax4.set_xlim(xlim)

  fig1.savefig('cmp.png', dpi=500)