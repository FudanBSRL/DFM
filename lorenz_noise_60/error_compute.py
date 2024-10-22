import torch
import numpy as np
from lorenz_ode import lorenz_ode
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from data_util import read_noise_data
from torch.autograd import Variable as V


def lorenz_ode_id(ksi_path,time_span, ic, t_eval, method):
  global Ksi 
  Ksi = torch.load(ksi_path).detach().numpy()
  lorenz_soln = solve_ivp(lorenz_id, time_span, ic, t_eval=t_eval, method=method, rtol=1e-10, atol=1e-10)
  return lorenz_soln


def lorenz_id(t, y):
  dy0 = Ksi[0, 0] + Ksi[1, 0] * y[0] + Ksi[2, 0] * y[1] + Ksi[3, 0] * y[2] + Ksi[4, 0] * y[0] * y[0] + Ksi[5, 0] * y[0] * y[1] + Ksi[6, 0] * y[0] * y[2] + Ksi[7, 0] * y[1] * y[1] + Ksi[8, 0] * y[1] * y[2] + Ksi[9, 0] * y[2] * y[2]
  dy1 = Ksi[0, 1] + Ksi[1, 1] * y[0] + Ksi[2, 1] * y[1] + Ksi[3, 1] * y[2] + Ksi[4, 1] * y[0] * y[0] + Ksi[5, 1] * y[0] * y[1] + Ksi[6, 1] * y[0] * y[2] + Ksi[7, 1] * y[1] * y[1] + Ksi[8, 1] * y[1] * y[2] + Ksi[9, 1] * y[2] * y[2]
  dy2 = Ksi[0, 2] + Ksi[1, 2] * y[0] + Ksi[2, 2] * y[1] + Ksi[3, 2] * y[2] + Ksi[4, 2] * y[0] * y[0] + Ksi[5, 2] * y[0] * y[1] + Ksi[6, 2] * y[0] * y[2] + Ksi[7, 2] * y[1] * y[1] + Ksi[8, 2] * y[1] * y[2] + Ksi[9, 2] * y[2] * y[2]
  
  return [dy0, dy1, dy2]


def plot_oriRC_ad_cmp():
  current_train = 25
  target_data = lorenz_ode(time_span=(0, current_train), ic=[-8., 7., 27.], t_eval=np.linspace(0, current_train, int(40 * current_train + 1)), method='RK45')
  td_T = target_data.t
  td_Y = target_data.y

  ad_T = torch.load('./datas/T_35.pt').detach().numpy()
  ad_Y = torch.load('./datas/ad_Y_35.pt').detach().numpy().T

  oriRC_T = torch.load('./datas/oriRC_predict_T').detach().numpy()
  oriRC_Y = torch.load('./datas/oriRC_predict_Y').detach().numpy().T

  start = 400
  xticks = (10, 12, 14, 16, 18, 20, 22, 24, 25)
  color = ['#0000cc', '#008800', '#ff0000']
  fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
  ax1.plot(td_T[start:], td_Y[0, start:], c=color[0], linestyle='-')
  ax1.plot(oriRC_T[start:], oriRC_Y[0, start:], c=color[1], linestyle='--')
  ax1.plot(ad_T[start:], ad_Y[0, start:], c=color[2], linestyle='-.')
  ax1.set_xlim((10,25))
  ax1.set_xticks(xticks)

  ax2.plot(td_T[start:], td_Y[1, start:], c=color[0], linestyle='-')
  ax2.plot(oriRC_T[start:], oriRC_Y[1, start:], c=color[1], linestyle='--')
  ax2.plot(ad_T[start:], ad_Y[1, start:], c=color[2], linestyle='-.')
  ax2.set_xlim((10,25))
  ax2.set_xticks(xticks)

  ax3.plot(td_T[start:], td_Y[2, start:], c=color[0], linestyle='-')
  ax3.plot(oriRC_T[start:], oriRC_Y[2, start:], c=color[1], linestyle='--')
  ax3.plot(ad_T[start:], ad_Y[2, start:], c=color[2], linestyle='-.')
  ax3.set_xlim((10,25))
  ax3.set_xticks(xticks)

  fig2.savefig(f'./error_pic/oriRC_ad_cmp.svg', dpi=500)
  fig2.savefig(f'./error_pic/oriRC_ad_cmp.png', dpi=500)


def plot_fltRC_ad_cmp():
  current_train = 25
  target_data = lorenz_ode(time_span=(0, current_train), ic=[-8., 7., 27.], t_eval=torch.linspace(0, current_train, int(40 * current_train + 1)), method='RK45')
  td_T = target_data.t
  td_Y = target_data.y

  ad_T = torch.load('./datas/T_25.pt').detach().numpy()
  ad_Y = torch.load('./datas/ad_Y_25.pt').detach().numpy().T

  fltRC_T = torch.load('./datas/fltRC_predict_T').detach().numpy()
  fltRC_Y = torch.load('./datas/fltRC_predict_Y').detach().numpy().T

  start = 400
  xticks = (10, 12, 14, 16, 18, 20, 22, 24, 25)
  color = ['#0000cc', '#008800', '#ff0000']
  fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
  ax1.plot(td_T[start:], td_Y[0, start:], c=color[0], linestyle='-')
  ax1.plot(fltRC_T[start:], fltRC_Y[0, start:], c=color[1], linestyle='--')
  ax1.plot(ad_T[start:], ad_Y[0, start:], c=color[2], linestyle='-.')
  ax1.set_xlim((10,25))
  ax1.set_xticks(xticks)

  ax2.plot(td_T[start:], td_Y[1, start:], c=color[0], linestyle='-')
  ax2.plot(fltRC_T[start:], fltRC_Y[1, start:], c=color[1], linestyle='--')
  ax2.plot(ad_T[start:], ad_Y[1, start:], c=color[2], linestyle='-.')
  ax2.set_xlim((10,25))
  ax2.set_xticks(xticks)

  ax3.plot(td_T[start:], td_Y[2, start:], c=color[0], linestyle='-')
  ax3.plot(fltRC_T[start:], fltRC_Y[2, start:], c=color[1], linestyle='--')
  ax3.plot(ad_T[start:], ad_Y[2, start:], c=color[2], linestyle='-.')
  ax3.set_xlim((10,25))
  ax3.set_xticks(xticks)

  fig2.savefig(f'./error_pic/fltRC_ad_cmp.svg', dpi=500)
  fig2.savefig(f'./error_pic/fltRC_ad_cmp.png', dpi=500)


def plot_oriRC_ad_abs_cmp():
  current_train = 25
  target_data = lorenz_ode(time_span=(0, current_train), ic=[-8., 7., 27.], t_eval=torch.linspace(0, current_train, int(40 * current_train + 1)), method='RK45')
  td_T = target_data.t
  td_Y = target_data.y

  ad_T = torch.load('./datas/T_25.pt').detach().numpy()
  ad_Y = torch.load('./datas/ad_Y_25.pt').detach().numpy().T
  err_ad = np.abs(td_Y - ad_Y)

  oriRC_T = torch.load('./datas/oriRC_predict_T').detach().numpy()
  oriRC_Y = torch.load('./datas/oriRC_predict_Y').detach().numpy().T
  err_oriRC = np.abs(td_Y - oriRC_Y)

  start = 400
  xticks = (10, 12, 14, 16, 18, 20, 22, 24, 25)
  color = ['#0000cc', '#008800', '#ff0000']
  fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
  ax1.plot(oriRC_T[start:], err_oriRC[0, start:], c=color[0], linestyle='-')
  ax1.plot(ad_T[start:], err_ad[0, start:], c=color[2], linestyle='-.')
  ax1.set_xlim((10,25))
  ax1.set_xticks(xticks)

  ax2.plot(oriRC_T[start:], err_oriRC[1, start:], c=color[0], linestyle='-')
  ax2.plot(ad_T[start:], err_ad[1, start:], c=color[2], linestyle='-.')
  ax2.set_xlim((10,25))
  ax2.set_xticks(xticks)

  ax3.plot(oriRC_T[start:], err_oriRC[2, start:], c=color[0], linestyle='-')
  ax3.plot(ad_T[start:], err_ad[2, start:], c=color[2], linestyle='-.')
  ax3.set_xlim((10,25))
  ax3.set_xticks(xticks)

  fig2.savefig(f'./error_pic/oriRC_ad_abs_cmp.svg')
  fig2.savefig(f'./error_pic/oriRC_ad_abs_cmp.png')

  keep = 160
  xticks = (10, 12, 14)
  color = ['#0000cc', '#008800', '#ff0000']
  fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
  ax1.plot(oriRC_T[start:start+keep], err_oriRC[0, start:start+keep], c=color[0], linestyle='-')
  ax1.plot(ad_T[start:start+keep], err_ad[0, start:start+keep], c=color[2], linestyle='-.')
  ax1.set_xlim((10,14))
  ax1.set_xticks(xticks)

  ax2.plot(oriRC_T[start:start+keep], err_oriRC[1, start:start+keep], c=color[0], linestyle='-')
  ax2.plot(ad_T[start:start+keep], err_ad[1, start:start+keep], c=color[2], linestyle='-.')
  ax2.set_xlim((10,14))
  ax2.set_xticks(xticks)

  ax3.plot(oriRC_T[start:start+keep], err_oriRC[2, start:start+keep], c=color[0], linestyle='-')
  ax3.plot(ad_T[start:start+keep], err_ad[2, start:start+keep], c=color[2], linestyle='-.')
  ax3.set_xlim((10,14))
  ax3.set_xticks(xticks)

  fig2.savefig(f'./error_pic/oriRC_ad_abs_cmp_4.svg')
  fig2.savefig(f'./error_pic/oriRC_ad_abs_cmp_4.png')


def plot_oriRC_ad_de_cmp():
  target_data = lorenz_ode(time_span=(0, 25), ic=[-8., 7., 27.], t_eval=torch.linspace(0, 25, 1001), method='RK45')
  td_T = target_data.t
  td_Y = target_data.y
  td_dY = (td_Y[:, 2:] - td_Y[:, :-2]) / (td_T[2:] - td_T[:-2]).reshape(1, -1)
  err_td_T = td_T[1:-1]
  err_td = np.zeros(shape=td_dY.shape)
  err_td[0, :] = td_dY[0, :] - 10 * (td_Y[1, 1:-1] - td_Y[0, 1:-1])
  err_td[1, :] = td_dY[1, :] - td_Y[0, 1:-1] * (28 - td_Y[2, 1:-1]) + td_Y[1, 1:-1]
  err_td[2, :] = td_dY[2, :] - td_Y[0, 1:-1] * td_Y[1, 1:-1] + 8 / 3 * td_Y[2, 1:-1]

  ad_T = torch.load('./datas/T_25.pt').detach().numpy()
  ad_Y = torch.load('./datas/ad_Y_25.pt').detach().numpy().T
  ad_dY = (ad_Y[:, 2:] - ad_Y[:, :-2]) / (ad_T[2:] - ad_T[:-2]).reshape(1, -1)
  err_ad_T = ad_T[1:-1]
  err_ad = np.zeros(shape=ad_dY.shape)
  err_ad[0, :] = ad_dY[0, :] - 10 * (ad_Y[1, 1:-1] - ad_Y[0, 1:-1])
  err_ad[1, :] = ad_dY[1, :] - ad_Y[0, 1:-1] * (28 - ad_Y[2, 1:-1]) + ad_Y[1, 1:-1]
  err_ad[2, :] = ad_dY[2, :] - ad_Y[0, 1:-1] * ad_Y[1, 1:-1] + 8 / 3 * ad_Y[2, 1:-1]

  rc_T = torch.load('./datas/oriRC_predict_T').detach().numpy()
  rc_Y = torch.load('./datas/oriRC_predict_Y').detach().numpy().T
  rc_dY = (rc_Y[:, 2:] - rc_Y[:, :-2]) / (rc_T[2:] - rc_T[:-2]).reshape(1, -1)
  err_rc_T = rc_T[1:-1]
  err_rc = np.zeros(shape=rc_dY.shape)
  err_rc[0, :] = rc_dY[0, :] - 10 * (rc_Y[1, 1:-1] - rc_Y[0, 1:-1])
  err_rc[1, :] = rc_dY[1, :] - rc_Y[0, 1:-1] * (28 - rc_Y[2, 1:-1]) + rc_Y[1, 1:-1]
  err_rc[2, :] = rc_dY[2, :] - rc_Y[0, 1:-1] * rc_Y[1, 1:-1] + 8 / 3 * rc_Y[2, 1:-1]


  err_td_x = err_td[0, :]
  err_td_y = err_td[1, :]
  err_td_z = err_td[2, :]
  # err_ad_x = np.abs(err_ad[0, :])
  # err_ad_y = np.abs(err_ad[1, :])
  # err_ad_z = np.abs(err_ad[2, :])
  # err_rc_x = np.abs(err_rc[0, :])
  # err_rc_y = np.abs(err_rc[1, :])
  # err_rc_z = np.abs(err_rc[2, :])
  err_ad_x = np.abs(err_ad[0, :] - err_td_x)
  err_ad_y = np.abs(err_ad[1, :] - err_td_y)
  err_ad_z = np.abs(err_ad[2, :] - err_td_z)
  err_rc_x = np.abs(err_rc[0, :] - err_td_x)
  err_rc_y = np.abs(err_rc[1, :] - err_td_y)
  err_rc_z = np.abs(err_rc[2, :] - err_td_z)

  
  xticks = (10, 12, 14, 16, 18, 20, 22, 24, 25)
  # xticks = (10, 12, 14, 16)

  start = 400
  keep = 600
  color = ['#385989', '#d22027']
  fig1, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=False)
  ax1.plot(err_rc_T[start:start+keep], err_rc_x[start:start+keep], c=color[0], linestyle='-')
  ax1.plot(err_ad_T[start:start+keep], err_ad_x[start:start+keep], c=color[1], linestyle='-')
  ax1.set_xticks(xticks)

  # ax1.set_xlabel('t')
  # ax1.set_ylabel('x')
  # ax1.legend(['true', 'regression', 'predicion', 'adjust'], loc='upper right')

  ax2.plot(err_rc_T[start:start+keep], err_rc_y[start:start+keep], c=color[0], linestyle='-')
  ax2.plot(err_ad_T[start:start+keep], err_ad_y[start:start+keep], c=color[1], linestyle='-')
  ax2.set_xticks(xticks)

  # ax2.set_xlabel('t')
  # ax2.set_ylabel('y')

  ax3.plot(err_rc_T[start:start+keep], err_rc_z[start:start+keep], c=color[0], linestyle='-')
  ax3.plot(err_ad_T[start:start+keep], err_ad_z[start:start+keep], c=color[1], linestyle='-')
  ax3.set_xticks(xticks)

  # ax3.set_xlabel('t')
  # ax3.set_ylabel('z')
  fig1.savefig('./error_pic/oriRC_ad_de_cmp.png')
  fig1.savefig('./error_pic/oriRC_ad_de_cmp.svg')


  xticks = (10, 12, 14)
  start = 400
  keep = 160
  color = ['#385989', '#d22027']
  fig1, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=False)
  ax1.plot(err_rc_T[start:start+keep], err_rc_x[start:start+keep], c=color[0], linestyle='-')
  ax1.plot(err_ad_T[start:start+keep], err_ad_x[start:start+keep], c=color[1], linestyle='-')
  ax1.set_xticks(xticks)

  # ax1.set_xlabel('t')
  # ax1.set_ylabel('x')
  # ax1.legend(['true', 'regression', 'predicion', 'adjust'], loc='upper right')

  ax2.plot(err_rc_T[start:start+keep], err_rc_y[start:start+keep], c=color[0], linestyle='-')
  ax2.plot(err_ad_T[start:start+keep], err_ad_y[start:start+keep], c=color[1], linestyle='-')
  ax2.set_xticks(xticks)

  # ax2.set_xlabel('t')
  # ax2.set_ylabel('y')

  ax3.plot(err_rc_T[start:start+keep], err_rc_z[start:start+keep], c=color[0], linestyle='-')
  ax3.plot(err_ad_T[start:start+keep], err_ad_z[start:start+keep], c=color[1], linestyle='-')
  ax3.set_xticks(xticks)

  # ax3.set_xlabel('t')
  # ax3.set_ylabel('z')
  fig1.savefig('./error_pic/oriRC_ad_de_cmp_4.png')
  fig1.savefig('./error_pic/oriRC_ad_de_cmp_4.svg')

  
def plot_fltRC_ad_abs_cmp():
  current_train = 25
  target_data = lorenz_ode(time_span=(0, current_train), ic=[-8., 7., 27.], t_eval=torch.linspace(0, current_train, int(40 * current_train + 1)), method='RK45')
  td_T = target_data.t
  td_Y = target_data.y

  ad_T = torch.load('./datas/T_25.pt').detach().numpy()
  ad_Y = torch.load('./datas/ad_Y_25.pt').detach().numpy().T
  err_ad = np.abs(td_Y - ad_Y)

  fltRC_T = torch.load('./datas/fltRC_predict_T').detach().numpy()
  fltRC_Y = torch.load('./datas/fltRC_predict_Y').detach().numpy().T
  err_fltRC = np.abs(td_Y - fltRC_Y)

  start = 400
  xticks = (10, 12, 14, 16, 18, 20, 22, 24, 25)
  color = ['#0000cc', '#008800', '#ff0000']
  fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
  ax1.plot(fltRC_T[start:], err_fltRC[0, start:], c=color[0], linestyle='-')
  ax1.plot(ad_T[start:], err_ad[0, start:], c=color[2], linestyle='-.')
  ax1.set_xlim((10,25))
  ax1.set_xticks(xticks)

  ax2.plot(fltRC_T[start:], err_fltRC[1, start:], c=color[0], linestyle='-')
  ax2.plot(ad_T[start:], err_ad[1, start:], c=color[2], linestyle='-.')
  ax2.set_xlim((10,25))
  ax2.set_xticks(xticks)

  ax3.plot(fltRC_T[start:], err_fltRC[2, start:], c=color[0], linestyle='-')
  ax3.plot(ad_T[start:], err_ad[2, start:], c=color[2], linestyle='-.')
  ax3.set_xlim((10,25))
  ax3.set_xticks(xticks)

  fig2.savefig(f'./error_pic/fltRC_ad_abs_cmp.svg')
  fig2.savefig(f'./error_pic/fltRC_ad_abs_cmp.png')

  keep = 240
  xticks = (10, 12, 14)
  color = ['#0000cc', '#008800', '#ff0000']
  fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
  ax1.plot(fltRC_T[start:start+keep], err_fltRC[0, start:start+keep], c=color[0], linestyle='-')
  ax1.plot(ad_T[start:start+keep], err_ad[0, start:start+keep], c=color[2], linestyle='-.')
  ax1.set_xlim((10,14))
  ax1.set_xticks(xticks)

  ax2.plot(fltRC_T[start:start+keep], err_fltRC[1, start:start+keep], c=color[0], linestyle='-')
  ax2.plot(ad_T[start:start+keep], err_ad[1, start:start+keep], c=color[2], linestyle='-.')
  ax2.set_xlim((10,14))
  ax2.set_xticks(xticks)

  ax3.plot(fltRC_T[start:start+keep], err_fltRC[2, start:start+keep], c=color[0], linestyle='-')
  ax3.plot(ad_T[start:start+keep], err_ad[2, start:start+keep], c=color[2], linestyle='-.')
  ax3.set_xlim((10,14))
  ax3.set_xticks(xticks)

  fig2.savefig(f'./error_pic/fltRC_ad_abs_cmp_6.svg')
  fig2.savefig(f'./error_pic/fltRC_ad_abs_cmp_6.png')


def plot_fltRC_ad_de_cmp():
  target_data = lorenz_ode(time_span=(0, 25), ic=[-8., 7., 27.], t_eval=torch.linspace(0, 25, 1001), method='RK45')
  td_T = target_data.t
  td_Y = target_data.y
  td_dY = (td_Y[:, 2:] - td_Y[:, :-2]) / (td_T[2:] - td_T[:-2]).reshape(1, -1)
  err_td_T = td_T[1:-1]
  err_td = np.zeros(shape=td_dY.shape)
  err_td[0, :] = td_dY[0, :] - 10 * (td_Y[1, 1:-1] - td_Y[0, 1:-1])
  err_td[1, :] = td_dY[1, :] - td_Y[0, 1:-1] * (28 - td_Y[2, 1:-1]) + td_Y[1, 1:-1]
  err_td[2, :] = td_dY[2, :] - td_Y[0, 1:-1] * td_Y[1, 1:-1] + 8 / 3 * td_Y[2, 1:-1]

  ad_T = torch.load('./datas/T_25.pt').detach().numpy()
  ad_Y = torch.load('./datas/ad_Y_25.pt').detach().numpy().T
  ad_dY = (ad_Y[:, 2:] - ad_Y[:, :-2]) / (ad_T[2:] - ad_T[:-2]).reshape(1, -1)
  err_ad_T = ad_T[1:-1]
  err_ad = np.zeros(shape=ad_dY.shape)
  err_ad[0, :] = ad_dY[0, :] - 10 * (ad_Y[1, 1:-1] - ad_Y[0, 1:-1])
  err_ad[1, :] = ad_dY[1, :] - ad_Y[0, 1:-1] * (28 - ad_Y[2, 1:-1]) + ad_Y[1, 1:-1]
  err_ad[2, :] = ad_dY[2, :] - ad_Y[0, 1:-1] * ad_Y[1, 1:-1] + 8 / 3 * ad_Y[2, 1:-1]

  rc_T = torch.load('./datas/fltRC_predict_T').detach().numpy()
  rc_Y = torch.load('./datas/fltRC_predict_Y').detach().numpy().T
  rc_dY = (rc_Y[:, 2:] - rc_Y[:, :-2]) / (rc_T[2:] - rc_T[:-2]).reshape(1, -1)
  err_rc_T = rc_T[1:-1]
  err_rc = np.zeros(shape=rc_dY.shape)
  err_rc[0, :] = rc_dY[0, :] - 10 * (rc_Y[1, 1:-1] - rc_Y[0, 1:-1])
  err_rc[1, :] = rc_dY[1, :] - rc_Y[0, 1:-1] * (28 - rc_Y[2, 1:-1]) + rc_Y[1, 1:-1]
  err_rc[2, :] = rc_dY[2, :] - rc_Y[0, 1:-1] * rc_Y[1, 1:-1] + 8 / 3 * rc_Y[2, 1:-1]


  err_td_x = err_td[0, :]
  err_td_y = err_td[1, :]
  err_td_z = err_td[2, :]
  # err_ad_x = np.abs(err_ad[0, :])
  # err_ad_y = np.abs(err_ad[1, :])
  # err_ad_z = np.abs(err_ad[2, :])
  # err_rc_x = np.abs(err_rc[0, :])
  # err_rc_y = np.abs(err_rc[1, :])
  # err_rc_z = np.abs(err_rc[2, :])
  err_ad_x = np.abs(err_ad[0, :] - err_td_x)
  err_ad_y = np.abs(err_ad[1, :] - err_td_y)
  err_ad_z = np.abs(err_ad[2, :] - err_td_z)
  err_rc_x = np.abs(err_rc[0, :] - err_td_x)
  err_rc_y = np.abs(err_rc[1, :] - err_td_y)
  err_rc_z = np.abs(err_rc[2, :] - err_td_z)

  
  xticks = (10, 12, 14, 16, 18, 20, 22, 24, 25)
  # xticks = (10, 12, 14, 16)

  start = 400
  keep = 600
  color = ['#385989', '#d22027']
  fig1, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=False)
  ax1.plot(err_rc_T[start:start+keep], err_rc_x[start:start+keep], c=color[0], linestyle='-')
  ax1.plot(err_ad_T[start:start+keep], err_ad_x[start:start+keep], c=color[1], linestyle='-')
  ax1.set_xticks(xticks)

  # ax1.set_xlabel('t')
  # ax1.set_ylabel('x')
  # ax1.legend(['true', 'regression', 'predicion', 'adjust'], loc='upper right')

  ax2.plot(err_rc_T[start:start+keep], err_rc_y[start:start+keep], c=color[0], linestyle='-')
  ax2.plot(err_ad_T[start:start+keep], err_ad_y[start:start+keep], c=color[1], linestyle='-')
  ax2.set_xticks(xticks)

  # ax2.set_xlabel('t')
  # ax2.set_ylabel('y')

  ax3.plot(err_rc_T[start:start+keep], err_rc_z[start:start+keep], c=color[0], linestyle='-')
  ax3.plot(err_ad_T[start:start+keep], err_ad_z[start:start+keep], c=color[1], linestyle='-')
  ax3.set_xticks(xticks)

  # ax3.set_xlabel('t')
  # ax3.set_ylabel('z')
  fig1.savefig('./error_pic/fltRC_ad_de_cmp.png')
  fig1.savefig('./error_pic/fltRC_ad_de_cmp.svg')


  xticks = (10, 12, 14, 16)
  start = 400
  keep = 240
  color = ['#385989', '#d22027']
  fig1, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=False)
  ax1.plot(err_rc_T[start:start+keep], err_rc_x[start:start+keep], c=color[0], linestyle='-')
  ax1.plot(err_ad_T[start:start+keep], err_ad_x[start:start+keep], c=color[1], linestyle='-')
  ax1.set_xticks(xticks)

  # ax1.set_xlabel('t')
  # ax1.set_ylabel('x')
  # ax1.legend(['true', 'regression', 'predicion', 'adjust'], loc='upper right')

  ax2.plot(err_rc_T[start:start+keep], err_rc_y[start:start+keep], c=color[0], linestyle='-')
  ax2.plot(err_ad_T[start:start+keep], err_ad_y[start:start+keep], c=color[1], linestyle='-')
  ax2.set_xticks(xticks)

  # ax2.set_xlabel('t')
  # ax2.set_ylabel('y')

  ax3.plot(err_rc_T[start:start+keep], err_rc_z[start:start+keep], c=color[0], linestyle='-')
  ax3.plot(err_ad_T[start:start+keep], err_ad_z[start:start+keep], c=color[1], linestyle='-')
  ax3.set_xticks(xticks)

  # ax3.set_xlabel('t')
  # ax3.set_ylabel('z')
  fig1.savefig('./error_pic/fltRC_ad_de_cmp_6.png')
  fig1.savefig('./error_pic/fltRC_ad_de_cmp_6.svg')


def nrmse_std(predictions, targets):
    # 计算RMSE
    rmse = np.sqrt(((predictions - targets) ** 2).mean())
    # return rmse
    # 计算每个维度的标准差
    std_of_values = np.std(targets, axis=1)
    # 由于标准差是针对每个维度的，我们可以取平均值来代表所有维度的平均标准差
    average_std = np.mean(std_of_values)
    
    # 计算NRMSE
    nrmse = rmse / average_std
    return nrmse


def compute_filter():
  pre_train_ex = f'rb_pre_without_model_ex7'
  trained_net = torch.load(f'./model/model_{pre_train_ex}.pth')

  target_data = lorenz_ode(time_span=(0, 25), ic=[-8., 7., 27.], t_eval=np.linspace(0, 25, 1001), method='RK45')
  acc_data = target_data.y

  noise_t, noise_data = read_noise_data('./datas/lorenz_noise_snr60.csv')
  y_trained_reg = trained_net(V(torch.from_numpy(noise_t[0: 400 + 1])).double().reshape(-1, 1)).detach().numpy().T

  nrmse_ori = nrmse_std(acc_data[:, 0:401], noise_data[:, 0:401])
  nrmse_flt = nrmse_std(acc_data[:, 0:401], y_trained_reg[:, 0:401])

  E_signal = np.mean(np.square(acc_data[:, 0:401]))
  E_noise = np.mean(np.square(acc_data[:, 0:401] - noise_data[:, 0:401]))
  E_reg = np.mean(np.square(acc_data[:, 0:401] - y_trained_reg[:, 0:401]))

  snr_noise = 10 * np.log10(E_signal / E_noise)
  snr_reg = 10 * np.log10(E_signal / E_reg)

  print(f'ori_nemse {nrmse_ori}, flt_nrmse {nrmse_flt}')
  print(f'ori_snr {snr_noise}, flt_snr {snr_reg}')


def compute_nrmse():
  target_data = lorenz_ode(time_span=(0, 25), ic=[-8., 7., 27.], t_eval=np.linspace(0, 25, 1001), method='RK45')
  acc_data = target_data.y

  exs = [29, 30, 31, 32, 33, 34, 35, 36]
  alphas = [0.9, 0.8, 0.5, 0.4, 0.2, 0.1, 0.08, 0.05]
  start = 40 * 10
  end = round(40 * 13) + 1
  for ex in exs:
    predict_Y = torch.load(f'./datas/predict_Y_{ex}.pt').T.detach().numpy()
    # print(predict_Y.shape)
    ad_Y = torch.load(f'./datas/ad_Y_{ex}.pt').T.detach().numpy()
    # print(ad_Y.shape)
    # end = min(ad_Y.shape[1], 721)
    nrmse_rc = nrmse_std(predict_Y[:, start:end], acc_data[:, start:end])
    nrmse_ad = nrmse_std(ad_Y[:, start:end], acc_data[:, start:end])
    print(f'{ex}, rc {nrmse_rc}, ad {nrmse_ad}')


def plot_ori_ad_phase():
  noise_t, td_Y = read_noise_data('./datas/lorenz_noise_snr60.csv')
  td_Y = td_Y.T

  ad_T = torch.load('./datas/T_25.pt').detach().numpy()
  ad_Y = torch.load('./datas/ad_Y_25.pt').detach().numpy()

  trained_net = torch.load(f'./model/model_rb_pre_without_model_ex7.pth')
  flt_Y = trained_net(V(torch.from_numpy(ad_T)).double().reshape(-1, 1)).detach().numpy()

  start = 400
  color = ['#0000cc', '#008800', '#ff0000']
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot(td_Y[start:, 0], td_Y[start:, 1], td_Y[start:, 2], c=color[0], linewidth=1)
  ax.plot(ad_Y[start:, 0], ad_Y[start:, 1], ad_Y[start:, 2], c=color[2], linewidth=1)
  fig.savefig(f'./error_pic/ori_ad_phase.png', dpi=500)
  fig.savefig(f'./error_pic/ori_ad_phase.svg', dpi=500)


def plot_ori_flt_phase():
  noise_t, td_Y = read_noise_data('./datas/lorenz_noise_snr60.csv')
  td_Y = td_Y.T

  trained_net = torch.load(f'./model/model_rb_pre_without_model_ex7.pth')
  flt_Y = trained_net(V(torch.from_numpy(noise_t)).double().reshape(-1, 1)).detach().numpy()

  start = 0
  end = 401
  color = ['#0000cc', '#008800', '#ff0000']
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot(td_Y[start:end, 0], td_Y[start:end, 1], td_Y[start:end, 2], c=color[0], linewidth=1)
  ax.plot(flt_Y[start:end, 0], flt_Y[start:end, 1], flt_Y[start:end, 2], c=color[2], linewidth=1)
  ax.set_xlabel('x')
  fig.savefig(f'./error_pic/ori_flt_phase.png', dpi=500)
  fig.savefig(f'./error_pic/ori_flt_phase.svg', dpi=500)


def plot_nonoise_noise_phase():
  noise_t, noise_Y = read_noise_data('./datas/lorenz_noise_snr60.csv')
  nonoise_t, nonoise_Y = read_noise_data('./datas/lorenz_25.csv')
  noise_Y = noise_Y.T
  nonoise_Y = nonoise_Y.T
  print(nonoise_Y.shape)

  start = 400
  color = ['#0000cc', '#008800', '#ff0000']
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot(nonoise_Y[start:, 0], nonoise_Y[start:, 1], nonoise_Y[start:, 2], c=color[0], linewidth=1)
  ax.scatter(noise_Y[start:, 0], noise_Y[start:, 1], noise_Y[start:, 2], c=color[2], linewidth=1, s=1)
  fig.savefig(f'./error_pic/nonoise_noise_scatter_phase.png', dpi=500)
  fig.savefig(f'./error_pic/nonoise_noise_scatter_phase.svg', dpi=500)

  
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot(nonoise_Y[start:, 0], nonoise_Y[start:, 1], nonoise_Y[start:, 2], c=color[0], linewidth=1)
  ax.plot(noise_Y[start:, 0], noise_Y[start:, 1], noise_Y[start:, 2], c=color[2], linewidth=1, linestyle='--')
  fig.savefig(f'./error_pic/nonoise_noise_phase.png', dpi=500)
  fig.savefig(f'./error_pic/nonoise_noise_phase.svg', dpi=500)


def plot_ad_cmp():
  td_T, td_Y = read_noise_data('./datas/lorenz_noise_snr60.csv')

  ad_T = torch.load('./datas/T_35.pt').detach().numpy()
  ad_Y = torch.load('./datas/ad_Y_35.pt').detach().numpy().T

  start = 400
  xticks = (10, 12, 14, 16, 18, 20, 22, 24, 25)
  color = ['#0000cc', '#008800', '#ff0000']
  fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
  ax1.plot(td_T[start:], td_Y[0, start:], c=color[0], linestyle='-')
  ax1.plot(ad_T[start:], ad_Y[0, start:], c=color[2], linestyle='-.')
  ax1.set_xlim((10,25))
  ax1.set_xticks(xticks)

  ax2.plot(td_T[start:], td_Y[1, start:], c=color[0], linestyle='-')
  ax2.plot(ad_T[start:], ad_Y[1, start:], c=color[2], linestyle='-.')
  ax2.set_xlim((10,25))
  ax2.set_xticks(xticks)

  ax3.plot(td_T[start:], td_Y[2, start:], c=color[0], linestyle='-')
  ax3.plot(ad_T[start:], ad_Y[2, start:], c=color[2], linestyle='-.')
  ax3.set_xlim((10,25))
  ax3.set_xticks(xticks)

  fig2.savefig(f'./img/ad_cmp.svg', dpi=500)
  fig2.savefig(f'./img/ad_cmp.png', dpi=500)


def plot_rc_cmp():
  td_T, td_Y = read_noise_data('./datas/lorenz_noise_snr60.csv')

  oriRC_T = torch.load('./datas/oriRC_predict_T').detach().numpy()
  oriRC_Y = torch.load('./datas/oriRC_predict_Y').detach().numpy().T

  start = 400
  xticks = (10, 12, 14, 16, 18, 20, 22, 24, 25)
  color = ['#0000cc', '#008800', '#ff0000']
  fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
  ax1.plot(td_T[start:], td_Y[0, start:], c=color[0], linestyle='-')
  ax1.plot(oriRC_T[start:], oriRC_Y[0, start:], c=color[2], linestyle='-.')
  ax1.set_xlim((10,25))
  ax1.set_xticks(xticks)

  ax2.plot(td_T[start:], td_Y[1, start:], c=color[0], linestyle='-')
  ax2.plot(oriRC_T[start:], oriRC_Y[1, start:], c=color[2], linestyle='-.')
  ax2.set_xlim((10,25))
  ax2.set_xticks(xticks)

  ax3.plot(td_T[start:], td_Y[2, start:], c=color[0], linestyle='-')
  ax3.plot(oriRC_T[start:], oriRC_Y[2, start:], c=color[2], linestyle='-.')
  ax3.set_xlim((10,25))
  ax3.set_xticks(xticks)

  fig2.savefig(f'./img/rc_cmp.svg', dpi=500)
  fig2.savefig(f'./img/rc_cmp.png', dpi=500)


def plot_id_cmp():
  td_T, td_Y = read_noise_data('./datas/lorenz_noise_snr60.csv')

  id_T = torch.load('./datas/id_T.pt').detach().numpy()
  id_Y = torch.load('./datas/id_Y.pt').detach().numpy()

  start = 400
  xticks = (10, 12, 14, 16, 18, 20, 22, 24, 25)
  color = ['#0000cc', '#008800', '#ff0000']
  fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
  ax1.plot(td_T[start:], td_Y[0, start:], c=color[0], linestyle='-')
  ax1.plot(id_T[start:], id_Y[0, start:], c=color[2], linestyle='-.')
  ax1.set_xlim((10,25))
  ax1.set_xticks(xticks)

  ax2.plot(td_T[start:], td_Y[1, start:], c=color[0], linestyle='-')
  ax2.plot(id_T[start:], id_Y[1, start:], c=color[2], linestyle='-.')
  ax2.set_xlim((10,25))
  ax2.set_xticks(xticks)

  ax3.plot(td_T[start:], td_Y[2, start:], c=color[0], linestyle='-')
  ax3.plot(id_T[start:], id_Y[2, start:], c=color[2], linestyle='-.')
  ax3.set_xlim((10,25))
  ax3.set_xticks(xticks)

  fig2.savefig(f'./img/id_cmp.svg', dpi=500)
  fig2.savefig(f'./img/id_cmp.png', dpi=500)


def plot_pinn_cmp():
  td_T, td_Y = read_noise_data('./datas/lorenz_noise_snr60.csv')

  ex = 1
  pinn_T = torch.load(f'./datas/T_PINN{ex}.pt').detach().numpy()
  pinn_Y = torch.load(f'./datas/Y_PINN{ex}.pt').detach().numpy().T

  start = 400
  xticks = (10, 12, 14, 16, 18, 20, 22, 24, 25)
  color = ['#0000cc', '#008800', '#ff0000']
  fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
  ax1.plot(td_T[start:], td_Y[0, start:], c=color[0], linestyle='-')
  ax1.plot(pinn_T[start:], pinn_Y[0, start:], c=color[2], linestyle='-.')
  ax1.set_xlim((10,25))
  ax1.set_xticks(xticks)

  ax2.plot(td_T[start:], td_Y[1, start:], c=color[0], linestyle='-')
  ax2.plot(pinn_T[start:], pinn_Y[1, start:], c=color[2], linestyle='-.')
  ax2.set_xlim((10,25))
  ax2.set_xticks(xticks)

  ax3.plot(td_T[start:], td_Y[2, start:], c=color[0], linestyle='-')
  ax3.plot(pinn_T[start:], pinn_Y[2, start:], c=color[2], linestyle='-.')
  ax3.set_xlim((10,25))
  ax3.set_xticks(xticks)

  fig2.savefig(f'./img/pinn_cmp.svg', dpi=500)
  fig2.savefig(f'./img/pinn_cmp.png', dpi=500)


if __name__=='__main__':
  
  # compute_filter()
  # compute_nrmse()

  # # 绘制原始数据和滤波后数据的RC图像对比
  # plot_oriRC_fltRC_cmp()
  # # 绘制原始数据RC图像和滤波后并使用微分方程调整后的图像对比
  # plot_oriRC_ad_cmp()
  # # 绘制滤波后RC图像和调整后图像对比
  # plot_fltRC_ad_cmp()

  # 绘制原始数据和各预测方法结果的对比
  # plot_ad_cmp()
  # plot_rc_cmp()
  # plot_id_cmp()
  # plot_pinn_cmp()

  # plot_ori_ad_phase()
  # plot_ori_flt_phase()
  plot_nonoise_noise_phase()
  
  # # 绘制原始数据RC图像和调整后图像绝对值偏差对比
  # plot_oriRC_ad_abs_cmp()
  # # 绘制原始数据RC图像和调整后图像微分方程偏差对比
  # plot_oriRC_ad_de_cmp()

  # # 绘制降噪数据RC图像和调整后图像绝对值偏差对比
  # plot_fltRC_ad_abs_cmp()
  # # 绘制降噪数据RC图像和调整后图像微分方程偏差对比
  # plot_fltRC_ad_de_cmp()

  