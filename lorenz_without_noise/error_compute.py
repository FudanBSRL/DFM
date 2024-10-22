import torch
import os
import numpy as np
from lorenz_ode import lorenz_ode
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data_util import read_data
import matplotlib
import matplotlib.font_manager as fm


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


def plot_real_ode_phase():
  current_train = 25
  target_data = lorenz_ode(time_span=(0, current_train), ic=[-8., 7., 27.], t_eval=torch.linspace(0, current_train, int(40 * current_train + 1)), method='RK45')
  td_T = target_data.t
  td_Y = target_data.y.T

  id_T = torch.load('./datas/id_T.pt').detach().numpy()
  id_Y = torch.load('./datas/id_Y.pt').detach().numpy().T

  start = 400
  color = ['#0000cc', '#008800', '#ff0000']
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot(td_Y[start:, 0], td_Y[start:, 1], td_Y[start:, 2], c=color[0], linewidth=1)
  ax.plot(id_Y[start:, 0], id_Y[start:, 1], id_Y[start:, 2], c=color[2], linewidth=1)
  fig.savefig(f'./error_pic/real_ode_phase.png', dpi=500)
  fig.savefig(f'./error_pic/real_ode_phase.svg', dpi=500)


def plot_real_pinn_phase(ex):
  current_train = 25
  target_data = lorenz_ode(time_span=(0, current_train), ic=[-8., 7., 27.], t_eval=torch.linspace(0, current_train, int(40 * current_train + 1)), method='RK45')
  td_T = target_data.t
  td_Y = target_data.y.T

  pinn_T = torch.load(f'./datas/T_PINN{ex}.pt').detach().numpy()
  pinn_Y = torch.load(f'./datas/Y_PINN{ex}.pt').detach().numpy()

  start = 400
  color = ['#0000cc', '#008800', '#ff0000']
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot(td_Y[start:, 0], td_Y[start:, 1], td_Y[start:, 2], c=color[0], linewidth=1)
  ax.plot(pinn_Y[start:, 0], pinn_Y[start:, 1], pinn_Y[start:, 2], c=color[2], linewidth=1)
  fig.savefig(f'./error_pic/real_rc_phase.png', dpi=500)
  fig.savefig(f'./error_pic/real_rc_phase.svg', dpi=500)


def plot_real_rc_phase():
  current_train = 25
  target_data = lorenz_ode(time_span=(0, current_train), ic=[-8., 7., 27.], t_eval=torch.linspace(0, current_train, int(40 * current_train + 1)), method='RK45')
  td_T = target_data.t
  td_Y = target_data.y.T

  rc_T = torch.load('./datas/rc_predict_T').detach().numpy()
  rc_Y = torch.load('./datas/rc_predict_Y').detach().numpy()

  start = 400
  color = ['#0000cc', '#008800', '#ff0000']
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot(td_Y[start:, 0], td_Y[start:, 1], td_Y[start:, 2], c=color[0], linewidth=1)
  ax.plot(rc_Y[start:, 0], rc_Y[start:, 1], rc_Y[start:, 2], c=color[2], linewidth=1)
  fig.savefig(f'./error_pic/real_pinn_phase.png', dpi=500)
  fig.savefig(f'./error_pic/real_pinn_phase.svg', dpi=500)

  
def plot_real_ad_phase(ex):
  current_train = 25
  target_data = lorenz_ode(time_span=(0, current_train), ic=[-8., 7., 27.], t_eval=torch.linspace(0, current_train, int(40 * current_train + 1)), method='RK45')
  td_T = target_data.t
  td_Y = target_data.y.T

  ad_T = torch.load(f'./datas/T_{ex}.pt').detach().numpy()
  ad_Y = torch.load(f'./datas/ad_Y_{ex}.pt').detach().numpy()

  start = 400
  color = ['#0000cc', '#008800', '#ff0000']
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot(td_Y[start:, 0], td_Y[start:, 1], td_Y[start:, 2], c=color[0], linewidth=1)
  ax.plot(ad_Y[start:, 0], ad_Y[start:, 1], ad_Y[start:, 2], c=color[2], linewidth=1)
  fig.savefig(f'./error_pic/real_ad_phase.png', dpi=500)
  fig.savefig(f'./error_pic/real_ad_phase.svg', dpi=500)


def plot_phase(pinn_ex, ad_ex):
  td_T, td_Y = read_data(f'./datas/lorenz_25.csv')
  td_Y = td_Y.T

  # rc_T = torch.load(f'./datas/rc_predict_T_{ad_ex}').detach().numpy()
  rc_Y = torch.load(f'./datas/predict_Y_{ad_ex}.pt').detach().numpy()
  
  id_T = torch.load('./datas/id_T.pt').detach().numpy()
  id_Y = torch.load('./datas/id_Y.pt').detach().numpy().T

  pinn_T = torch.load(f'./datas/T_PINN{pinn_ex}.pt').detach().numpy()
  pinn_Y = torch.load(f'./datas/Y_PINN{pinn_ex}.pt').detach().numpy()

  ad_T = torch.load(f'./datas/T_{ad_ex}.pt').detach().numpy()
  ad_Y = torch.load(f'./datas/ad_Y_{ad_ex}.pt').detach().numpy()

  arial_font = fm.FontProperties(fname='/usr/share/fonts/truetype/msttcorefonts/Arial.ttf')
  matplotlib.rcParams['font.family'] = arial_font.get_name()

  start = 400
  end = 880
  color = ['#0000cc', '#000000', '#ff0000']
  fig = plt.figure()
  ax1 = fig.add_subplot(221, projection='3d')
  ax1.plot(td_Y[start:end, 0], td_Y[start:end, 1], td_Y[start:end, 2], c=color[0], linewidth=1)
  ax1.plot(rc_Y[start:end, 0], rc_Y[start:end, 1], rc_Y[start:end, 2], c=color[2], linewidth=1)
  for i in range(start, end):
    ax1.plot([td_Y[i, 0], rc_Y[i, 0]], [td_Y[i, 1], rc_Y[i, 1]], [td_Y[i, 2], rc_Y[i, 2]], c=color[1], linewidth=1)
  ax1.set_xlabel('x')
  ax1.set_ylabel('y')
  ax1.set_zlabel('z')

  ax2 = fig.add_subplot(222, projection='3d')
  ax2.plot(td_Y[start:end, 0], td_Y[start:end, 1], td_Y[start:end, 2], c=color[0], linewidth=1)
  ax2.plot(id_Y[start:end, 0], id_Y[start:end, 1], id_Y[start:end, 2], c=color[2], linewidth=1)
  for i in range(start, end):
    ax2.plot([td_Y[i, 0], id_Y[i, 0]], [td_Y[i, 1], id_Y[i, 1]], [td_Y[i, 2], id_Y[i, 2]], c=color[1], linewidth=1)
  ax2.set_xlabel('x')
  ax2.set_ylabel('y')
  ax2.set_zlabel('z')

  ax3 = fig.add_subplot(223, projection='3d')
  ax3.plot(td_Y[start:end, 0], td_Y[start:end, 1], td_Y[start:end, 2], c=color[0], linewidth=1)
  ax3.plot(pinn_Y[start:end, 0], pinn_Y[start:end, 1], pinn_Y[start:end, 2], c=color[2], linewidth=1)
  for i in range(start, end):
    ax3.plot([td_Y[i, 0], pinn_Y[i, 0]], [td_Y[i, 1], pinn_Y[i, 1]], [td_Y[i, 2], pinn_Y[i, 2]], c=color[1], linewidth=1)
  ax3.set_xlabel('x')
  ax3.set_ylabel('y')
  ax3.set_zlabel('z')
    
  ax4 = fig.add_subplot(224, projection='3d')
  ax4.plot(td_Y[start:end, 0], td_Y[start:end, 1], td_Y[start:end, 2], c=color[0], linewidth=1)
  ax4.plot(ad_Y[start:end, 0], ad_Y[start:end, 1], ad_Y[start:end, 2], c=color[2], linewidth=1)
  for i in range(start, end):
    ax4.plot([td_Y[i, 0], ad_Y[i, 0]], [td_Y[i, 1], ad_Y[i, 1]], [td_Y[i, 2], ad_Y[i, 2]], c=color[1], linewidth=1)
  ax4.set_xlabel('x')
  ax4.set_ylabel('y')
  ax4.set_zlabel('z')
  
  fig.savefig(f'./error_pic/phase.png', dpi=500)
  fig.savefig(f'./error_pic/phase.svg', dpi=500)


def compare_real_ode_pinn_rc_ad(ex_pinn, ex_ad):
  current_train = 25
  target_data = lorenz_ode(time_span=(0, current_train), ic=[-8., 7., 27.], t_eval=torch.linspace(0, current_train, int(40 * current_train + 1)), method='RK45')
  td_T = target_data.t
  td_Y = target_data.y
  
  id_T = torch.load('./datas/id_T.pt').detach().numpy()
  id_Y = torch.load('./datas/id_Y.pt').detach().numpy()
  
  pinn_T = torch.load(f'./datas/T_PINN{ex_pinn}.pt').detach().numpy()
  pinn_Y = torch.load(f'./datas/Y_PINN{ex_pinn}.pt').detach().numpy().T

  rc_T = torch.load('./datas/rc_predict_T').detach().numpy()
  rc_Y = torch.load('./datas/rc_predict_Y').detach().numpy().T

  ad_T = torch.load(f'./datas/T_{ex_ad}.pt').detach().numpy()
  ad_Y = torch.load(f'./datas/ad_Y_{ex_ad}.pt').detach().numpy().T

  start = 400
  xticks = (10, 12, 14, 16, 18, 20, 22, 24, 25)
  color = ['#0000cc', '#008800', '#ff0000', '#ff8800', '#ff0088cc']
  fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
  ax1.plot(td_T[start:], td_Y[0, start:], c=color[0], linestyle='-')
  ax1.plot(rc_T[start:], rc_Y[0, start:], c=color[1], linestyle='--')
  ax1.plot(ad_T[start:], ad_Y[0, start:], c=color[2], linestyle='-.')
  ax1.plot(id_T[:], id_Y[0, :], c=color[3], linestyle='-.')
  ax1.plot(pinn_T[start:], pinn_Y[0, start:], c=color[4], linestyle='-.')
  ax1.set_xlim((10,25))
  ax1.set_xticks(xticks)

  ax2.plot(td_T[start:], td_Y[1, start:], c=color[0], linestyle='-')
  ax2.plot(rc_T[start:], rc_Y[1, start:], c=color[1], linestyle='--')
  ax2.plot(ad_T[start:], ad_Y[1, start:], c=color[2], linestyle='-.')
  ax2.plot(id_T[:], id_Y[1, :], c=color[3], linestyle='-.')
  ax2.plot(pinn_T[start:], pinn_Y[1, start:], c=color[4], linestyle='-.')
  ax2.set_xlim((10,25))
  ax2.set_xticks(xticks)

  ax3.plot(td_T[start:], td_Y[2, start:], c=color[0], linestyle='-')
  ax3.plot(rc_T[start:], rc_Y[2, start:], c=color[1], linestyle='--')
  ax3.plot(ad_T[start:], ad_Y[2, start:], c=color[2], linestyle='-.')
  ax3.plot(id_T[:], id_Y[2, :], c=color[3], linestyle='-.')
  ax3.plot(pinn_T[start:], pinn_Y[2, start:], c=color[4], linestyle='-.')
  ax3.set_xlim((10,25))
  ax3.set_xticks(xticks)

  fig2.savefig(f'./error_pic/real_ode_pinn_rc_ad.svg', dpi=500)
  fig2.savefig(f'./error_pic/real_ode_pinn_rc_ad.png', dpi=500)


def compare_real_ode():
  current_train = 25
  target_data = lorenz_ode(time_span=(0, current_train), ic=[-8., 7., 27.], t_eval=torch.linspace(0, current_train, int(40 * current_train + 1)), method='RK45')
  td_T = target_data.t
  td_Y = target_data.y
  
  id_T = torch.load('./datas/id_T.pt').detach().numpy()
  id_Y = torch.load('./datas/id_Y.pt').detach().numpy()

  start = 400
  xticks = (10, 12, 14, 16, 18, 20, 22, 24, 25)
  color = ['#0000cc', '#008800', '#ff0000', '#ff8800', '#ff0088cc']
  fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
  ax1.plot(td_T[start:], td_Y[0, start:], c=color[0], linestyle='-')
  ax1.plot(id_T[:], id_Y[0, :], c=color[2], linestyle='--')
  ax1.set_xlim((10,25))
  ax1.set_xticks(xticks)

  ax2.plot(td_T[start:], td_Y[1, start:], c=color[0], linestyle='-')
  ax2.plot(id_T[:], id_Y[1, :], c=color[2], linestyle='--')
  ax2.set_xlim((10,25))
  ax2.set_xticks(xticks)

  ax3.plot(td_T[start:], td_Y[2, start:], c=color[0], linestyle='-')
  ax3.plot(id_T[:], id_Y[2, :], c=color[2], linestyle='--')
  ax3.set_xlim((10,25))
  ax3.set_xticks(xticks)

  fig2.savefig(f'./error_pic/real_ode.svg', dpi=500)
  fig2.savefig(f'./error_pic/real_ode.png', dpi=500)


def compare_real_pinn(ex_pinn):
  current_train = 25
  target_data = lorenz_ode(time_span=(0, current_train), ic=[-8., 7., 27.], t_eval=torch.linspace(0, current_train, int(40 * current_train + 1)), method='RK45')
  td_T = target_data.t
  td_Y = target_data.y
  
  pinn_T = torch.load(f'./datas/T_PINN{ex_pinn}.pt').detach().numpy()
  pinn_Y = torch.load(f'./datas/Y_PINN{ex_pinn}.pt').detach().numpy().T

  start = 400
  xticks = (10, 12, 14, 16, 18, 20, 22, 24, 25)
  color = ['#0000cc', '#008800', '#ff0000', '#ff8800', '#ff0088cc']
  fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
  ax1.plot(td_T[start:], td_Y[0, start:], c=color[0], linestyle='-')
  ax1.plot(pinn_T[start:], pinn_Y[0, start:], c=color[2], linestyle='--')
  ax1.set_xlim((10,25))
  ax1.set_xticks(xticks)

  ax2.plot(td_T[start:], td_Y[1, start:], c=color[0], linestyle='-')
  ax2.plot(pinn_T[start:], pinn_Y[1, start:], c=color[2], linestyle='--')
  ax2.set_xlim((10,25))
  ax2.set_xticks(xticks)

  ax3.plot(td_T[start:], td_Y[2, start:], c=color[0], linestyle='-')
  ax3.plot(pinn_T[start:], pinn_Y[2, start:], c=color[2], linestyle='--')
  ax3.set_xlim((10,25))
  ax3.set_xticks(xticks)

  fig2.savefig(f'./error_pic/real_pinn.svg', dpi=500)
  fig2.savefig(f'./error_pic/real_pinn.png', dpi=500)


def compare_real_rc():
  current_train = 25
  target_data = lorenz_ode(time_span=(0, current_train), ic=[-8., 7., 27.], t_eval=torch.linspace(0, current_train, int(40 * current_train + 1)), method='RK45')
  td_T = target_data.t
  td_Y = target_data.y

  rc_T = torch.load('./datas/rc_predict_T').detach().numpy()
  rc_Y = torch.load('./datas/rc_predict_Y').detach().numpy().T

  start = 400
  xticks = (10, 12, 14, 16, 18, 20, 22, 24, 25)
  color = ['#0000cc', '#008800', '#ff0000', '#ff8800', '#ff0088cc']
  fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
  ax1.plot(td_T[start:], td_Y[0, start:], c=color[0], linestyle='-')
  ax1.plot(rc_T[start:], rc_Y[0, start:], c=color[2], linestyle='--')
  ax1.set_xlim((10,25))
  ax1.set_xticks(xticks)

  ax2.plot(td_T[start:], td_Y[1, start:], c=color[0], linestyle='-')
  ax2.plot(rc_T[start:], rc_Y[1, start:], c=color[2], linestyle='--')
  ax2.set_xlim((10,25))
  ax2.set_xticks(xticks)

  ax3.plot(td_T[start:], td_Y[2, start:], c=color[0], linestyle='-')
  ax3.plot(rc_T[start:], rc_Y[2, start:], c=color[2], linestyle='--')
  ax3.set_xlim((10,25))
  ax3.set_xticks(xticks)

  fig2.savefig(f'./error_pic/real_rc.svg', dpi=500)
  fig2.savefig(f'./error_pic/real_rc.png', dpi=500)


def compare_real_ad(ex_ad):
  current_train = 25
  target_data = lorenz_ode(time_span=(0, current_train), ic=[-8., 7., 27.], t_eval=torch.linspace(0, current_train, int(40 * current_train + 1)), method='RK45')
  td_T = target_data.t
  td_Y = target_data.y

  ad_T = torch.load(f'./datas/T_{ex_ad}.pt').detach().numpy()
  ad_Y = torch.load(f'./datas/ad_Y_{ex_ad}.pt').detach().numpy().T

  start = 400
  xticks = (10, 12, 14, 16, 18, 20, 22, 24, 25)
  color = ['#0000cc', '#008800', '#ff0000', '#ff8800', '#ff0088cc']
  fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
  ax1.plot(td_T[start:], td_Y[0, start:], c=color[0], linestyle='-')
  ax1.plot(ad_T[start:], ad_Y[0, start:], c=color[2], linestyle='--')
  ax1.set_xlim((10,25))
  ax1.set_xticks(xticks)

  ax2.plot(td_T[start:], td_Y[1, start:], c=color[0], linestyle='-')
  ax2.plot(ad_T[start:], ad_Y[1, start:], c=color[2], linestyle='--')
  ax2.set_xlim((10,25))
  ax2.set_xticks(xticks)

  ax3.plot(td_T[start:], td_Y[2, start:], c=color[0], linestyle='-')
  ax3.plot(ad_T[start:], ad_Y[2, start:], c=color[2], linestyle='--')
  ax3.set_xlim((10,25))
  ax3.set_xticks(xticks)

  fig2.savefig(f'./error_pic/real_ad.svg', dpi=500)
  fig2.savefig(f'./error_pic/real_ad.png', dpi=500)


def plot_abs_error_rc_ad(ex_ad):
  current_train = 25
  target_data = lorenz_ode(time_span=(0, current_train), ic=[-8., 7., 27.], t_eval=torch.linspace(0, current_train, int(40 * current_train + 1)), method='RK45')
  td_T = target_data.t
  td_Y = target_data.y

  ad_T = torch.load(f'./datas/T_{ex_ad}.pt').detach().numpy()
  ad_Y = torch.load(f'./datas/ad_Y_{ex_ad}.pt').detach().numpy().T
  err_ad = np.abs(td_Y - ad_Y)

  rc_T = torch.load('./datas/rc_predict_T').detach().numpy()
  rc_Y = torch.load('./datas/rc_predict_Y').detach().numpy().T
  err_rc = np.abs(td_Y - rc_Y)

  start = 400
  xticks = (10, 12, 14, 16, 18, 20, 22, 24, 25)
  color = ['#0000cc', '#008800', '#ff0000']
  fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
  ax1.plot(rc_T[start:], err_rc[0, start:], c=color[0], linestyle='-')
  ax1.plot(ad_T[start:], err_ad[0, start:], c=color[2], linestyle='-.')
  ax1.set_xlim((10,25))
  ax1.set_xticks(xticks)

  ax2.plot(rc_T[start:], err_rc[1, start:], c=color[0], linestyle='-')
  ax2.plot(ad_T[start:], err_ad[1, start:], c=color[2], linestyle='-.')
  ax2.set_xlim((10,25))
  ax2.set_xticks(xticks)

  ax3.plot(rc_T[start:], err_rc[2, start:], c=color[0], linestyle='-')
  ax3.plot(ad_T[start:], err_ad[2, start:], c=color[2], linestyle='-.')
  ax3.set_xlim((10,25))
  ax3.set_xticks(xticks)

  fig2.savefig(f'./error_pic/abs_error_rc_ad.svg', dpi=500)
  fig2.savefig(f'./error_pic/abs_error_rc_ad.png', dpi=500)

  keep = 240
  xlim = (10, 16)
  # xticks = (10, 12, 14)
  color = ['#0000cc', '#008800', '#ff0000']
  fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
  ax1.plot(rc_T[start:start+keep], err_rc[0, start:start+keep], c=color[0], linestyle='-')
  ax1.plot(ad_T[start:start+keep], err_ad[0, start:start+keep], c=color[2], linestyle='-')
  ax1.set_xlim(xlim)
  # ax1.set_xticks(xticks)

  ax2.plot(rc_T[start:start+keep], err_rc[1, start:start+keep], c=color[0], linestyle='-')
  ax2.plot(ad_T[start:start+keep], err_ad[1, start:start+keep], c=color[2], linestyle='-')
  ax2.set_xlim(xlim)
  # ax2.set_xticks(xticks)

  ax3.plot(rc_T[start:start+keep], err_rc[2, start:start+keep], c=color[0], linestyle='-')
  ax3.plot(ad_T[start:start+keep], err_ad[2, start:start+keep], c=color[2], linestyle='-')
  ax3.set_xlim(xlim)
  # ax3.set_xticks(xticks)

  fig2.savefig(f'./error_pic/abs_error_rc_ad_16.svg', dpi=500)
  fig2.savefig(f'./error_pic/abs_error_rc_ad_16.png', dpi=500)


def plot_abs_error_rc_ode45_pinn_ad(ex_pinn, ex_ad):
  current_train = 25
  td_T, td_Y = read_data(f'./datas/lorenz_25.csv')
  # td_T = target_data.t
  # td_Y = target_data.y

  ad_T = torch.load(f'./datas/T_{ex_ad}.pt').detach().numpy()
  ad_Y = torch.load(f'./datas/ad_Y_{ex_ad}.pt').detach().numpy().T
  err_ad = np.abs(td_Y - ad_Y)

  rc_T = torch.load('./datas/rc_predict_T').detach().numpy()
  rc_Y = torch.load('./datas/rc_predict_Y').detach().numpy().T
  err_rc = np.abs(td_Y - rc_Y)

  pinn_T = torch.load(f'./datas/T_PINN{ex_pinn}.pt').detach().numpy()
  pinn_Y = torch.load(f'./datas/Y_PINN{ex_pinn}.pt').detach().numpy().T
  err_pinn = np.abs(td_Y - pinn_Y)

  ode_T = torch.load('./datas/id_T.pt').detach().numpy()
  ode_Y = torch.load('./datas/id_Y.pt').detach().numpy()
  err_ode = np.abs(td_Y - ode_Y)

  start = 400
  xticks = (10, 12, 14, 16, 18, 20, 22, 24, 25)
  color = ['#0000cc', '#008800', '#ff8800', '#ff0000']
  fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
  ax1.plot(rc_T[start:], err_rc[0, start:], c=color[0], linestyle='-')
  ax1.plot(ode_T[start:], err_pinn[0, start:], c=color[1], linestyle='-')
  ax1.plot(pinn_T[start:], err_ode[0, start:], c=color[2], linestyle='-')
  ax1.plot(ad_T[start:], err_ad[0, start:], c=color[3], linestyle='-')
  ax1.set_xlim((10,25))
  ax1.set_xticks(xticks)

  ax2.plot(rc_T[start:], err_rc[1, start:], c=color[0], linestyle='-')
  ax2.plot(ode_T[start:], err_pinn[1, start:], c=color[1], linestyle='-')
  ax2.plot(pinn_T[start:], err_ode[1, start:], c=color[2], linestyle='-')
  ax2.plot(ad_T[start:], err_ad[1, start:], c=color[3], linestyle='-')
  ax2.set_xlim((10,25))
  ax2.set_xticks(xticks)

  ax3.plot(rc_T[start:], err_rc[2, start:], c=color[0], linestyle='-')
  ax3.plot(ode_T[start:], err_pinn[2, start:], c=color[1], linestyle='-')
  ax3.plot(pinn_T[start:], err_ode[2, start:], c=color[2], linestyle='-')
  ax3.plot(ad_T[start:], err_ad[2, start:], c=color[3], linestyle='-')
  ax3.set_xlim((10,25))
  ax3.set_xticks(xticks)

  fig2.savefig(f'./error_pic/abs_error_rc_ode45_pinn_ad.svg', dpi=500)
  fig2.savefig(f'./error_pic/abs_error_rc_ode45_pinn_ad.png', dpi=500)

  keep = 240
  xlim = (10, 16)
  # xticks = (10, 12, 14)
  color = ['#0000cc', '#008800', '#ff8800', '#ff0000']
  fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
  ax1.plot(rc_T[start:start+keep], err_rc[0, start:start+keep], c=color[0], linestyle='-')
  ax1.plot(ode_T[start:start+keep], err_pinn[0, start:start+keep], c=color[1], linestyle='-')
  ax1.plot(pinn_T[start:start+keep], err_ode[0, start:start+keep], c=color[2], linestyle='-')
  ax1.plot(ad_T[start:start+keep], err_ad[0, start:start+keep], c=color[3], linestyle='-')
  ax1.set_xlim(xlim)
  # ax1.set_xticks(xticks)

  ax2.plot(rc_T[start:start+keep], err_rc[1, start:start+keep], c=color[0], linestyle='-')
  ax2.plot(ode_T[start:start+keep], err_pinn[1, start:start+keep], c=color[1], linestyle='-')
  ax2.plot(pinn_T[start:start+keep], err_ode[1, start:start+keep], c=color[2], linestyle='-')
  ax2.plot(ad_T[start:start+keep], err_ad[1, start:start+keep], c=color[3], linestyle='-')
  ax2.set_xlim(xlim)
  # ax2.set_xticks(xticks)

  ax3.plot(rc_T[start:start+keep], err_rc[2, start:start+keep], c=color[0], linestyle='-')
  ax3.plot(ode_T[start:start+keep], err_pinn[2, start:start+keep], c=color[1], linestyle='-')
  ax3.plot(pinn_T[start:start+keep], err_ode[2, start:start+keep], c=color[2], linestyle='-')
  ax3.plot(ad_T[start:start+keep], err_ad[2, start:start+keep], c=color[3], linestyle='-')
  ax3.set_xlim(xlim)
  # ax3.set_xticks(xticks)

  fig2.savefig(f'./error_pic/abs_error_rc_ode45_pinn_ad_16.svg', dpi=500)
  fig2.savefig(f'./error_pic/abs_error_rc_ode45_pinn_ad_16.png', dpi=500)


def nrmse_std(predictions, targets):
    # 计算RMSE
    rmse = np.sqrt(((predictions - targets) ** 2).mean())
    # return rmse
    # print(rmse)
    # 计算每个维度的标准差
    std_of_values = np.std(targets, axis=1)
    # 由于标准差是针对每个维度的，我们可以取平均值来代表所有维度的平均标准差
    average_std = np.mean(std_of_values)
    
    # 计算NRMSE
    nrmse = rmse / average_std
    return nrmse


def compute_nrmse():
  # target_data = lorenz_ode(time_span=(0, 25), ic=[-8., 7., 27.], t_eval=torch.linspace(0, 25, 1001), method='RK45')
  # acc_data = target_data.y

  acc_t, acc_data = read_data(f'./datas/lorenz_25.csv')
  # exs = [301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311]
  # alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
  # exs = [321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331]
  # alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
  # exs = [341, 351, 361, 371, 342, 352, 362, 372, 343, 353, 363, 373, 344, 354, 364, 374, 345, 355, 365, 375, 346, 356, 366, 376, 347, 357, 367, 377]
  # windows = [0.1, 0.1, 0.1, 0.1, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75, 1, 1, 1, 1, 1.25, 1.25, 1.25, 1.25, 1.5, 1.5, 1.5, 1.5]
  # alphas = [0.95, 0.5, 0.1, 0.9, 0.95, 0.5, 0.1, 0.9, 0.95, 0.5, 0.1, 0.9, 0.95, 0.5, 0.1, 0.9, 0.95, 0.5, 0.1, 0.9, 0.95, 0.5, 0.1, 0.9, 0.95, 0.5, 0.1, 0.9]
  
  # exs = [341, 351, 361, 342, 352, 362, 343, 353, 363, 344, 354, 364, 345, 355, 365]
  # windows = [0.1, 0.1, 0.1, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 1, 1, 1]
  # alphas = [0.1, 0.5, 0.9, 0.1, 0.5, 0.9, 0.1, 0.5, 0.9, 0.1, 0.5, 0.9, 0.1, 0.5, 0.9]
  # exs = [371, 372, 373, 374, 375, 381, 382, 383, 384, 385, 391, 392, 393, 394, 395]
  # windows = [0.1, 0.25, 0.5, 0.75, 1, 0.1, 0.25, 0.5, 0.75, 1, 0.1, 0.25, 0.5, 0.75, 1]
  # alphas = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9]
  exs = [501, 502, 503, 504, 505, 506, 507, 508, 509, 510]
  windows = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
  alphas = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

  start = 4 * 40 + 1
  end = round(40 * 8.9) + 1
  for ex, window, alpha in zip(exs, windows, alphas):
    if os.path.exists(f'./datas/ad_Y_{ex}.pt'):
      predict_Y = torch.load(f'./datas/predict_Y_{ex}.pt').T.detach().numpy()
      ad_Y = torch.load(f'./datas/ad_Y_{ex}.pt').T.detach().numpy()
      end = min(round(len(ad_Y[0])), round(10 * 40) + 1)
      nrmse_rc = nrmse_std(predict_Y[:, start:end], acc_data[:, start:end])
      nrmse_ad = nrmse_std(ad_Y[:, start:end], acc_data[:, start:end])
      print(f'{ex}, alpha {alpha}, window {window}, rc {nrmse_rc}, ad {nrmse_ad}')


def compute_nrmse15():
  # target_data = lorenz_ode(time_span=(0, 25), ic=[-8., 7., 27.], t_eval=torch.linspace(0, 25, 1001), method='RK45')
  # acc_data = target_data.y
  acc_t, acc_data = read_data(f'./datas/lorenz_25.csv')

  # exs = [230, 231, 232, 233, 234, 235, 236, 241, 237, 238, 242, 239, 243, 240, 244]
  # alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.8, 0.9, 0.9, 0.95, 0.95, 0.99, 0.99]
  # exs = [243, 251, 252, 253, 254, 255, 256, 257]
  # windows = [0.5, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
  # exs = [251, 261, 271, 281, 252, 262, 272, 282, 253, 263, 273, 283, 254, 264, 274, 284, 255, 265, 275, 285, 256, 266, 276, 286, 257, 267, 277, 287]
  # windows = [0.1, 0.1, 0.1, 0.1, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75, 1, 1, 1, 1, 1.25, 1.25, 1.25, 1.25, 1.5, 1.5, 1.5, 1.5]
  # alphas = [0.95, 0.5, 0.1, 0.9, 0.95, 0.5, 0.1, 0.9, 0.95, 0.5, 0.1, 0.9, 0.95, 0.5, 0.1, 0.9, 0.95, 0.5, 0.1, 0.9, 0.95, 0.5, 0.1, 0.9, 0.95, 0.5, 0.1, 0.9]
  # exs = [251, 261, 271, 252, 262, 272, 253, 263, 273, 254, 264, 274, 255, 265, 275]
  # windows = [0.1, 0.1, 0.1, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 1, 1, 1]
  # alphas = [0.1, 0.5, 0.9, 0.1, 0.5, 0.9, 0.1, 0.5, 0.9, 0.1, 0.5, 0.9, 0.1, 0.5, 0.9]
  # exs = [401, 402, 403, 404, 405, 411, 412, 413, 414, 415, 421, 422, 423, 424, 425]
  # windows = [0.1, 0.25, 0.5, 0.75, 1, 0.1, 0.25, 0.5, 0.75, 1, 0.1, 0.25, 0.5, 0.75, 1]
  # alphas = [0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9]
  # exs = [1252]
  # windows = [0.25]
  # alphas = [0.1]
  exs = [401, 402, 403, 404, 405, 406, 407, 408, 409, 410]
  windows = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
  alphas = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

  start = 40 * 10 + 1
  end = round(40 * 18) + 1

  # PINN
  pinn_Y = torch.load(f'./datas/Y_PINN{7}.pt').detach().numpy().T
  nrmse_pinn = nrmse_std(pinn_Y[:, start:end], acc_data[:, start:end])
  print(f'pinn {nrmse_pinn}')

  # ODE45
  id_Y = torch.load(f'./datas/id_Y.pt').detach().numpy()
  # print(id_Y[:, id_start:id_end].shape, acc_data[:, start:end].shape)
  nrmse_id = nrmse_std(id_Y[:, start:end], acc_data[:, start:end])
  print(f'ode45 {nrmse_id}')

  # end = round(40 * 20) + 1
  # ours
  for ex, window, alpha in zip(exs, windows, alphas):
    if os.path.exists(f'./datas/ad_Y_{ex}.pt'):
      predict_Y = torch.load(f'./datas/predict_Y_{ex}.pt').T.detach().numpy()
      ad_Y = torch.load(f'./datas/ad_Y_{ex}.pt').T.detach().numpy()
      end = min(round(len(ad_Y[0])), round(18 * 40) + 1)
      nrmse_rc = nrmse_std(predict_Y[:, start:end], acc_data[:, start:end])
      nrmse_ad = nrmse_std(ad_Y[:, start:end], acc_data[:, start:end])
      print(f'{ex}, alpha {alpha}, window {window}, rc {nrmse_rc}, ad {nrmse_ad}')


def plot_nrmse10_20():
  acc_t, acc_data = read_data(f'./datas/lorenz_25.csv')

  start = 40 * 10 + 1
  end = round(40 * 20) + 1

  # ODE45
  id_start = 1
  id_end = round(40 * 10) + 1
  id_Y = torch.load(f'./datas/id_Y.pt').detach().numpy()
  nrmse_id = nrmse_std(id_Y[:, id_start:id_end], acc_data[:, start:end])

  alpha_list = [0]
  nrmse_list = [0]
  
  exs = [230, 231, 232, 233, 234, 235, 241, 237, 242, 239, 240]
  alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
  for ex, alpha in zip(exs, alphas):
    if os.path.exists(f'./datas/ad_Y_{ex}.pt'):
      predict_Y = torch.load(f'./datas/predict_Y_{ex}.pt').T.detach().numpy()
      ad_Y = torch.load(f'./datas/ad_Y_{ex}.pt').T.detach().numpy()
      end = min(round(len(ad_Y[0])), round(20 * 40) + 1)
      nrmse_rc = nrmse_std(predict_Y[:, start:end], acc_data[:, start:end])
      nrmse_ad = nrmse_std(ad_Y[:, start:end], acc_data[:, start:end])
      if nrmse_list[0] == 0:
        nrmse_list[0] = nrmse_rc
      alpha_list.append(alpha)
      nrmse_list.append(nrmse_ad)

  start = 40 * 10 + 1
  end = round(40 * 20) + 1
  # PINN
  pinn_Y = torch.load(f'./datas/Y_PINN{7}.pt').detach().numpy().T
  nrmse_pinn = nrmse_std(pinn_Y[:, start:end], acc_data[:, start:end])
  alpha_list.append(1)
  nrmse_list.append(nrmse_pinn)

  fig = plt.figure()
  plt.plot(alpha_list, nrmse_list, color='red')
  plt.scatter(alpha_list, nrmse_list, color='red', marker='^')
  fig.savefig(f'./img/alpha_chooce.png', dpi=500)
  fig.savefig(f'./img/alpha_chooce.svg', dpi=500)

  return alpha_list, nrmse_list


def plot_nrmse8_10():
  acc_t, acc_data = read_data(f'./datas/lorenz_25.csv')
  exs = [321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331]
  alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

  alpha_list = [0]
  nrmse_list = [0]

  start = 8 * 40 + 1
  # end = round(40 * 9.5) + 1
  for ex, alpha in zip(exs, alphas):
    if os.path.exists(f'./datas/ad_Y_{ex}.pt'):
      predict_Y = torch.load(f'./datas/predict_Y_{ex}.pt').T.detach().numpy()
      ad_Y = torch.load(f'./datas/ad_Y_{ex}.pt').T.detach().numpy()
      end = min(round(len(ad_Y[0])), 10 * 40 + 1)
      nrmse_rc = nrmse_std(predict_Y[:, start:end], acc_data[:, start:end])
      nrmse_ad = nrmse_std(ad_Y[:, start:end], acc_data[:, start:end])
      if nrmse_list[0] == 0:
        nrmse_list[0] = nrmse_rc
      alpha_list.append(alpha)
      nrmse_list.append(nrmse_ad)
  
  start = 40 * 8 + 1
  end = round(40 * 10) + 1
  # PINN
  pinn_Y = torch.load(f'./datas/Y_PINN{8}.pt').detach().numpy().T
  nrmse_pinn = nrmse_std(pinn_Y[:, start:end], acc_data[:, start:end])
  alpha_list.append(1)
  nrmse_list.append(nrmse_pinn)
    
  fig = plt.figure()
  plt.plot(alpha_list, nrmse_list, color='green')
  plt.scatter(alpha_list, nrmse_list, color='green', marker='s')
  fig.savefig(f'./img/alpha_chooce8_10.png', dpi=500)
  fig.savefig(f'./img/alpha_chooce8_10.svg', dpi=500)
  
  return alpha_list, nrmse_list


def plot_chooce_window10_20():
  acc_t, acc_data = read_data(f'./datas/lorenz_25.csv')

  start = 40 * 10 + 1
  end = round(40 * 20) + 1

  # ODE45
  id_start = 1
  id_end = round(40 * 10) + 1
  id_Y = torch.load(f'./datas/id_Y.pt').detach().numpy()
  nrmse_id = nrmse_std(id_Y[:, id_start:id_end], acc_data[:, start:end])

  window_list = []
  nrmse_list = []
  
  # exs = [251, 252, 253, 254, 255, 256, 257]
  # windows = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
  exs = [252, 253, 254, 255, 256, 257]
  windows = [0.25, 0.5, 0.75, 1, 1.25, 1.5]
  for ex, window in zip(exs, windows):
    if os.path.exists(f'./datas/ad_Y_{ex}.pt'):
      predict_Y = torch.load(f'./datas/predict_Y_{ex}.pt').T.detach().numpy()
      ad_Y = torch.load(f'./datas/ad_Y_{ex}.pt').T.detach().numpy()
      end = min(round(len(ad_Y[0])), round(20 * 40) + 1)
      nrmse_rc = nrmse_std(predict_Y[:, start:end], acc_data[:, start:end])
      nrmse_ad = nrmse_std(ad_Y[:, start:end], acc_data[:, start:end])
      window_list.append(window)
      nrmse_list.append(nrmse_ad)

  fig = plt.figure()
  plt.plot(window_list, nrmse_list, color='red')
  plt.scatter(window_list, nrmse_list, color='red', marker='^')
  fig.savefig(f'./img/window_chooce.png', dpi=500)
  fig.savefig(f'./img/window_chooce.svg', dpi=500)

  return window_list, nrmse_list


def plot_choose_window8_10():
  acc_t, acc_data = read_data(f'./datas/lorenz_25.csv')
  # exs = [341, 342, 343, 344, 345, 346, 347]
  # windows = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
  exs = [342, 343, 344, 345, 346, 347]
  windows = [0.25, 0.5, 0.75, 1, 1.25, 1.5]

  window_list = []
  nrmse_list = []

  start = 8 * 40 + 1
  # end = round(40 * 9.5) + 1
  for ex, window in zip(exs, windows):
    if os.path.exists(f'./datas/ad_Y_{ex}.pt'):
      predict_Y = torch.load(f'./datas/predict_Y_{ex}.pt').T.detach().numpy()
      ad_Y = torch.load(f'./datas/ad_Y_{ex}.pt').T.detach().numpy()
      end = min(round(len(ad_Y[0])), 10 * 40 + 1)
      nrmse_rc = nrmse_std(predict_Y[:, start:end], acc_data[:, start:end])
      nrmse_ad = nrmse_std(ad_Y[:, start:end], acc_data[:, start:end])
      window_list.append(window)
      nrmse_list.append(nrmse_ad)
    
  fig = plt.figure()
  plt.plot(window_list, nrmse_list, color='green')
  plt.scatter(window_list, nrmse_list, color='green', marker='s')
  fig.savefig(f'./img/window_chooce8_10.png', dpi=500)
  fig.savefig(f'./img/window_chooce8_10.svg', dpi=500)
  
  return window_list, nrmse_list


def plot_abs_bf_ad(ex_ad):
  td_T, td_Y = read_data(f'./datas/lorenz_25.csv')

  ad_T = torch.load(f'./datas/T_{ex_ad}.pt').detach().numpy()
  ad_Y = torch.load(f'./datas/ad_Y_{ex_ad}.pt').detach().numpy().T
  bf_Y = torch.load(f'./datas/before_ad_Y_{ex_ad}.pt').detach().numpy().T
  err_ad = np.sum((td_Y - ad_Y) ** 2, axis=0)
  err_bf = np.sum((td_Y - bf_Y) ** 2, axis=0)

  fig = plt.figure()
  start = round(40 * 10) + 1
  end = round(40 * 10.5) + 1
  plt.plot(ad_T[start:end], err_ad[start:end], color='red')
  plt.plot(ad_T[start:end], err_bf[start:end], color='green')

  start = round(40 * 10.5) + 1
  end = round(40 * 11) + 1
  plt.plot(ad_T[start:end], err_ad[start:end], color='red')
  plt.plot(ad_T[start:end], err_bf[start:end], color='green')
  
  start = round(40 * 11) + 1
  end = round(40 * 11.5) + 1
  plt.plot(ad_T[start:end], err_ad[start:end], color='red')
  plt.plot(ad_T[start:end], err_bf[start:end], color='green')

  start = round(40 * 11.5) + 1
  end = round(40 * 12) + 1
  plt.plot(ad_T[start:end], err_ad[start:end], color='red')
  plt.plot(ad_T[start:end], err_bf[start:end], color='green')

  # start = round(40 * 12) + 1
  # end = round(40 * 12.5) + 1
  # plt.plot(ad_T[start:end], err_ad[start:end], color='red')
  # plt.plot(ad_T[start:end], err_bf[start:end], color='green')

  # start = round(40 * 12.5) + 1
  # end = round(40 * 13) + 1
  # plt.plot(ad_T[start:end], err_ad[start:end], color='red')
  # plt.plot(ad_T[start:end], err_bf[start:end], color='green')

  # start = round(40 * 13) + 1
  # end = round(40 * 13.5) + 1
  # plt.plot(ad_T[start:end], err_ad[start:end], color='red')
  # plt.plot(ad_T[start:end], err_bf[start:end], color='green')

  # plt.xlim((10, 20))
  # plt.ylim((0, 0.0001))
  fig.savefig(f'./img/abs_bf_ad.png', dpi=500)
  fig.savefig(f'./img/abs_bf_ad.svg', dpi=500)


if __name__=='__main__':

  plot_real_ode_phase()
  plot_real_pinn_phase(ex=7)
  plot_real_rc_phase()
  plot_real_ad_phase(ex=243)

  plot_phase(pinn_ex=7, ad_ex=243)

  # compare_real_ode_pinn_rc_ad(ex_pinn=7, ex_ad=243)
  # compare_real_ode()
  # compare_real_pinn(ex_pinn=7)
  # compare_real_rc()
  # compare_real_ad(ex_ad=243)

  # plot_abs_error_rc_ad(ex_ad=243)
  # plot_abs_error_rc_ode45_pinn_ad(ex_pinn=7, ex_ad=243)

  # print('2s test result')
  # compute_nrmse()

  # print('\n15s predict result')
  # compute_nrmse15()

  # a1, p1 = plot_nrmse8_10()
  # a2, p2 = plot_nrmse10_20()
  # fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=False)
  # ax1.plot(a1, p1, color='green')
  # ax1.scatter(a1, p1, color='green', marker='s')
  # ax2.plot(a2, p2, color='red')
  # ax2.scatter(a2, p2, color='red', marker='^')
  # fig.savefig(f'./img/alpha_chooce_cmp.png', dpi=500)
  # fig.savefig(f'./img/alpha_chooce_cmp.svg', dpi=500)

  # a1, p1 = plot_choose_window8_10()
  # a2, p2 = plot_chooce_window10_20()
  # fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=False)
  # ax1.plot(a1, p1, color='green')
  # ax1.scatter(a1, p1, color='green', marker='s')
  # ax2.plot(a2, p2, color='red')
  # ax2.scatter(a2, p2, color='red', marker='^')
  # fig.savefig(f'./img/window_chooce_cmp.png', dpi=500)
  # fig.savefig(f'./img/window_chooce_cmp.svg', dpi=500)

  # plot_abs_bf_ad(ex_ad=243)