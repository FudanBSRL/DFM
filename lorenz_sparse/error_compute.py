import torch
import numpy as np
from lorenz_ode import lorenz_ode
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from data_util import read_data

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


def ode_abs_error():
    current_train = 25
    target_data = lorenz_ode(time_span=(0, current_train), ic=[-8., 7., 27.], t_eval=torch.linspace(0, current_train, int(100 * current_train + 1)), method='RK45')
    ic = target_data.y[:, 1000]
    # print(target_data.t[1000])
    ex = f'rb_pre_without_model_ex225'
    id_data = lorenz_ode_id(ksi_path=f'./ksis/ksi_{ex}.pt', time_span=(0, 15), ic=ic, t_eval=torch.linspace(0, 15, int(100 * 15 + 1)), method='RK45')
    id_data.t = id_data.t + 10

    # 绘制绝对误差
    keep = 500
    error_t = id_data.t[:keep]
    error_x = np.abs(target_data.y[0, 1000:1000+keep] - id_data.y[0, :keep])
    error_y = np.abs(target_data.y[1, 1000:1000+keep] - id_data.y[1, :keep])
    error_z = np.abs(target_data.y[2, 1000:1000+keep] - id_data.y[2, :keep])

    color = ['purple', 'red']
    fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=False)
    ax1.plot(error_t, error_x, c=color[1], linestyle='-')

    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    # ax1.legend(['true', 'regression', 'predicion', 'adjust'], loc='upper right')

    ax2.plot(error_t, error_y, c=color[1], linestyle='-')

    ax2.set_xlabel('t')
    ax2.set_ylabel('y')

    ax3.plot(error_t, error_y, c=color[1], linestyle='-')

    ax3.set_xlabel('t')
    ax3.set_ylabel('z')
    # fig1.legend(['true', 'ODE45'], loc='upper center', ncol=2)
    fig2.savefig('./error_pic/ode_error.png')

    return error_t, error_x, error_y, error_z


def res_compare():
  current_train = 25
  target_data = lorenz_ode(time_span=(0, current_train), ic=[-8., 7., 27.], t_eval=torch.linspace(0, current_train, int(40 * current_train + 1)), method='RK45')
  td_T = target_data.t
  td_Y = target_data.y

  ad_T = torch.load('./datas/T_21.pt').detach().numpy()
  ad_Y = torch.load('./datas/ad_Y_21.pt').detach().numpy().T

  rc_T = torch.load('./datas/rc_predict_T').detach().numpy()
  rc_Y = torch.load('./datas/rc_predict_Y').detach().numpy().T

  start = 400
  xticks = (10, 12, 14, 16, 18, 20, 22, 24, 25)
  color = ['#0000cc', '#008800', '#ff0000']
  fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
  ax1.plot(td_T[start:], td_Y[0, start:], c=color[0], linestyle='-')
  ax1.plot(rc_T[start:], rc_Y[0, start:], c=color[1], linestyle='--')
  ax1.plot(ad_T[start:], ad_Y[0, start:], c=color[2], linestyle='-.')
  ax1.set_xlim((10,25))
  ax1.set_xticks(xticks)

  ax2.plot(td_T[start:], td_Y[1, start:], c=color[0], linestyle='-')
  ax2.plot(rc_T[start:], rc_Y[1, start:], c=color[1], linestyle='--')
  ax2.plot(ad_T[start:], ad_Y[1, start:], c=color[2], linestyle='-.')
  ax2.set_xlim((10,25))
  ax2.set_xticks(xticks)

  ax3.plot(td_T[start:], td_Y[2, start:], c=color[0], linestyle='-')
  ax3.plot(rc_T[start:], rc_Y[2, start:], c=color[1], linestyle='--')
  ax3.plot(ad_T[start:], ad_Y[2, start:], c=color[2], linestyle='-.')
  ax3.set_xlim((10,25))
  ax3.set_xticks(xticks)

  fig2.savefig(f'./error_pic/res.svg')
  fig2.savefig(f'./error_pic/res.png')


def ad_abs_error():
  current_train = 25
  target_data = lorenz_ode(time_span=(0, current_train), ic=[-8., 7., 27.], t_eval=torch.linspace(0, current_train, int(40 * current_train + 1)), method='RK45')
  # ic = target_data.y[:, 1000]
  # print(target_data.t[1000])
  # ex = f'rb_pre_without_model_ex225'
  ad_T = torch.load('./datas/T_21.pt').detach().numpy()
  ad_Y = torch.load('./datas/ad_Y_21.pt').detach().numpy().T

  # 绘制绝对误差
  keep = 600
  start = 400
  error_t = ad_T[start:start+keep]
  error_x = np.abs(target_data.y[0, start:start+keep] - ad_Y[0, start:start+keep])
  error_y = np.abs(target_data.y[1, start:start+keep] - ad_Y[1, start:start+keep])
  error_z = np.abs(target_data.y[2, start:start+keep] - ad_Y[2, start:start+keep])

  color = ['purple', 'red']
  fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=False)
  ax1.plot(error_t, error_x, c=color[1], linestyle='-')

  ax1.set_xlabel('t')
  ax1.set_ylabel('x')
  # ax1.legend(['true', 'regression', 'predicion', 'adjust'], loc='upper right')

  ax2.plot(error_t, error_y, c=color[1], linestyle='-')

  ax2.set_xlabel('t')
  ax2.set_ylabel('y')

  ax3.plot(error_t, error_z, c=color[1], linestyle='-')

  ax3.set_xlabel('t')
  ax3.set_ylabel('z')
  # fig1.legend(['true', 'ODE45'], loc='upper center', ncol=2)
  fig2.savefig('./error_pic/ad_error.png')

  return error_t, error_x, error_y, error_z


def rc_abs_error():
    target_data = lorenz_ode(time_span=(0, 25), ic=[-8., 7., 27.], t_eval=torch.linspace(0, 25, 1001), method='RK45')
    acc_data = target_data.y
    T = torch.load('./datas/rc_predict_T').detach().numpy()
    predict_Y = torch.load('./datas/rc_predict_Y').detach().numpy()
    color = ['purple', 'red']
    start = 400
    keep = 600

    error_t = T[start:start+keep]
    error_x = np.abs(predict_Y[start:start+keep, 0] - acc_data[0, start:start+keep])
    error_y = np.abs(predict_Y[start:start+keep, 1] - acc_data[1, start:start+keep])
    error_z = np.abs(predict_Y[start:start+keep, 2] - acc_data[2, start:start+keep])

    fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=False)
    ax1.plot(error_t, error_x, c=color[1], linestyle='-')

    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    # ax1.legend(['true', 'regression', 'predicion', 'adjust'], loc='upper right')

    ax2.plot(error_t, error_y, c=color[1], linestyle='-')

    ax2.set_xlabel('t')
    ax2.set_ylabel('y')

    ax3.plot(error_t, error_z, c=color[1], linestyle='-')

    ax3.set_xlabel('t')
    ax3.set_ylabel('z')
    # fig1.legend(['true', 'ODE45'], loc='upper center', ncol=2)
    fig2.savefig('./error_pic/rc_error.png')
    
    return error_t, error_x, error_y, error_z


def compare(error_ad, error_rc, file_name, keep):
    
    # keep = 240

    if (keep == 240):
      xticks = (10, 12, 14, 16)
    else:
      xticks = (10, 12, 14, 16, 18, 20, 22, 24, 25)

    color = ['#385989', '#d22027']
    fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=False)
    ax1.plot(error_rc[0][0:keep], error_rc[1][0:keep], c=color[0], linestyle='-')
    ax1.plot(error_ad[0][0:keep], error_ad[1][0:keep], c=color[1], linestyle='-')
    ax1.set_xticks(xticks)

    # ax1.set_xlabel('t')
    # ax1.set_ylabel('x')
    # ax1.legend(['true', 'regression', 'predicion', 'adjust'], loc='upper right')

    ax2.plot(error_rc[0][0:keep], error_rc[2][0:keep], c=color[0], linestyle='-')
    ax2.plot(error_ad[0][0:keep], error_ad[2][0:keep], c=color[1], linestyle='-')
    ax2.set_xticks(xticks)

    # ax2.set_xlabel('t')
    # ax2.set_ylabel('y')

    ax3.plot(error_rc[0][0:keep], error_rc[3][0:keep], c=color[0], linestyle='-')
    ax3.plot(error_ad[0][0:keep], error_ad[3][0:keep], c=color[1], linestyle='-')
    ax3.set_xticks(xticks)

    # ax3.set_xlabel('t')
    # ax3.set_ylabel('z')
    # fig1.legend(['true', 'ODE45'], loc='upper center', ncol=2)
    # fig2.legend(['ode error', 'rc error'], loc='upper center', ncol=2)
    fig2.savefig(f'./error_pic/{file_name}')


def nrmse():

  target_data = lorenz_ode(time_span=(0, 25), ic=[-8., 7., 27.], t_eval=torch.linspace(0, 25, 1001), method='RK45')
  acc_data = target_data.y
  total_var = np.var(acc_data[0, :]) + np.var(acc_data[1, :]) + np.var(acc_data[2, :])
  print("total_var:", total_var)

  exs = [21]
  # for i in range(193, 204):
  #   exs.append(i)
    
  dt = 0.025
  lyaptime = 1.104
  lyaptime_pts = round(lyaptime/dt)

  for ex in exs:
    predict_Y = torch.load(f'./datas/predict_Y_{ex}.pt').T.detach().numpy()
    rc_nrmse = np.sqrt(np.mean((acc_data[:, 400:400+lyaptime_pts] - predict_Y[:, 400:400+lyaptime_pts])**2) / total_var)
    ad_Y = torch.load(f'./datas/ad_Y_{ex}.pt').T.detach().numpy()
    ad_nrmse = np.sqrt(np.mean((acc_data[:, 400:400+lyaptime_pts] - ad_Y[:, 400:400+lyaptime_pts])**2) / total_var)
    print(f"{ex} rc_nrmse:", rc_nrmse)
    print(f"{ex} ad_nrmse:", ad_nrmse)
    print(f"-----------------------------")
  

def DE_loss():
    target_data = lorenz_ode(time_span=(0, 25), ic=[-8., 7., 27.], t_eval=torch.linspace(0, 25, 1001), method='RK45')
    td_T = target_data.t
    td_Y = target_data.y
    td_dY = (td_Y[:, 2:] - td_Y[:, :-2]) / (td_T[2:] - td_T[:-2]).reshape(1, -1)
    err_td_T = td_T[1:-1]
    err_td = np.zeros(shape=td_dY.shape)
    err_td[0, :] = td_dY[0, :] - 10 * (td_Y[1, 1:-1] - td_Y[0, 1:-1])
    err_td[1, :] = td_dY[1, :] - td_Y[0, 1:-1] * (28 - td_Y[2, 1:-1]) + td_Y[1, 1:-1]
    err_td[2, :] = td_dY[2, :] - td_Y[0, 1:-1] * td_Y[1, 1:-1] + 8 / 3 * td_Y[2, 1:-1]

    # start = 400
    # keep = 600
    # color = ['purple', 'red']
    # fig1, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=False)
    # ax1.plot(err_td_T[start:start+keep], err_td[0, start:start+keep], c=color[0], linestyle='-')

    # ax2.plot(err_td_T[start:start+keep], err_td[1, start:start+keep], c=color[0], linestyle='-')

    # ax3.plot(err_td_T[start:start+keep], err_td[2, start:start+keep], c=color[0], linestyle='-')

    # fig1.savefig('./error_pic/test.png')

    ad_T = torch.load('./datas/T_21.pt').detach().numpy()
    ad_Y = torch.load('./datas/ad_Y_21.pt').detach().numpy().T
    ad_dY = (ad_Y[:, 2:] - ad_Y[:, :-2]) / (ad_T[2:] - ad_T[:-2]).reshape(1, -1)
    err_ad_T = ad_T[1:-1]
    err_ad = np.zeros(shape=ad_dY.shape)
    err_ad[0, :] = ad_dY[0, :] - 10 * (ad_Y[1, 1:-1] - ad_Y[0, 1:-1])
    err_ad[1, :] = ad_dY[1, :] - ad_Y[0, 1:-1] * (28 - ad_Y[2, 1:-1]) + ad_Y[1, 1:-1]
    err_ad[2, :] = ad_dY[2, :] - ad_Y[0, 1:-1] * ad_Y[1, 1:-1] + 8 / 3 * ad_Y[2, 1:-1]

    rc_T = torch.load('./datas/rc_predict_T').detach().numpy()
    rc_Y = torch.load('./datas/rc_predict_Y').detach().numpy().T
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

    
    # xticks = (10, 12, 14, 16, 18, 20, 22, 24, 25)
    xticks = (10, 12, 14, 16)

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
    fig1.savefig('./error_pic/ode_error_15.svg')


def plot_phase():
  current_train = 25
  target_data = lorenz_ode(time_span=(0, current_train), ic=[-8., 7., 27.], t_eval=torch.linspace(0, current_train, int(40 * current_train + 1)), method='RK45')
  td_T = target_data.t
  td_Y = target_data.y

  ad_T = torch.load('./datas/T_15.pt').detach().numpy()
  ad_Y = torch.load('./datas/ad_Y_15.pt').detach().numpy().T

  rc_T = torch.load('./datas/rc_predict_T').detach().numpy()
  rc_Y = torch.load('./datas/rc_predict_Y').detach().numpy().T

  start = 400
  color = ['#0000cc', '#008800', '#ff0000']

  # ground truth y-x
  fig1 = plt.figure(1)
  plt.plot(td_Y[0, start:], td_Y[1, start:])
  fig1.savefig(f'./error_pic/phase_td_yx.svg')
  fig1.savefig(f'./error_pic/phase_td_yx.png')

  # ground truth z-x
  fig2 = plt.figure(2)
  plt.plot(td_Y[0, start:], td_Y[2, start:])
  fig2.savefig(f'./error_pic/phase_td_zx.svg')
  fig2.savefig(f'./error_pic/phase_td_zx.png')

  # ground truth z-y
  fig3 = plt.figure(3)
  plt.plot(td_Y[1, start:], td_Y[2, start:])
  fig3.savefig(f'./error_pic/phase_td_zy.svg')
  fig3.savefig(f'./error_pic/phase_td_zy.png')

  # Reservoir Computing y-x
  fig4 = plt.figure(4)
  plt.plot(rc_Y[0, start:], rc_Y[1, start:])
  fig4.savefig(f'./error_pic/phase_rc_yx.svg')
  fig4.savefig(f'./error_pic/phase_rc_yx.png')

  # Reservoir Computing z-x
  fig5 = plt.figure(5)
  plt.plot(rc_Y[0, start:], rc_Y[2, start:])
  fig5.savefig(f'./error_pic/phase_rc_zx.svg')
  fig5.savefig(f'./error_pic/phase_rc_zx.png')

  # Reservoir Computing z-y
  fig6 = plt.figure(6)
  plt.plot(rc_Y[1, start:], rc_Y[2, start:])
  fig6.savefig(f'./error_pic/phase_rc_zy.svg')
  fig6.savefig(f'./error_pic/phase_rc_zy.png')

  # Adjusted y-x
  fig7 = plt.figure(7)
  plt.plot(ad_Y[0, start:], ad_Y[1, start:])
  fig7.savefig(f'./error_pic/phase_ad_yx.svg')
  fig7.savefig(f'./error_pic/phase_ad_yx.png')

  # Adjusted z-x
  fig8 = plt.figure(8)
  plt.plot(ad_Y[0, start:], ad_Y[2, start:])
  fig8.savefig(f'./error_pic/phase_ad_zx.svg')
  fig8.savefig(f'./error_pic/phase_ad_zx.png')

  # Adjusted z-y
  fig9 = plt.figure(9)
  plt.plot(ad_Y[1, start:], ad_Y[2, start:])
  fig9.savefig(f'./error_pic/phase_ad_zy.svg')
  fig9.savefig(f'./error_pic/phase_ad_zy.png')

  # compare Adjusted and ground truth y-x
  fig10 = plt.figure(10)
  plt.plot(td_Y[0, start:], td_Y[1, start:])
  plt.plot(ad_Y[0, start:], ad_Y[1, start:], color='red')
  fig10.savefig(f'./error_pic/phase_cmp_ad_td_yx.svg')
  fig10.savefig(f'./error_pic/phase_cmp_ad_td_yx.png')

  # compare Adjusted and ground truth z-x
  fig11 = plt.figure(11)
  plt.plot(td_Y[0, start:], td_Y[2, start:])
  plt.plot(ad_Y[0, start:], ad_Y[2, start:], color='red')
  fig11.savefig(f'./error_pic/phase_cmp_ad_td_zx.svg')
  fig11.savefig(f'./error_pic/phase_cmp_ad_td_zx.png')

  # compare Adjusted and ground truth z-y
  fig12 = plt.figure(12)
  plt.plot(td_Y[1, start:], td_Y[2, start:])
  plt.plot(ad_Y[1, start:], ad_Y[2, start:], color='red')
  fig12.savefig(f'./error_pic/phase_cmp_ad_td_zy.svg')
  fig12.savefig(f'./error_pic/phase_cmp_ad_td_zy.png')
  

def IDE_loss():
  target_data = lorenz_ode(time_span=(0, 25), ic=[-8., 7., 27.], t_eval=torch.linspace(0, 25, 1001), method='RK45')
  td_T = target_data.t
  td_Y = target_data.y
  td_dY = (td_Y[:, 2:] - td_Y[:, :-2]) / (td_T[2:] - td_T[:-2]).reshape(1, -1)
  err_td_T = td_T[1:-1]
  err_td = np.zeros(shape=td_dY.shape)
  err_td[0, :] = td_dY[0, :] - 10 * td_Y[1, 1:-1] + 10 * td_Y[0, 1:-1]
  err_td[1, :] = td_dY[1, :] - 28 * td_Y[0, 1:-1] + 1 * td_Y[0, 1:-1] * td_Y[2, 1:-1] + 1 * td_Y[1, 1:-1]
  err_td[2, :] = td_dY[2, :] - 1 * td_Y[0, 1:-1] * td_Y[1, 1:-1] + 8 / 3 * td_Y[2, 1:-1]
  # err_td[0, :] = td_dY[0, :] - 10.0003 * td_Y[1, 1:-1] + 10 * td_Y[0, 1:-1]
  # err_td[1, :] = td_dY[1, :] - 28 * td_Y[0, 1:-1] + 1 * td_Y[0, 1:-1] * td_Y[2, 1:-1] + 1.0008 * td_Y[1, 1:-1]
  # err_td[2, :] = td_dY[2, :] - 0.9999 * td_Y[0, 1:-1] * td_Y[1, 1:-1] + 2.6663 * td_Y[2, 1:-1]

  ad_T = torch.load('./datas/T_15.pt').detach().numpy()
  ad_Y = torch.load('./datas/ad_Y_15.pt').detach().numpy().T
  ad_dY = (ad_Y[:, 2:] - ad_Y[:, :-2]) / (ad_T[2:] - ad_T[:-2]).reshape(1, -1)
  err_ad_T = ad_T[1:-1]
  err_ad = np.zeros(shape=ad_dY.shape)
  err_ad[0, :] = ad_dY[0, :] - 10.0003 * ad_Y[1, 1:-1] + 10 * ad_Y[0, 1:-1]
  err_ad[1, :] = ad_dY[1, :] - 28 * ad_Y[0, 1:-1] + 1 * ad_Y[0, 1:-1] * ad_Y[2, 1:-1] + 1.0008 * ad_Y[1, 1:-1]
  err_ad[2, :] = ad_dY[2, :] - 0.9999 * ad_Y[0, 1:-1] * ad_Y[1, 1:-1] + 2.6663 * ad_Y[2, 1:-1]

  # rc_T = torch.load('./datas/rc_predict_T').detach().numpy()
  # rc_Y = torch.load('./datas/rc_predict_Y').detach().numpy().T
  # rc_dY = (rc_Y[:, 2:] - rc_Y[:, :-2]) / (rc_T[2:] - rc_T[:-2]).reshape(1, -1)
  # err_rc_T = rc_T[1:-1]
  # err_rc = np.zeros(shape=rc_dY.shape)
  # err_rc[0, :] = rc_dY[0, :] - 10 * (rc_Y[1, 1:-1] - rc_Y[0, 1:-1])
  # err_rc[1, :] = rc_dY[1, :] - rc_Y[0, 1:-1] * (28 - rc_Y[2, 1:-1]) + rc_Y[1, 1:-1]
  # err_rc[2, :] = rc_dY[2, :] - rc_Y[0, 1:-1] * rc_Y[1, 1:-1] + 8 / 3 * rc_Y[2, 1:-1]


  # err_td_x = err_td[0, :]
  # err_td_y = err_td[1, :]
  # err_td_z = err_td[2, :]
  # err_ad_x = np.abs(err_ad[0, :] - err_td_x)
  # err_ad_y = np.abs(err_ad[1, :] - err_td_y)
  # err_ad_z = np.abs(err_ad[2, :] - err_td_z)
  # err_rc_x = np.abs(err_rc[0, :] - err_td_x)
  # err_rc_y = np.abs(err_rc[1, :] - err_td_y)
  # err_rc_z = np.abs(err_rc[2, :] - err_td_z)

  
  # xticks = (10, 12, 14, 16)

  start = 400
  keep = 600
  color = ['#385989', '#d22027']
  fig1, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=False)
  ax1.plot(err_td_T[start:start+keep], err_td[0, start:start+keep], c=color[0], linestyle='-')
  ax1.plot(err_ad_T[start:start+keep], err_ad[0, start:start+keep], c=color[1], linestyle='--')
  # ax1.set_xticks(xticks)

  ax2.plot(err_td_T[start:start+keep], err_td[1, start:start+keep], c=color[0], linestyle='-')
  ax2.plot(err_ad_T[start:start+keep], err_ad[1, start:start+keep], c=color[1], linestyle='--')
  # ax2.set_xticks(xticks)

  ax3.plot(err_td_T[start:start+keep], err_td[1, start:start+keep], c=color[0], linestyle='-')
  ax3.plot(err_ad_T[start:start+keep], err_ad[1, start:start+keep], c=color[1], linestyle='--')
  # ax3.set_xticks(xticks)

  fig1.savefig('./error_pic/IDE.svg')
  fig1.savefig('./error_pic/IDE.png')


def asd_compare():
  current_train = 25
  target_data = lorenz_ode(time_span=(0, current_train), ic=[-8., 7., 27.], t_eval=torch.linspace(0, current_train, int(20 * current_train + 1)), method='RK45')
  td_T = target_data.t
  td_Y = target_data.y

  ad_T = torch.load('./datas/T_112.pt').detach().numpy()
  ad_Y = torch.load('./datas/ad_Y_112.pt').detach().numpy().T

  b_T = torch.load('./datas/T_112.pt').detach().numpy()
  b_Y = torch.load('./datas/before_ad_Y_112.pt').detach().numpy().T


  err_ad = np.sum((ad_Y - td_Y) ** 2, axis=0)
  err_b = np.sum((b_Y - td_Y) ** 2, axis=0)
  print(err_ad)

  start = 201
  end = 261
  color = ['#0000cc', '#008800', '#ff0000']
  fig2, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True)
  ax1.plot(td_T[start:end], err_ad[start:end], c=color[0], linestyle='-')

  start = 201
  end = 221
  ax1.plot(td_T[start:end], err_b[start:end], c=color[1], linestyle='--')
  # ax3.set_xlim((10,18))
  
  start = 221
  end = 240
  ax1.plot(td_T[start:end], err_b[start:end], c=color[1], linestyle='--')

  start = 241
  end = 261
  ax1.plot(td_T[start:end], err_b[start:end], c=color[1], linestyle='--')

  fig2.savefig(f'./error_pic/123.svg', dpi=500)
  fig2.savefig(f'./error_pic/123.png', dpi=500)

  # start = 480
  # end = 520
  # color = ['#0000cc', '#008800', '#ff0000']
  # fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
  # ax1.plot(td_T[start:end], err_ad[0, start:end], c=color[0], linestyle='-')
  # ax1.plot(td_T[start:end], err_b[0, start:end], c=color[1], linestyle='--')
  # # ax1.set_xlim((10,18))

  # ax2.plot(td_T[start:end], err_ad[1, start:end], c=color[0], linestyle='-')
  # ax2.plot(td_T[start:end], err_b[1, start:end], c=color[1], linestyle='--')
  # # ax2.set_xlim((10,18))

  # ax3.plot(td_T[start:end], err_ad[2, start:end], c=color[0], linestyle='-')
  # ax3.plot(td_T[start:end], err_b[2, start:end], c=color[1], linestyle='--')
  # # ax3.set_xlim((10,18))

  # fig2.savefig(f'./error_pic/123.svg')
  # fig2.savefig(f'./error_pic/123.png')


def plot_real_rc_ad(ad_ex):
  current_train = 25
  target_data = lorenz_ode(time_span=(0, current_train), ic=[-8., 7., 27.], t_eval=torch.linspace(0, current_train, int(20 * current_train + 1)), method='RK45')
  td_T = target_data.t
  td_Y = target_data.y

  ad_T = torch.load(f'./datas/T_{ad_ex}.pt').detach().numpy()
  ad_Y = torch.load(f'./datas/ad_Y_{ad_ex}.pt').detach().numpy().T

  # rc_T = torch.load(f'./datas/T_112.pt').detach().numpy()
  rc_Y = torch.load(f'./datas/predict_Y_{ad_ex}.pt').detach().numpy().T


  start = 200
  end = 500
  color = ['#0000cc', '#008800', '#ff0000']
  fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
  ax1.plot(td_T[start:end], td_Y[0, start:end], c=color[0], linestyle='-')
  ax1.plot(td_T[start:end], rc_Y[0, start:end], c=color[1], linestyle='-')
  ax1.plot(td_T[start:end], ad_Y[0, start:end], c=color[2], linestyle='-')

  ax2.plot(td_T[start:end], td_Y[1, start:end], c=color[0], linestyle='-')
  ax2.plot(td_T[start:end], rc_Y[1, start:end], c=color[1], linestyle='-')
  ax2.plot(td_T[start:end], ad_Y[1, start:end], c=color[2], linestyle='-')
  
  ax3.plot(td_T[start:end], td_Y[2, start:end], c=color[0], linestyle='-')
  ax3.plot(td_T[start:end], rc_Y[2, start:end], c=color[1], linestyle='-')
  ax3.plot(td_T[start:end], ad_Y[2, start:end], c=color[2], linestyle='-')


  fig2.savefig(f'./error_pic/res.svg', dpi=500)
  fig2.savefig(f'./error_pic/res.png', dpi=500)


def compare_real_bf_rc_ad(ad_ex):
  current_train = 25
  target_data = lorenz_ode(time_span=(0, current_train), ic=[-8., 7., 27.], t_eval=torch.linspace(0, current_train, int(20 * current_train + 1)), method='RK45')
  td_T = target_data.t
  td_Y = target_data.y
  
  f = 1000
  T = np.linspace(0, 25, 25 * f + 1)
  # 使用三次样条插值
  spline = interp1d(td_T, td_Y, kind='cubic', axis=1)
  # 创建插值函数
  td_Y = spline(T)

  td_dY = np.zeros_like(td_Y)
  # td_dY[:, 1:-1] = (td_Y[:, 2:] - td_Y[:, 0:-2]) / (T[2:] - T[:-2])
  td_dY[:, 2:-2] = (- td_Y[:, 4:] + 8 * td_Y[:, 3:-1] - 8 * td_Y[:, 1:-3] + td_Y[:, :-4]) / (12 / f)

  err_td = np.zeros(shape=td_dY.shape)
  err_td[0, :] = td_dY[0, :] - 10 * (td_Y[1, :] - td_Y[0, :])
  err_td[1, :] = td_dY[1, :] - td_Y[0, :] * (28 - td_Y[2, :]) + td_Y[1, :]
  err_td[2, :] = td_dY[2, :] - td_Y[0, :] * td_Y[1, :] + 8 / 3 * td_Y[2, :]

  # T = torch.load(f'./datas/T_{ad_ex}.pt').detach().numpy()
  # ad_Y = torch.load(f'./datas/ad_Y_{ad_ex}.pt').detach().numpy().T
  # b_Y = torch.load(f'./datas/before_ad_Y_{ad_ex}.pt').detach().numpy().T
  # rc_Y = torch.load(f'./datas/predict_Y_{ad_ex}.pt').detach().numpy().T

  # err_ad = np.abs(ad_Y - td_Y)
  # err_b = np.abs(b_Y - td_Y)

  start = 10 * f
  end = 13 * f
  color = ['#0000cc', '#008800', '#ff0000']
  fig1, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
  ax1.plot(T[start:end], err_td[0, start:end], c=color[0], linestyle='-')

  ax2.plot(T[start:end], err_td[1, start:end], c=color[0], linestyle='-')

  ax3.plot(T[start:end], err_td[2, start:end], c=color[0], linestyle='-')

  fig1.savefig(f'./error_pic/123.svg', dpi=500)
  fig1.savefig(f'./error_pic/123.png', dpi=500)

  # start = 200
  # end = 221
  # color = ['#0000cc', '#008800', '#ff0000']
  # fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
  # ax1.plot(T[start:end], err_ad[0, start:end], c=color[0], linestyle='-')
  # ax1.plot(T[start:end], err_b[0, start:end], c=color[1], linestyle='-')

  # ax2.plot(T[start:end], err_ad[1, start:end], c=color[0], linestyle='-')
  # ax2.plot(T[start:end], err_b[1, start:end], c=color[1], linestyle='-')

  # ax3.plot(T[start:end], err_ad[2, start:end], c=color[0], linestyle='-')
  # ax3.plot(T[start:end], err_b[2, start:end], c=color[1], linestyle='-')

  # start = 221
  # end = 241
  # ax1.plot(T[start:end], err_ad[0, start:end], c=color[0], linestyle='-')
  # ax1.plot(T[start:end], err_b[0, start:end], c=color[1], linestyle='-')

  # ax2.plot(T[start:end], err_ad[1, start:end], c=color[0], linestyle='-')
  # ax2.plot(T[start:end], err_b[1, start:end], c=color[1], linestyle='-')

  # ax3.plot(T[start:end], err_ad[2, start:end], c=color[0], linestyle='-')
  # ax3.plot(T[start:end], err_b[2, start:end], c=color[1], linestyle='-')

  # start = 241
  # end = 261
  # ax1.plot(T[start:end], err_ad[0, start:end], c=color[0], linestyle='-')
  # ax1.plot(T[start:end], err_b[0, start:end], c=color[1], linestyle='-')

  # ax2.plot(T[start:end], err_ad[1, start:end], c=color[0], linestyle='-')
  # ax2.plot(T[start:end], err_b[1, start:end], c=color[1], linestyle='-')

  # ax3.plot(T[start:end], err_ad[2, start:end], c=color[0], linestyle='-')
  # ax3.plot(T[start:end], err_b[2, start:end], c=color[1], linestyle='-')

  # fig2.savefig(f'./error_pic/real_bf_rc_ad.svg', dpi=500)
  # fig2.savefig(f'./error_pic/real_bf_rc_ad.png', dpi=500)

  # err_ad = np.sum((ad_Y - td_Y) ** 2, axis=0)
  # err_b = np.sum((b_Y - td_Y) ** 2, axis=0)
  # print(err_ad)
  # pass


def plot_ad_cmp():
  td_T, td_Y = read_data('./datas/lorenz_sparse.csv')

  ad_T = torch.load('./datas/T_144.pt').detach().numpy()
  ad_Y = torch.load('./datas/ad_Y_144.pt').detach().numpy().T

  start = 200
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
  td_T, td_Y = read_data('./datas/lorenz_sparse.csv')

  oriRC_T = torch.load('./datas/oriRC_predict_T').detach().numpy()
  oriRC_Y = torch.load('./datas/oriRC_predict_Y').detach().numpy().T

  start = 200
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
  td_T, td_Y = read_data('./datas/lorenz_sparse.csv')

  id_T = torch.load('./datas/id_T.pt').detach().numpy()
  id_Y = torch.load('./datas/id_Y.pt').detach().numpy()

  start = 200
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
  td_T, td_Y = read_data('./datas/lorenz_sparse.csv')

  ex = 1
  pinn_T = torch.load(f'./datas/T_PINN{ex}.pt').detach().numpy()
  pinn_Y = torch.load(f'./datas/Y_PINN{ex}.pt').detach().numpy().T

  start = 200
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
  # error_ad = ad_abs_error()
  # error_rc = rc_abs_error()
  # compare(error_ad, error_rc, 'abs_error_compare_6.svg', keep=240)
  # nrmse()
  # DE_loss()
  # plot_phase()
  # ksi = torch.load('ksis/ksi_rb_pre_without_model_ex225.pt')
  # print(ksi)
  # IDE_loss()
  # res_compare()

  # plot_real_rc_ad(ad_ex=144)

  # compare_real_bf_rc_ad(ad_ex=144)

  # 绘制原始数据和各预测方法结果的对比
  plot_ad_cmp()
  plot_rc_cmp()
  plot_id_cmp()
  plot_pinn_cmp()