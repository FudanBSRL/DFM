import torch
import numpy as np
from dp_ode import dp_ode
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from data_util import read_data_ex


def res_cmp(ex):
    d_T, td_Y = read_data(f'./csvs/d_pendulum.csv')
      
    id_T = torch.load(f'./datas/id_T.pt').detach().numpy()
    id_Y = torch.load(f'./datas/id_Y.pt').detach().numpy()

    ad_T = torch.load(f'./datas/T_{ex}.pt').detach().numpy()
    ad_Y = torch.load(f'./datas/ad_Y_{ex}.pt').detach().numpy().T

    rc_t = torch.load(f'./datas/oriRC_predict_T').detach().numpy()
    rc_Y = torch.load(f'./datas/oriRC_predict_Y').detach().numpy().T

    
    colors = ['blue', 'green', 'orange', 'red']
    linestyles = ['-', '-.', ':', '--']
    fig2, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True)
    axis = 0
    ax1.plot(d_T, td_Y[axis, :], c=colors[0], linestyle=linestyles[0])
    ax1.plot(id_T, id_Y[axis, :], c=colors[1], linestyle=linestyles[1])
    ax1.plot(rc_t, rc_Y[axis, :], c=colors[2], linestyle=linestyles[2])
    ax1.plot(ad_T, ad_Y[axis, :], c=colors[3], linestyle=linestyles[3])

    axis = 1
    ax2.plot(d_T, td_Y[axis, :], c=colors[0], linestyle=linestyles[0])
    ax2.plot(id_T, id_Y[axis, :], c=colors[1], linestyle=linestyles[1])
    ax2.plot(rc_t, rc_Y[axis, :], c=colors[2], linestyle=linestyles[2])
    ax2.plot(ad_T, ad_Y[axis, :], c=colors[3], linestyle=linestyles[3])

    axis = 2
    ax3.plot(d_T, td_Y[axis, :], c=colors[0], linestyle=linestyles[0])
    ax3.plot(id_T, id_Y[axis, :], c=colors[1], linestyle=linestyles[1])
    ax3.plot(rc_t, rc_Y[axis, :], c=colors[2], linestyle=linestyles[2])
    ax3.plot(ad_T, ad_Y[axis, :], c=colors[3], linestyle=linestyles[3])
    
    axis = 3
    ax4.plot(d_T, td_Y[axis, :], c=colors[0], linestyle=linestyles[0])
    ax4.plot(id_T, id_Y[axis, :], c=colors[1], linestyle=linestyles[1])
    ax4.plot(rc_t, rc_Y[axis, :], c=colors[2], linestyle=linestyles[2])
    ax4.plot(ad_T, ad_Y[axis, :], c=colors[3], linestyle=linestyles[3])

    fig2.legend(['true', 'ODE45', 'RC', 'MMPINN'], loc='upper center', ncol=4)
    fig2.align_labels()
    fig2.savefig(f'error_pic/res_cmp.png', dpi=500)
    fig2.savefig(f'error_pic/res_cmp.svg', dpi=500)


def error_cmp(ex):
    td_T, td_Y = read_data_ex(f'./csvs/ex_data1.CSV')
    td_Y[0:2] = np.sin(td_Y[0:2])
    
    id_T = torch.load(f'./datas/id_T.pt').detach().numpy()
    id_Y = torch.load(f'./datas/id_Y.pt').detach().numpy()
    id_Y[0:2] = np.sin(id_Y[0:2])

    ad_T = torch.load(f'./datas/T_{ex}.pt').detach().numpy()
    ad_Y = torch.load(f'./datas/ad_Y_{ex}.pt').detach().numpy().T
    ad_Y[0:2] = np.sin(ad_Y[0:2])

    rc_t = torch.load(f'./datas/oriRC_predict_T').detach().numpy()
    rc_Y = torch.load(f'./datas/oriRC_predict_Y').detach().numpy().T
    rc_Y[0:2] = np.sin(rc_Y[0:2])
    
    error_id = np.abs(td_Y[:, 0:len(id_Y[0])] - id_Y)
    error_ad = np.abs(td_Y[:, 0:len(ad_Y[0])] - ad_Y)
    error_rc = np.abs(td_Y[:, 0:len(rc_Y[0])] - rc_Y)

    # print(error_id.shape)

    xlim = (20, 30)

    fig1, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True)
    ax1.plot(id_T[0:len(error_id[0])], error_id[0, :])
    ax1.plot(id_T[0:len(error_ad[0])], error_ad[0, :])
    ax1.plot(id_T[0:len(error_rc[0])], error_rc[0, :])
    ax1.set_xlim(xlim)
    ax1.set_ylabel(r'error $\theta_1$')

    ax2.plot(id_T[0:len(error_id[0])], error_id[1, :])
    ax2.plot(id_T[0:len(error_ad[0])], error_ad[1, :])
    ax2.plot(id_T[0:len(error_rc[0])], error_rc[1, :])
    ax2.set_xlim(xlim)
    ax2.set_ylabel(r'error $\theta_2$')
    
    ax3.plot(id_T[0:len(error_id[0])], error_id[2, :])
    ax3.plot(id_T[0:len(error_ad[0])], error_ad[2, :])
    ax3.plot(id_T[0:len(error_rc[0])], error_rc[2, :])
    ax3.set_xlim(xlim)
    ax3.set_ylabel(r'error $\omega_1$')
    
    ax4.plot(id_T[0:len(error_id[0])], error_id[3, :])
    ax4.plot(id_T[0:len(error_ad[0])], error_ad[3, :])
    ax4.plot(id_T[0:len(error_rc[0])], error_rc[3, :])
    ax4.set_xlim(xlim)
    ax4.set_ylabel(r'error $\omega_2$')
    ax4.set_xlabel(r'$t(s)$')

    fig1.legend(["ODE45", "DFM", "RC"], loc='upper center', ncol=3)
    fig1.savefig(f'./error_pic/error_cmp_{ex}.png', dpi=500)

    
    l2_id = np.sqrt(np.sum((td_Y[:, 0:len(id_Y[0])] - id_Y) ** 2, axis=0))
    l2_ad = np.sqrt(np.sum((td_Y[:, 0:len(ad_Y[0])] - ad_Y) ** 2, axis=0))
    l2_rc = np.sqrt(np.sum((td_Y[:, 0:len(rc_Y[0])] - rc_Y) ** 2, axis=0))
    fig2, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=False)
    ax1.plot(id_T[0:len(l2_id)], l2_id, c='blue')
    ax1.plot(id_T[0:len(l2_rc)], l2_rc, c='green')
    ax1.plot(id_T[0:len(l2_ad)], l2_ad, c='red')
    ax1.set_xlim(xlim)
    ax1.set_xlabel(r'$t(s)$')
    ax1.set_ylabel('MSE')
    fig2.legend(['ODE45', 'RC', 'DFM'], loc='upper center', ncol=3)
    fig2.align_labels()
    fig2.savefig(f'./error_pic/l2_cmp_{ex}.png', dpi=500)


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


def cmp_nrmse():
    td_T, td_Y = read_data_ex(f'./csvs/d_pendulum_f2.csv')

    alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
    nrmses = []

    rc_Y = torch.load(f'./datas/oriRC_predict_Y').detach().numpy().T
    nrmses.append(nrmse_std(td_Y[:, 801:], rc_Y[:, 801:]))

    exs = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    for ex in exs:
      ad_Y = torch.load(f'./datas/ad_Y_{ex}.pt').detach().numpy().T
      nrmses.append(nrmse_std(td_Y[:, 801:], ad_Y[:, 801:]))

    id_Y = torch.load(f'./datas/id_Y.pt').detach().numpy()
    nrmses.append(nrmse_std(td_Y[:, 801:], id_Y[:, 801:]))

    fig = plt.figure()
    plt.plot(alphas, nrmses, color='red')
    plt.scatter(alphas, nrmses, color='red', marker='^')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('NRMSE')
    fig.savefig(f'./img/cmp_nrmse.png', dpi=500)
    fig.savefig(f'./img/cmp_nrmse.svg', dpi=500)


def plot_cmp(ex):
    ex = 25010010
    td_T, td_Y, _, _ = read_data_ex(f'./csvs/ex_data1.CSV', rate=40)
    
    id_T = torch.load(f'./datas/id_T.pt').detach().numpy()
    id_Y = torch.load(f'./datas/id_Y.pt').detach().numpy()

    ad_T = torch.load(f'./datas/T_{ex}.pt').detach().numpy()
    ad_Y = torch.load(f'./datas/ad_Y_{ex}.pt').detach().numpy().T

    pinn_T = torch.load(f'./datas/T_{25021000}.pt').detach().numpy()
    pinn_Y = torch.load(f'./datas/ad_Y_{25021000}.pt').detach().numpy().T

    rc_t = torch.load(f'./datas/oriRC_predict_T').detach().numpy()
    rc_Y = torch.load(f'./datas/oriRC_predict_Y').detach().numpy().T

    fig1, axs = plt.subplots(nrows=4, ncols=2, sharex=True, figsize=(30, 10))

    xlim = (21, 26)

    colors = ['blue', 'red', 'green', 'black']
    axs[0, 0].plot(td_T, np.sin(td_Y[0, :]), c=colors[0])
    axs[0, 0].plot(id_T[0: len(id_Y[0])], np.sin(id_Y[0, :]), c=colors[1])
    axs[0, 0].set_xlim(xlim)
    axs[0, 0].set_ylabel("ODE45")
    axs[0, 1].plot(td_T, np.sin(td_Y[1, :]), c=colors[0])
    axs[0, 1].plot(id_T[0: len(id_Y[0])], np.sin(id_Y[1, :]), c=colors[1])
    axs[0, 1].set_xlim(xlim)
    # axs[0, 2].plot(td_T, td_Y[2, :], c=colors[0])
    # axs[0, 2].plot(id_T[0: len(id_Y[0])], id_Y[2, :], c=colors[1])
    # axs[0, 2].set_xlim(xlim)
    # axs[0, 2].set_ylim(-12, 12)
    # axs[0, 3].plot(td_T, td_Y[3, :], c=colors[0])
    # axs[0, 3].plot(id_T[0: len(id_Y[0])], id_Y[3, :], c=colors[1])
    # axs[0, 3].set_xlim(xlim)
    # axs[0, 3].set_ylim(-30, 30)

    axs[1, 0].plot(td_T, np.sin(td_Y[0, :]), c=colors[0])
    axs[1, 0].plot(pinn_T, np.sin(pinn_Y[0, :]), c=colors[1])
    axs[1, 0].set_xlim(xlim)
    axs[1, 0].set_ylabel("PINN")
    axs[1, 1].plot(td_T, np.sin(td_Y[1, :]), c=colors[0])
    axs[1, 1].plot(pinn_T, np.sin(pinn_Y[1, :]), c=colors[1])
    axs[1, 1].set_xlim(xlim)
    # axs[1, 2].plot(td_T, td_Y[2, :], c=colors[0])
    # axs[1, 2].plot(pinn_T, pinn_Y[2, :], c=colors[1])
    # axs[1, 2].set_xlim(xlim)
    # axs[1, 2].set_ylim(-12, 12)
    # axs[1, 3].plot(td_T, td_Y[3, :], c=colors[0])
    # axs[1, 3].plot(pinn_T, pinn_Y[3, :], c=colors[1])
    # axs[1, 3].set_xlim(xlim)
    # axs[1, 3].set_ylim(-30, 30)

    axs[2, 0].plot(td_T, np.sin(td_Y[0, :]), c=colors[0])
    axs[2, 0].plot(rc_t, np.sin(rc_Y[0, :]), c=colors[1])
    axs[2, 0].set_xlim(xlim)
    axs[2, 0].set_ylabel("NG-RC")
    axs[2, 1].plot(td_T, np.sin(td_Y[1, :]), c=colors[0])
    axs[2, 1].plot(rc_t, np.sin(rc_Y[1, :]), c=colors[1])
    axs[2, 1].set_xlim(xlim)
    # axs[2, 2].plot(td_T, td_Y[2, :], c=colors[0])
    # axs[2, 2].plot(rc_t, rc_Y[2, :], c=colors[1])
    # axs[2, 2].set_xlim(xlim)
    # axs[2, 2].set_ylim(-12, 12)
    # axs[2, 3].plot(td_T, td_Y[3, :], c=colors[0])
    # axs[2, 3].plot(rc_t, rc_Y[3, :], c=colors[1])
    # axs[2, 3].set_xlim(xlim)
    # axs[2, 3].set_ylim(-30, 30)

    axs[3, 0].plot(td_T, np.sin(td_Y[0, :]), c=colors[0])
    axs[3, 0].plot(ad_T, np.sin(ad_Y[0, :]), c=colors[1])
    axs[3, 0].set_xlim(xlim)
    axs[3, 0].set_ylabel("DF")
    axs[3, 1].plot(td_T, np.sin(td_Y[1, :]), c=colors[0])
    axs[3, 1].plot(ad_T, np.sin(ad_Y[1, :]), c=colors[1])
    axs[3, 1].set_xlim(xlim)
    # axs[3, 2].plot(td_T, td_Y[2, :], c=colors[0])
    # axs[3, 2].plot(ad_T, ad_Y[2, :], c=colors[1])
    # axs[3, 2].set_xlim(xlim)
    # axs[3, 2].set_ylim(-12, 12)
    # axs[3, 3].plot(td_T, td_Y[3, :], c=colors[0])
    # axs[3, 3].plot(ad_T, ad_Y[3, :], c=colors[1])
    # axs[3, 3].set_xlim(xlim)
    # axs[3, 3].set_ylim(-30, 30)

    # fig1.legend(["true", "id", "ad", "rc"])
    fig1.savefig(f'./error_pic/cmp_{ex}_sin.png', dpi=800)
    fig1.savefig(f'./error_pic/cmp_{ex}_sin.svg', dpi=800)

    fig2, axs = plt.subplots(nrows=4, ncols=2, sharex=True, figsize=(30, 10))

    xlim = (21, 26)

    colors = ['blue', 'red', 'green', 'black']
    axs[0, 0].plot(td_T, td_Y[0, :], c=colors[0])
    axs[0, 0].plot(id_T[0: len(id_Y[0])], id_Y[0, :], c=colors[1])
    axs[0, 0].set_xlim(xlim)
    axs[0, 0].set_ylabel("ODE45")
    axs[0, 1].plot(td_T, td_Y[1, :], c=colors[0])
    axs[0, 1].plot(id_T[0: len(id_Y[0])], id_Y[1, :], c=colors[1])
    axs[0, 1].set_xlim(xlim)
    axs[0, 1].set_ylim(-45, -20)
    # axs[0, 2].plot(td_T, td_Y[2, :], c=colors[0])
    # axs[0, 2].plot(id_T[0: len(id_Y[0])], id_Y[2, :], c=colors[1])
    # axs[0, 2].set_xlim(xlim)
    # axs[0, 2].set_ylim(-12, 12)
    # axs[0, 3].plot(td_T, td_Y[3, :], c=colors[0])
    # axs[0, 3].plot(id_T[0: len(id_Y[0])], id_Y[3, :], c=colors[1])
    # axs[0, 3].set_xlim(xlim)
    # axs[0, 3].set_ylim(-30, 30)

    axs[1, 0].plot(td_T, td_Y[0, :], c=colors[0])
    axs[1, 0].plot(pinn_T, pinn_Y[0, :], c=colors[1])
    axs[1, 0].set_xlim(xlim)
    axs[1, 0].set_ylabel("PINN")
    axs[1, 1].plot(td_T, td_Y[1, :], c=colors[0])
    axs[1, 1].plot(pinn_T, pinn_Y[1, :], c=colors[1])
    axs[1, 1].set_xlim(xlim)
    axs[1, 1].set_ylim(-45, -20)
    # axs[1, 2].plot(td_T, td_Y[2, :], c=colors[0])
    # axs[1, 2].plot(pinn_T, pinn_Y[2, :], c=colors[1])
    # axs[1, 2].set_xlim(xlim)
    # axs[1, 2].set_ylim(-12, 12)
    # axs[1, 3].plot(td_T, td_Y[3, :], c=colors[0])
    # axs[1, 3].plot(pinn_T, pinn_Y[3, :], c=colors[1])
    # axs[1, 3].set_xlim(xlim)
    # axs[1, 3].set_ylim(-30, 30)

    axs[2, 0].plot(td_T, td_Y[0, :], c=colors[0])
    axs[2, 0].plot(rc_t, rc_Y[0, :], c=colors[1])
    axs[2, 0].set_xlim(xlim)
    axs[2, 0].set_ylabel("NG-RC")
    axs[2, 1].plot(td_T, td_Y[1, :], c=colors[0])
    axs[2, 1].plot(rc_t, rc_Y[1, :], c=colors[1])
    axs[2, 1].set_xlim(xlim)
    axs[2, 1].set_ylim(-45, -20)
    # axs[2, 2].plot(td_T, td_Y[2, :], c=colors[0])
    # axs[2, 2].plot(rc_t, rc_Y[2, :], c=colors[1])
    # axs[2, 2].set_xlim(xlim)
    # axs[2, 2].set_ylim(-12, 12)
    # axs[2, 3].plot(td_T, td_Y[3, :], c=colors[0])
    # axs[2, 3].plot(rc_t, rc_Y[3, :], c=colors[1])
    # axs[2, 3].set_xlim(xlim)
    # axs[2, 3].set_ylim(-30, 30)

    axs[3, 0].plot(td_T, td_Y[0, :], c=colors[0])
    axs[3, 0].plot(ad_T, ad_Y[0, :], c=colors[1])
    axs[3, 0].set_xlim(xlim)
    axs[3, 0].set_ylabel("DF")
    axs[3, 1].plot(td_T, td_Y[1, :], c=colors[0])
    axs[3, 1].plot(ad_T, ad_Y[1, :], c=colors[1])
    axs[3, 1].set_xlim(xlim)
    axs[3, 1].set_ylim(-45, -20)
    # axs[3, 2].plot(td_T, td_Y[2, :], c=colors[0])
    # axs[3, 2].plot(ad_T, ad_Y[2, :], c=colors[1])
    # axs[3, 2].set_xlim(xlim)
    # axs[3, 2].set_ylim(-12, 12)
    # axs[3, 3].plot(td_T, td_Y[3, :], c=colors[0])
    # axs[3, 3].plot(ad_T, ad_Y[3, :], c=colors[1])
    # axs[3, 3].set_xlim(xlim)
    # axs[3, 3].set_ylim(-30, 30)

    # fig1.legend(["true", "id", "ad", "rc"])
    fig2.savefig(f'./error_pic/cmp_{ex}.png', dpi=800)
    fig2.savefig(f'./error_pic/cmp_{ex}.svg', dpi=800)


def plot_2d(ex):
    ex = 25010010
    dt = 0.01
    start = 2100
    end = 2401
    
    td_T, td_Y, l1, l2 = read_data_ex(f'./csvs/ex_data1.CSV', rate=round(1/dt), filter=False)
    td_Y = td_Y[:, start: end]
    td_x1 = l1 * np.sin(td_Y[0, :])
    td_y1 = - l1 * np.cos(td_Y[0, :])
    td_x2 = td_x1 + l2 * np.sin(td_Y[1, :])
    td_y2 = td_y1 - l2 * np.cos(td_Y[1, :])

    id_T = torch.load(f'./datas/id_T.pt').detach().numpy()
    id_Y = torch.load(f'./datas/id_Y.pt').detach().numpy()
    id_Y = id_Y[:, start: end]
    id_x1 = l1 * np.sin(id_Y[0, :])
    id_y1 = - l1 * np.cos(id_Y[0, :])
    id_x2 = id_x1 + l2 * np.sin(id_Y[1, :])
    id_y2 = id_y1 - l2 * np.cos(id_Y[1, :])

    # print(id_T.shape)

    ad_T = torch.load(f'./datas/T_{ex}.pt').detach().numpy()
    ad_Y = torch.load(f'./datas/ad_Y_{ex}.pt').detach().numpy().T
    ad_Y = ad_Y[:, start: end]
    ad_x1 = l1 * np.sin(ad_Y[0, :])
    ad_y1 = - l1 * np.cos(ad_Y[0, :])
    ad_x2 = ad_x1 + l2 * np.sin(ad_Y[1, :])
    ad_y2 = ad_y1 - l2 * np.cos(ad_Y[1, :])

    # print(ad_x1.shape)

    pinn_T = torch.load(f'./datas/T_{25021000}.pt').detach().numpy()
    pinn_Y = torch.load(f'./datas/ad_Y_{25021000}.pt').detach().numpy().T
    pinn_Y = pinn_Y[:, start: end]
    pinn_x1 = l1 * np.sin(pinn_Y[0, :])
    pinn_y1 = - l1 * np.cos(pinn_Y[0, :])
    pinn_x2 = pinn_x1 + l2 * np.sin(pinn_Y[1, :])
    pinn_y2 = pinn_y1 - l2 * np.cos(pinn_Y[1, :])

    rc_t = torch.load(f'./datas/oriRC_predict_T').detach().numpy()
    rc_Y = torch.load(f'./datas/oriRC_predict_Y').detach().numpy().T
    rc_Y = rc_Y[:, start: end]
    rc_x1 = l1 * np.sin(rc_Y[0, :])
    rc_y1 = - l1 * np.cos(rc_Y[0, :])
    rc_x2 = rc_x1 + l2 * np.sin(rc_Y[1, :])
    rc_y2 = rc_y1 - l2 * np.cos(rc_Y[1, :])

    fig1, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(10, 7))
    xlim = (-(l1 + l2), l1 + l2)
    ylim = (-(l1 + l2), 0.05)
    
    axs[0, 0].plot(td_x1, td_y1, c='blue')
    axs[0, 0].plot(td_x2, td_y2, c='red')

    axs[0, 0].set_aspect(1)
    axs[0, 0].set_xlabel("x")
    axs[0, 0].set_ylabel("y")
    axs[0, 0].set_title("true")
    axs[0, 0].set_xlim(xlim)
    axs[0, 0].set_ylim(ylim)
    

    axs[0, 1].plot(id_x1, id_y1, c='blue')
    axs[0, 1].plot(id_x2, id_y2, c='red')

    axs[0, 1].set_aspect(1)
    axs[0, 1].set_xlabel("x")
    axs[0, 1].set_ylabel("y")
    axs[0, 1].set_title("ODE45")
    axs[0, 1].set_xlim(xlim)
    axs[0, 1].set_ylim(ylim)
    

    axs[1, 0].plot(rc_x1, rc_y1, c='blue')
    axs[1, 0].plot(rc_x2, rc_y2, c='red')

    axs[1, 0].set_aspect(1)
    axs[1, 0].set_xlabel("x")
    axs[1, 0].set_ylabel("y")
    axs[1, 0].set_title("RC")
    axs[1, 0].set_xlim(xlim)
    axs[1, 0].set_ylim(ylim)
    

    axs[1, 1].plot(ad_x1, ad_y1, c='blue')
    axs[1, 1].plot(ad_x2, ad_y2, c='red')

    axs[1, 1].set_aspect(1)
    axs[1, 1].set_xlabel("x")
    axs[1, 1].set_ylabel("y")
    axs[1, 1].set_title("DFM")
    axs[1, 1].set_xlim(xlim)
    axs[1, 1].set_ylim(ylim)

    # fig1.legend(["true", "id", "ad", "rc"])
    fig1.savefig(f'./error_pic/cmp_{ex}_2d.png', dpi=500)

    
    colors = ['blue', 'green', 'orange', 'red']
    fig2, axs = plt.subplots(nrows=2, ncols=4, sharex=True, figsize=(24, 10))
    xlim1 = (-(l1 + l2), l1 + l2)
    ylim1 = (-(l1 + 0.05), 0)
    xlim2 = (-(l1 + l2), l1 + l2)
    ylim2 = (-(l1 + l2), 0.05)
    alphas = [0.7, 0.7, 0.7, 0.7]
    linewidths = [3, 3, 3, 3]

    # id
    axs[0, 0].plot(td_x1, td_y1, c=colors[0], linewidth=linewidths[0])
    axs[0, 0].plot(id_x1, id_y1, c=colors[3], alpha=alphas[1], linewidth=linewidths[1])

    axs[0, 0].set_aspect(1)
    axs[0, 0].set_xlabel("x")
    axs[0, 0].set_ylabel("y")
    axs[0, 0].set_title("mass1")
    axs[0, 0].set_xlim(xlim1)
    axs[0, 0].set_ylim(ylim1)
    
    axs[1, 0].plot(td_x2, td_y2, c=colors[0], linewidth=linewidths[0])
    axs[1, 0].plot(id_x2, id_y2, c=colors[3], alpha=alphas[1], linewidth=linewidths[1])

    for i in range(len(id_x2)):
        axs[1, 0].plot([td_x2[i], id_x2[i]], [td_y2[i], id_y2[i]], alpha=0.4, linestyle='--', color='gray')

    axs[1, 0].set_aspect(1)
    axs[1, 0].set_xlabel("x")
    axs[1, 0].set_ylabel("y")
    axs[1, 0].set_title("mass2")
    axs[1, 0].set_xlim(xlim2)
    axs[1, 0].set_ylim(ylim2)
    
    # pinn
    axs[0, 1].plot(td_x1, td_y1, c=colors[0], linewidth=linewidths[0])
    axs[0, 1].plot(pinn_x1, pinn_y1, c=colors[3], alpha=alphas[2], linewidth=linewidths[2])

    axs[0, 1].set_aspect(1)
    axs[0, 1].set_xlabel("x")
    axs[0, 1].set_ylabel("y")
    axs[0, 1].set_title("mass1")
    axs[0, 1].set_xlim(xlim1)
    axs[0, 1].set_ylim(ylim1)
    
    axs[1, 1].plot(td_x2, td_y2, c=colors[0], linewidth=linewidths[0])
    axs[1, 1].plot(pinn_x2, pinn_y2, c=colors[3], alpha=alphas[2], linewidth=linewidths[2])

    for i in range(len(pinn_x2)):
        axs[1, 1].plot([td_x2[i], pinn_x2[i]], [td_y2[i], pinn_y2[i]], alpha=0.4, linestyle='--', color='gray')

    axs[1, 1].set_aspect(1)
    axs[1, 1].set_xlabel("x")
    axs[1, 1].set_ylabel("y")
    axs[1, 1].set_title("mass2")
    axs[1, 1].set_xlim(xlim2)
    axs[1, 1].set_ylim(ylim2)
    
    # rc
    axs[0, 2].plot(td_x1, td_y1, c=colors[0], linewidth=linewidths[0])
    axs[0, 2].plot(rc_x1, rc_y1, c=colors[3], alpha=alphas[2], linewidth=linewidths[2])

    axs[0, 2].set_aspect(1)
    axs[0, 2].set_xlabel("x")
    axs[0, 2].set_ylabel("y")
    axs[0, 2].set_title("mass1")
    axs[0, 2].set_xlim(xlim1)
    axs[0, 2].set_ylim(ylim1)
    
    axs[1, 2].plot(td_x2, td_y2, c=colors[0], linewidth=linewidths[0])
    axs[1, 2].plot(rc_x2, rc_y2, c=colors[3], alpha=alphas[2], linewidth=linewidths[2])

    for i in range(len(rc_x2)):
        axs[1, 2].plot([td_x2[i], rc_x2[i]], [td_y2[i], rc_y2[i]], alpha=0.4, linestyle='--', color='gray')

    axs[1, 2].set_aspect(1)
    axs[1, 2].set_xlabel("x")
    axs[1, 2].set_ylabel("y")
    axs[1, 2].set_title("mass2")
    axs[1, 2].set_xlim(xlim2)
    axs[1, 2].set_ylim(ylim2)
    
    # ad
    axs[0, 3].plot(td_x1, td_y1, c=colors[0], linewidth=linewidths[0])
    axs[0, 3].plot(ad_x1, ad_y1, c=colors[3], alpha=alphas[3], linewidth=linewidths[3])

    axs[0, 3].set_aspect(1)
    axs[0, 3].set_xlabel("x")
    axs[0, 3].set_ylabel("y")
    axs[0, 3].set_title("mass1")
    axs[0, 3].set_xlim(xlim1)
    axs[0, 3].set_ylim(ylim1)
    
    axs[1, 3].plot(td_x2, td_y2, c=colors[0], linewidth=linewidths[0])
    axs[1, 3].plot(ad_x2, ad_y2, c=colors[3], alpha=alphas[3], linewidth=linewidths[3])

    for i in range(len(ad_x2)):
        axs[1, 3].plot([td_x2[i], ad_x2[i]], [td_y2[i], ad_y2[i]], alpha=0.4, linestyle='--', color='gray')

    axs[1, 3].set_aspect(1)
    axs[1, 3].set_xlabel("x")
    axs[1, 3].set_ylabel("y")
    axs[1, 3].set_title("mass2")
    axs[1, 3].set_xlim(xlim2)
    axs[1, 3].set_ylim(ylim2)

    # fig1.legend(["true", "id", "ad", "rc"])
    fig2.legend(['true', 'predict'], loc='upper center', ncol=2)
    fig2.savefig(f'./error_pic/cmp_{ex}_2d_2.png', dpi=500)
    fig2.savefig(f'./error_pic/cmp_{ex}_2d_2.svg', dpi=500)


if __name__=='__main__':
  
  # cmp_nrmse()
  # error_cmp(ex=13)
  # error_cmp(ex=22)
  # error_cmp(ex=23)
  # error_cmp(ex=24)
  # error_cmp(ex=25)
#   error_cmp(ex=1050300)

    # plot_cmp(ex=1050300)
    
    # plot_2d(ex=1050300)
    plot_cmp(ex=901000001)
    
    plot_2d(ex=901000001)
  


  # res_cmp(ex=23)
    # base_path = f'./'
    # pre_train_ex = f'rb_pre_without_model_ex1'
    # # acc_data_filename = f'./datas/acc_data_{pre_train_ex}.npy'
    # # train netword，get 0~10s y,z data, by x data
    # trained_net = torch.load(f'{base_path}/model/model_{pre_train_ex}.pth')
    # # params = torch.load(f'/home/cxz/rb_pre_without_model1/params/rb_pre_without_model_ex213.pt')
    # ksi = torch.load(f'{base_path}/ksis/ksi_{pre_train_ex}.pt')
    # print(ksi)