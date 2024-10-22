import torch
import numpy as np
from dp_ode import dp_ode
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from data_util import read_data


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
    td_T, td_Y = read_data(f'./csvs/d_pendulum.csv')
    
    id_T = torch.load(f'./datas/id_T.pt').detach().numpy()
    id_Y = torch.load(f'./datas/id_Y.pt').detach().numpy()

    ad_T = torch.load(f'./datas/T_{ex}.pt').detach().numpy()
    ad_Y = torch.load(f'./datas/ad_Y_{ex}.pt').detach().numpy().T

    rc_t = torch.load(f'./datas/oriRC_predict_T').detach().numpy()
    rc_Y = torch.load(f'./datas/oriRC_predict_Y').detach().numpy().T
    
    error_id = td_Y - id_Y
    error_ad = td_Y - ad_Y
    error_rc = td_Y - rc_Y

    xlim = (20, 40)

    fig1, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=False)
    ax1.plot(id_T, error_id[0, :])
    ax1.plot(id_T, error_ad[0, :])
    ax1.plot(id_T, error_rc[0, :])
    ax1.set_xlim(xlim)

    ax2.plot(id_T, error_id[1, :])
    ax2.plot(id_T, error_ad[1, :])
    ax2.plot(id_T, error_rc[1, :])
    ax2.set_xlim(xlim)
    
    ax3.plot(id_T, error_id[2, :])
    ax3.plot(id_T, error_ad[2, :])
    ax3.plot(id_T, error_rc[2, :])
    ax3.set_xlim(xlim)
    
    ax4.plot(id_T, error_id[3, :])
    ax4.plot(id_T, error_ad[3, :])
    ax4.plot(id_T, error_rc[3, :])
    ax4.set_xlim(xlim)

    fig1.legend(["id", "ad", "rc"])
    fig1.savefig(f'./error_pic/error_cmp_{ex}.png', dpi=500)

    
    l2_id = np.sqrt(np.sum((td_Y - id_Y) ** 2, axis=0))
    l2_ad = np.sqrt(np.sum((td_Y - ad_Y) ** 2, axis=0))
    l2_rc = np.sqrt(np.sum((td_Y - rc_Y) ** 2, axis=0))
    fig2, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=False)
    ax1.plot(id_T, l2_id)
    ax1.plot(id_T, l2_ad)
    ax1.plot(id_T, l2_rc)
    ax1.set_xlim(xlim)
    fig2.legend(['ODE45', 'MMPINN', 'RC'], loc='upper center', ncol=3)
    fig2.align_labels()
    fig2.savefig(f'./error_pic/l2_cmp_{ex}.png', dpi=500)



if __name__=='__main__':
  

  # error_cmp(ex=13)
  # error_cmp(ex=22)
  # error_cmp(ex=23)
  # error_cmp(ex=24)
  # error_cmp(ex=25)
  # res_cmp(ex=23)
    base_path = f'./'
    pre_train_ex = f'rb_pre_without_model_ex1'
    # acc_data_filename = f'./datas/acc_data_{pre_train_ex}.npy'
    # train netword，get 0~10s y,z data, by x data
    trained_net = torch.load(f'{base_path}/model/model_{pre_train_ex}.pth')
    # params = torch.load(f'/home/cxz/rb_pre_without_model1/params/rb_pre_without_model_ex213.pt')
    ksi = torch.load(f'{base_path}/ksis/ksi_{pre_train_ex}.pt')
    print(ksi)