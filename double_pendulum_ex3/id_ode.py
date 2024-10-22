import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
from data_util import read_data_ex

ex = 'rb_pre_without_model_ex4'
Ksi = None
y_0 = None


def lorenz_id(t, y):

    print(t)

    feature_list = [1.0]

    base_list = []
    for i in range(4):
        base_list.append(y[i])

    for i in range(2):
        base_list.append(np.sin(y[i]))

    for i in range(2):
        base_list.append(np.cos(y[i]))

    feature_list.extend(base_list)

    for i in range(len(base_list)):
        for j in range(i, len(base_list)):
            feature_list.append(base_list[i] * base_list[j])
            if i < 4 and j < 4:
                feature_list.append(np.sin(base_list[i] + base_list[j]))
                feature_list.append(np.cos(base_list[i] + base_list[j]))
                if i != j:
                    feature_list.append(np.sin(base_list[i] - base_list[j]))
                    feature_list.append(np.cos(base_list[i] - base_list[j]))

    for i in range(len(base_list)):
        for j in range(i, len(base_list)):
            for k in range(j, len(base_list)):
                feature_list.append(base_list[i] * base_list[j] * base_list[k])

    for i in range(len(base_list)):
        for j in range(i, len(base_list)):
            for k in range(j, len(base_list)):
                for l in range(k, len(base_list)):
                    feature_list.append(base_list[i] * base_list[j] * base_list[k] * base_list[l])

    # x1 = 0
    # y1 = 0
    # x2 = 0
    # y2 = 0

    # x1_dot = 0
    # y1_dot = 0
    # x2_dot = 0
    # y2_dot = 0

    th1 = 0
    th2 = 0
    w1 = 0
    w2 = 0

    for i in range(len(feature_list)):
        th1 += Ksi[i, 0] * feature_list[i]
        th2 += Ksi[i, 1] * feature_list[i]
        w1 += Ksi[i, 2] * feature_list[i]
        w2 += Ksi[i, 3] * feature_list[i]
        # x1 += Ksi[i, 0] * feature_list[i]
        # y1 += Ksi[i, 1] * feature_list[i]
        # x2 += Ksi[i, 2] * feature_list[i]
        # y2 += Ksi[i, 3] * feature_list[i]
        # x1_dot += Ksi[i, 4] * feature_list[i]
        # y1_dot += Ksi[i, 5] * feature_list[i]
        # x2_dot += Ksi[i, 6] * feature_list[i]
        # y2_dot += Ksi[i, 7] * feature_list[i]

    return [th1, th2, w1, w2]


def lorenz_ode_id(time_span, ic, t_eval, method):
    lorenz_soln = solve_ivp(lorenz_id, time_span, ic, t_eval=t_eval, method=method)
    # lorenz_soln = solve_ivp(lorenz_id, time_span, ic, t_eval=t_eval, method=method, rtol=1e-10, atol=1e-10)
    return lorenz_soln


if __name__ == '__main__':
    torch.set_printoptions(threshold=10_000)

    base_path = f'./'
    trained_net = torch.load(f'{base_path}/model/model_{ex}.pth')

    Ksi = torch.load(f'./ksis/ksi_{ex}.pt').detach().numpy()
    # Ksi = torch.load(f'./ksi_{ex}_tmp3.pt')
    torch.save(Ksi, f'./ksi_{ex}_tmp.pt')
    # print(Ksi[0:10, :])
    # print(Ksi.shape)
    # print((Ksi[:, 0] != 0).sum())
    # print((Ksi[:, 1] != 0).sum())
    # print((Ksi[:, 2] != 0).sum())
    # print((Ksi[:, 3] != 0).sum())

    dt = 0.01

    acc_t, acc_data, _, _ = read_data_ex(f'./csvs/ex_data1.CSV', rate=round(1/dt), filter=False)

    # y_trained_reg = trained_net(torch.from_numpy(acc_t[0: 21*round(1/dt) + 1]).float().reshape(-1, 1) - 15).double().detach().numpy().T
    
    train_time = 21
    test_time = 5
    # ic = y_trained_reg[:, int(train_time * 40)]
    ic = acc_data[:, int(train_time * round(1 / dt))]

    id_data = lorenz_ode_id(time_span=(0, test_time), ic=ic,
                            t_eval=np.linspace(0, test_time, round(test_time * round(1 / dt) + 1)), method='RK45')
    id_data.t = id_data.t + train_time

    xlim = (train_time, train_time + test_time)
    # 结果对比
    color = ['blue', 'red']
    fig1, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=False)
    ax1.plot(acc_t, acc_data[0, :], c=color[0], linestyle='-')  # 25s
    ax1.plot(id_data.t, id_data.y[0, :], c=color[1], linestyle='-')  # trian data
    ax1.set_xlim(xlim)

    ax2.plot(acc_t, acc_data[1, :], c=color[0], linestyle='-')  # 25s
    ax2.plot(id_data.t, id_data.y[1, :], c=color[1], linestyle='-')  # trian data
    ax2.set_xlim(xlim)
    ax2.set_ylim((-45, -25))

    ax3.plot(acc_t, acc_data[2, :], c=color[0], linestyle='-')  # 25s
    ax3.plot(id_data.t, id_data.y[2, :], c=color[1], linestyle='-')  # trian data
    ax3.set_xlim(xlim)

    ax4.plot(acc_t, acc_data[3, :], c=color[0], linestyle='-')  # 25s
    ax4.plot(id_data.t, id_data.y[3, :], c=color[1], linestyle='-')  # trian data
    ax4.set_xlim(xlim)

    # ax5.plot(acc_t, acc_data[4, :], c=color[0], linestyle='-')  # 25s
    # ax5.plot(id_data.t, id_data.y[4, :], c=color[1], linestyle='--') # trian data
    # ax5.set_xlim(xlim)

    # ax6.plot(acc_t, acc_data[5, :], c=color[0], linestyle='-')  # 25s
    # ax6.plot(id_data.t, id_data.y[5, :], c=color[1], linestyle='--') # trian data
    # ax6.set_xlim(xlim)

    # ax7.plot(acc_t, acc_data[6, :], c=color[0], linestyle='-')  # 25s
    # ax7.plot(id_data.t, id_data.y[6, :], c=color[1], linestyle='--') # trian data
    # ax7.set_xlim(xlim)

    # ax8.plot(acc_t, acc_data[7, :], c=color[0], linestyle='-')  # 25s
    # ax8.plot(id_data.t, id_data.y[7, :], c=color[1], linestyle='--') # trian data
    # ax8.set_xlim(xlim)

    fig1.legend(['true', 'ODE45'], loc='upper center', ncol=2)
    fig1.savefig('res.png', dpi=500)

    td_T = torch.from_numpy(acc_t).double()
    td_Y = torch.from_numpy(acc_data).double()

    id_T = td_T
    id_Y = torch.from_numpy(id_data.y).double()

    id_Y = torch.cat((td_Y[:, 0:train_time*round(1/dt)+1], id_Y[:, 1:]), axis=1)

    # print(id_Y.shape)
    torch.save(id_T, "./datas/id_T.pt")
    torch.save(id_Y, "./datas/id_Y.pt")

    err_Y = np.sqrt(torch.sum((id_Y[:, :1200] - acc_data[:, :1200]) ** 2, axis=0))

    # 误差
    fig1, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=False)
    ax1.plot(id_T[:1200], err_Y, c=color[1], linestyle='--')  # trian data

    ax1.set_xlim(xlim)

    fig1.savefig('cmp.png', dpi=500)
