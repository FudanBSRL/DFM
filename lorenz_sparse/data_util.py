import torch
import numpy as np
import matplotlib.pyplot as plt
from lorenz_ode import lorenz_ode

def init_data_noise(desired_snr):
    data = lorenz_ode(time_span=(0, 25), ic=[-8, 7, 27], t_eval=torch.linspace(0, 25, 1001), method='RK45')
    t = data.t
    acc_data = data.y

    # 计算信号的功率
    signal_power = np.mean(np.square(acc_data))

    # 根据所需的SNR计算噪声的功率
    noise_power = signal_power / (10**(desired_snr/10))

    # 生成随机噪声
    noise = np.random.normal(0, np.sqrt(noise_power), acc_data.shape)

    # 添加噪声到原始信号上
    noisy_signal = acc_data + noise
    
    return t, noisy_signal, acc_data


def init_data(maxtime, dt):
    data = lorenz_ode(time_span=(0, maxtime), ic=[-8, 7, 27], t_eval=np.linspace(0, maxtime, round(maxtime / dt) + 1), method='RK45')
    t = data.t.reshape(1, -1)
    print(t)
    acc_data = data.y

    np.savetxt(f'./datas/lorenz_sparse.csv', np.concatenate((t, acc_data), axis=0), delimiter=',')


def read_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    t = data[0, :]
    acc_data = data[1:, :]
    return t, acc_data


if __name__ == '__main__':
    init_data(25, 0.05)
    # np.savetxt('./datas/lorenz_noise_snr40.csv', (t, data), delimiter=',')

    # color = ['red', 'green']
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(origin[0, :], origin[1, :], origin[2, :], c=color[1], linewidth=0.5)
    # ax.plot(data[0, :], data[1, :], data[2, :], c=color[0], linewidth=0.5)

    # fig1, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=False)
    # ax1.plot(data[0, :], data[1, :], c=color[0], linestyle='-', linewidth=0.5)  # 25s
    # ax1.plot(origin[0, :], origin[1, :], c=color[1], linestyle='-', linewidth=0.5)  # 25s

    # # ax1.set_xlim([0, 25])
    # # fig1.legend(['true', 'RC predict'], loc='upper center', ncol=2)

    # ax2.plot(data[0, :], data[2, :], c=color[0], linestyle='-', linewidth=0.5)  # 25s
    # ax2.plot(origin[0, :], origin[2, :], c=color[1], linestyle='-', linewidth=0.5)  # 25s
    
    # # ax2.set_xlabel('t')
    # # ax2.set_ylabel('y')
    # # ax2.set_xlim([0, 25])

    # ax3.plot(data[1, :], data[2, :], c=color[0], linestyle='-', linewidth=0.5)  # 25s
    # ax3.plot(origin[1, :], origin[2, :], c=color[1], linestyle='-', linewidth=0.5)  # 25s

    # ax3.set_xlabel('t')
    # ax3.set_ylabel('z')
    # ax3.set_xlim([0, 25])
    # fig.savefig(f'test_noise_pic.png', dpi=500)

    # color = ['purple', 'red', 'green']
    # fig1, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=False)
    # ax1.plot(t, data[0, :], c=color[0], linestyle='-', linewidth=0.5)  # 25s

    # ax1.set_xlim([0, 25])
    # fig1.legend(['true', 'RC predict'], loc='upper center', ncol=2)

    # ax2.plot(t, data[1, :], c=color[0], linestyle='-', linewidth=0.5)  # 25s
    
    # ax2.set_xlabel('t')
    # ax2.set_ylabel('y')
    # ax2.set_xlim([0, 25])

    # ax3.plot(t, data[2, :], c=color[0], linestyle='-', linewidth=0.5)  # 25s

    # ax3.set_xlabel('t')
    # ax3.set_ylabel('z')
    # ax3.set_xlim([0, 25])
    # fig1.savefig(f'test_noise_pic.png')
