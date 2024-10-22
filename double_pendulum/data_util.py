import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from dp_ode import dp_ode

def init_data_noise(desired_snr):
    data = dp_ode(time_span=(0, 25), ic=[-8, 7, 27], t_eval=torch.linspace(0, 25, 1001), method='RK45')
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
    
    return t, noisy_signal


def init_data():
    maxtime = 40
    t_eval = np.linspace(0, maxtime, int(40 * maxtime + 1))
    sol = dp_ode(time_span=[0, maxtime], ic=[math.pi * 1 / 4 , math.pi * 1 / 5, 0, 0], t_eval=t_eval, method="RK45")

    return sol.t, sol.y


def read_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    t = data[0, :]
    noise_data = data[1:, :]
    return t, noise_data


if __name__ == '__main__':
    # desired_snr = 60
    # # t, data = read_noise_data('./datas/lorenz_noise_snr40.csv')
    # t, data = init_data_noise(desired_snr=desired_snr)
    # np.savetxt(f'./datas/lorenz_noise_snr{desired_snr}.csv', np.concatenate((t.reshape(1, -1), data), axis=0), delimiter=',')

    t, data = init_data()
    np.savetxt(f'./csvs/d_pendulum.csv', np.concatenate((t.reshape(1, -1), data), axis=0), delimiter=',')

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
