import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from dp_ode import dp_ode, l1, l2

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
    sol = dp_ode(time_span=[0, maxtime], ic=[math.pi * 1 / 4 , math.pi * (-1), 0, 0], t_eval=t_eval, method="RK45")

    return sol.t, sol.y


def read_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    t = data[0, :]
    noise_data = data[1:, :]
    return t, noise_data


def pos2dagl(x, y):
    if y < 0:
        return np.arctan(x / (-y))
    
    if y > 0 and x > 0:
        return np.pi / 2 + np.arctan(y / x)
    
    if y > 0 and x < 0:
        return - np.pi / 2 - np.arctan(y / (-x))


# 移动平均滤波函数
def moving_average_filter(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')


def read_data_ex(filename, rate=40, filter=True):
    data_all = np.loadtxt(filename, delimiter=',', dtype=np.float32)
    # print(data_all.dtype)
    t = data_all[:, 0]
    data = data_all[:, 1:].T

    noise_data_t = (data[:, 1:] - data[:, 0:-1]) / (t[1:] - t[:-1])

    time_tmp = np.linspace(10, 70, 60 * 100 + 1)
    x1 = np.interp(time_tmp, t, data[0, :])
    y1 = np.interp(time_tmp, t, data[1, :])
    x2 = np.interp(time_tmp, t, data[2, :])
    y2 = np.interp(time_tmp, t, data[3, :])

    l1 = np.mean(np.sqrt(x1**2 + y1**2))
    l2 = np.mean(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))

    th1 = np.zeros_like(time_tmp).reshape(1, -1)
    th2 = np.zeros_like(time_tmp).reshape(1, -1)
    th1_c = np.zeros_like(time_tmp).reshape(1, -1)
    th2_c = np.zeros_like(time_tmp).reshape(1, -1)
    
    offset1 = 0
    offset2 = 0
    for i in range(60 * 100 + 1):
        th1[0, i] = pos2dagl(x1[i], y1[i])
        th2[0, i] = pos2dagl(x2[i] - x1[i], y2[i] - y1[i])
        if i > 0:
            if th1[0, i] - th1[0, i - 1] > np.pi:
                offset1 -= 2 * np.pi
            if th1[0, i] - th1[0, i - 1] < -np.pi:
                offset1 += 2 * np.pi
            if th2[0, i] - th2[0, i - 1] > np.pi:
                offset2 -= 2 * np.pi
            if th2[0, i] - th2[0, i - 1] < -np.pi:
                offset2 += 2 * np.pi
        th1_c[0, i] = th1[0, i] + offset1
        th2_c[0, i] = th2[0, i] + offset2

    if filter:
        window_size = round(rate / 20 + 1)
        # window_size = 11
        th1_c = moving_average_filter(th1_c[0, :], window_size).reshape(1, -1)
        th2_c = moving_average_filter(th2_c[0, :], window_size).reshape(1, -1)

    th = np.concatenate((th1_c, th2_c), axis=0)
    th_c = np.concatenate((th1_c, th2_c), axis=0)
    w = (th_c[:, 2:] - th_c[:, 0:-2]) / (time_tmp[2:] - time_tmp[0:-2])

    time = np.linspace(20, 60, 40 * rate + 1)
    th1 = np.interp(time, time_tmp, th[0, :]).reshape(1, -1)
    th2 = np.interp(time, time_tmp, th[1, :]).reshape(1, -1)

    w1 = np.interp(time, time_tmp[1:-1], w[0, :]).reshape(1, -1)
    w2 = np.interp(time, time_tmp[1:-1], w[1, :]).reshape(1, -1)

    time -= 20
    return time, np.concatenate((th1, th2, w1, w2), axis=0), l1, l2

    
    # # print(l1, l2)

    # # th1 = np.arctan2(x1, -y1)
    # # th2 = np.arctan2(x2 - x1, -y2 + y1)

    # th1 = np.arctan(x1 / -y1)
    # th2 = np.arctan((x2 - x1) / (-y2 + y1))

    # print(np.arctan(np.inf))
    
    # for i in range(40 * 40 + 1):
    #     if x1[0, i] > 0 and y1[0, i] > 0:
    #         th1[0, i] += np.pi / 2
    #     if x1[0, i] < 0 and y1[0, i] > 0:
    #         th1[0, i] -= np.pi / 2

    #     if x2[0, i] > x1[0, i] and y2[0, i] > y1[0, i]:
    #         th2[0, i] = - th2[0, i] + np.pi / 2
    #     if x2[0, i] < x1[0, i] and y2[0, i] > y1[0, i]:
    #         th2[0, i] = - th2[0, i] - np.pi / 2

    # w1 = ((x1_t * (-y1) - x1 * (-y1_t)) / y1**2) / (1 + (x1 / -y1)**2)
    # w2 = (((x2_t - x1_t) * (-y2 + y1) - (x2 - x1) * (y2_t - y1_t)) / (-y2 + y1)**2) / (1 + ((x2 - x1) / (-y2 + y1))**2)

    # # th1 = np.arcsin(x1 / np.sqrt(x1**2 + y1**2))
    # # th2 = np.arcsin((x2 - x1) / np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
    # # w1 = ( (x1_t * np.sqrt(x1**2 + y1**2) - x1 * (2 * x1 * x1_t + 2 * y1 * y1_t) / (2 * np.sqrt(x1**2 + y1**2))) / (x1**2 + y1**2) ) / \
    # #         np.sqrt(1 - (x1 / np.sqrt(x1**2 + y1**2))**2)
    # # w2 = (((x2_t - x1_t) * np.sqrt((x2 - x1)**2 + (y2 - y1)**2) - (x2 - x1) * (2 * (x2 - x1) * (x2_t - x1_t) + 2 * (y2 - y1) * (y2_t - y1_t)) / (2 * np.sqrt((x2 - x1)**2 + (y2 - y1)**2))) / ((x2 - x1)**2 + (y2 - y1)**2)) / \
    # #         np.sqrt(1 - (np.sqrt((x2 - x1)**2 + (y2 - y1)**2))**2)
    
    # # th1 = np.arctan2(x1 / y1)
    # # th2 = np.arctan2(x2 / y2)
    
    # time -= 20
    # return time, np.concatenate((th1, th2, w1, w2), axis=0)


    # return time, np.concatenate((x1, y1, x2, y2, x1_t, y1_t, x2_t, y2_t), axis=0)


def file2dat(filename):
    data_all = np.loadtxt(filename, delimiter=',', dtype=np.float32)
    # print(data_all.dtype)
    t = data_all[:, 0]
    data = data_all[:, 1:].T

    noise_data_t = (data[:, 1:] - data[:, 0:-1]) / (t[1:] - t[:-1])

    time_tmp = np.linspace(10, 70, 60 * 100 + 1)
    x1 = np.interp(time_tmp, t, data[0, :])
    y1 = np.interp(time_tmp, t, data[1, :])
    x2 = np.interp(time_tmp, t, data[2, :])
    y2 = np.interp(time_tmp, t, data[3, :])

    l1 = np.mean(np.sqrt(x1**2 + y1**2))
    l2 = np.mean(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))

    th1 = np.zeros_like(time_tmp).reshape(1, -1)
    th2 = np.zeros_like(time_tmp).reshape(1, -1)
    th1_c = np.zeros_like(time_tmp).reshape(1, -1)
    th2_c = np.zeros_like(time_tmp).reshape(1, -1)
    
    offset1 = 0
    offset2 = 0
    for i in range(60 * 100 + 1):
        th1[0, i] = pos2dagl(x1[i], y1[i])
        th2[0, i] = pos2dagl(x2[i] - x1[i], y2[i] - y1[i])
        if i > 0:
            if th1[0, i] - th1[0, i - 1] > np.pi:
                offset1 -= 2 * np.pi
            if th1[0, i] - th1[0, i - 1] < -np.pi:
                offset1 += 2 * np.pi
            if th2[0, i] - th2[0, i - 1] > np.pi:
                offset2 -= 2 * np.pi
            if th2[0, i] - th2[0, i - 1] < -np.pi:
                offset2 += 2 * np.pi
        th1_c[0, i] = th1[0, i] + offset1
        th2_c[0, i] = th2[0, i] + offset2


    th = np.concatenate((th1_c, th2_c), axis=0)
    th_c = np.concatenate((th1_c, th2_c), axis=0)
    w = (th_c[:, 2:] - th_c[:, 0:-2]) / (time_tmp[2:] - time_tmp[0:-2])

    time = np.linspace(20, 50, 30 * 100 + 1)
    th1 = np.interp(time, time_tmp, th[0, :]).reshape(-1, 1)
    th2 = np.interp(time, time_tmp, th[1, :]).reshape(-1, 1)

    w1 = np.interp(time, time_tmp[1:-1], w[0, :]).reshape(-1, 1)
    w2 = np.interp(time, time_tmp[1:-1], w[1, :]).reshape(-1, 1)

    time -= 20
    np.savetxt(f'./csvs/dp2.dat', np.concatenate((th1, th2, w1, w2), axis=1), delimiter=' ', fmt="%.10f")
    


if __name__ == '__main__':

    filename = f'./csvs/ex_data1.CSV'
    file2dat(filename)

    # read_data_ex(f'./csvs/ex_data1.CSV')
    # t, data, l1, l2 = read_data_ex(f'./csvs/ex_data1.CSV', rate=100)


    # fig2, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True)
    # xlim = (0, 30)
    # ax1.plot(t, data[0, :], linestyle='-')
    # ax1.set_xlim(xlim)
    # ax2.plot(t, data[1, :], linestyle='-')
    # ax2.set_xlim(xlim)
    # ax3.plot(t, data[2, :], linestyle='-')
    # ax3.set_xlim(xlim)
    # ax4.plot(t, data[3, :], linestyle='-')
    # ax4.set_xlim(xlim)
    # fig2.savefig("test.png", dpi=500)
    # fig2.savefig("test.svg", dpi=500)

    # x1 = l1 * np.sin(data[0, :])
    # y1 = - l1 * np.cos(data[0, :])
    # x2 = x1 + l2 * np.sin(data[1, :])
    # y2 = y1 - l2 * np.cos(data[1, :])

    # fig3, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=False)
    # ax1.plot(x1, y1, linestyle='-')
    # ax1.set_aspect(1)
    # ax2.plot(x2, y2, linestyle='-')
    # ax2.set_aspect(1)
    # fig3.savefig("test_2d.png", dpi=500)
    # fig3.savefig("test_2d.svg", dpi=500)



    # desired_snr = 60
    # # t, data = read_noise_data('./datas/lorenz_noise_snr40.csv')
    # t, data = init_data_noise(desired_snr=desired_snr)
    # np.savetxt(f'./datas/lorenz_noise_snr{desired_snr}.csv', np.concatenate((t.reshape(1, -1), data), axis=0), delimiter=',')

    # t, data = init_data()
    # np.savetxt(f'./csvs/d_pendulum_f3.csv', np.concatenate((t.reshape(1, -1), data), axis=0), delimiter=',')
    # fig2, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=False)
    # ax1.plot(t, data[0, :], linestyle='-')
    # ax2.plot(t, data[1, :], linestyle='-')
    # ax3.plot(t, data[2, :], linestyle='-')
    # ax4.plot(t, data[3, :], linestyle='-')
    # fig2.savefig("test.png", dpi=500)

    # theta1, theta2, omega1, omega2 = data[0], data[1], data[2], data[3]
    # x1 = l1 * np.sin(theta1)
    # x1_dot = l1 * np.cos(theta1) * omega1
    # y1 = -l1 * np.cos(theta1)
    # y1_dot = l1 * np.sin(theta1) * omega1
    # x2 = x1 + l2 * np.sin(theta2)
    # x2_dot = x1_dot + l2 * np.cos(theta2) * omega2
    # y2 = y1 - l2 * np.cos(theta2)
    # y2_dot = y1_dot + l2 * np.sin(theta2) * omega2

    # np.savetxt(f'./csvs/d_pendulum_f3_xy.csv', 
    #            np.concatenate(
    #                (
    #                    t.reshape(1, -1), 
    #                    x1.reshape(1, -1), 
    #                    y1.reshape(1, -1), 
    #                    x2.reshape(1, -1), 
    #                    y2.reshape(1, -1), 
    #                    x1_dot.reshape(1, -1), 
    #                    y1_dot.reshape(1, -1), 
    #                    x2_dot.reshape(1, -1), 
    #                    y2_dot.reshape(1, -1), 
    #                 ), axis=0), delimiter=',')

    # fig3, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(nrows=8, ncols=1, sharex=True)
    # ax1.plot(t, x1, linestyle='-')
    # ax2.plot(t, y1, linestyle='-')
    # ax3.plot(t, x2, linestyle='-')
    # ax4.plot(t, y2, linestyle='-')
    # ax5.plot(t, x1_dot, linestyle='-')
    # ax6.plot(t, y1_dot, linestyle='-')
    # ax7.plot(t, x2_dot, linestyle='-')
    # ax8.plot(t, y2_dot, linestyle='-')
    # fig3.savefig("test_xy.png", dpi=500)

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
