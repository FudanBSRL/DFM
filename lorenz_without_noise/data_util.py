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
    
    return t, noisy_signal


def read_noise_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    t = data[0, :]
    noise_data = data[1:, :]
    return t, noise_data


def init_data(maxtime, dt):
    data = lorenz_ode(time_span=(0, maxtime), ic=[-8, 7, 27], t_eval=np.linspace(0, maxtime, round(maxtime / dt) + 1), method='RK45')
    t = data.t.reshape(1, -1)
    print(t)
    acc_data = data.y

    np.savetxt(f'./datas/lorenz_{maxtime}.csv', np.concatenate((t, acc_data), axis=0), delimiter=',')


def read_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    t = data[0, :]
    acc_data = data[1:, :]
    return t, acc_data


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


def data_to_csv():
    # ex_ad = 243
    # ex_pinn = 7

    # ad_T = torch.load(f'./datas/T_{ex_ad}.pt').detach().numpy()
    # ad_Y = torch.load(f'./datas/ad_Y_{ex_ad}.pt').detach().numpy().T
    # ad_data = np.concatenate((ad_T.reshape(1, -1), ad_Y), axis=0)
    # np.savetxt("./csvs/ad_data_05_095.csv", ad_data, delimiter=",")

    # rc_T = torch.load('./datas/rc_predict_T').detach().numpy()
    # rc_Y = torch.load('./datas/rc_predict_Y').detach().numpy().T
    # rc_data = np.concatenate((rc_T.reshape(1, -1), rc_Y), axis=0)
    # np.savetxt("./csvs/rc_data.csv", rc_data, delimiter=",")
    
    # pinn_T = torch.load(f'./datas/T_PINN{ex_pinn}.pt').detach().numpy()
    # pinn_Y = torch.load(f'./datas/Y_PINN{ex_pinn}.pt').detach().numpy().T
    # pinn_data = np.concatenate((pinn_T.reshape(1, -1), pinn_Y), axis=0)
    # np.savetxt("./csvs/pinn_data.csv", pinn_data, delimiter=",")

    # ode_T = torch.load('./datas/id_T.pt').detach().numpy()
    # ode_Y = torch.load('./datas/id_Y.pt').detach().numpy()
    # ode_data = np.concatenate((ode_T.reshape(1, -1), ode_Y), axis=0)
    # np.savetxt("./csvs/ode_data.csv", ode_data, delimiter=",")

    # acc_t, acc_data = read_data(f'./datas/lorenz_25.csv')
    # start = 40 * 7 + 1
    # end = 40 * 10 + 1

    # window_list = []
    # nrmse_ad_list = []
    # nrmse_rc_list = []

    # exs = [341, 342, 343, 344, 345]
    # windows = [0.1, 0.25, 0.5, 0.75, 1]
    # windows_mark = ['01', '025', '05', '075', '1']
    # alpha = 0.1
    # for ex, window, window_mark in zip(exs, windows, windows_mark):
    #     predict_Y = torch.load(f'./datas/predict_Y_{ex}.pt').T.detach().numpy()
    #     ad_T = torch.load(f'./datas/T_{ex}.pt').detach().numpy()
    #     ad_Y = torch.load(f'./datas/ad_Y_{ex}.pt').T.detach().numpy()
    #     nrmse_rc = nrmse_std(predict_Y[:, start:end], acc_data[:, start:end])
    #     nrmse_ad = nrmse_std(ad_Y[:, start:end], acc_data[:, start:end])

    #     window_list.append(window)
    #     nrmse_ad_list.append(nrmse_ad)
    #     nrmse_rc_list.append(nrmse_rc)

    #     ad_data = np.concatenate((ad_T.reshape(1, -1), ad_Y), axis=0)
    #     np.savetxt(f"./csvs/choose_param/alpha01/ad_data_{window_mark}_01.csv", ad_data, delimiter=",")

    # nrmse_data = np.array([window_list, nrmse_ad_list, nrmse_rc_list])
    # np.savetxt(f"./csvs/choose_param/alpha01/nrmse_01.csv", nrmse_data, delimiter=",")

    # window_list = []
    # nrmse_ad_list = []
    # nrmse_rc_list = []

    # exs = [351, 352, 353, 354, 355]
    # windows = [0.1, 0.25, 0.5, 0.75, 1]
    # windows_mark = ['01', '025', '05', '075', '1']
    # alpha = 0.5
    # for ex, window, window_mark in zip(exs, windows, windows_mark):
    #     predict_Y = torch.load(f'./datas/predict_Y_{ex}.pt').T.detach().numpy()
    #     ad_T = torch.load(f'./datas/T_{ex}.pt').detach().numpy()
    #     ad_Y = torch.load(f'./datas/ad_Y_{ex}.pt').T.detach().numpy()
    #     nrmse_rc = nrmse_std(predict_Y[:, start:end], acc_data[:, start:end])
    #     nrmse_ad = nrmse_std(ad_Y[:, start:end], acc_data[:, start:end])

    #     window_list.append(window)
    #     nrmse_ad_list.append(nrmse_ad)
    #     nrmse_rc_list.append(nrmse_rc)

    #     ad_data = np.concatenate((ad_T.reshape(1, -1), ad_Y), axis=0)
    #     np.savetxt(f"./csvs/choose_param/alpha05/ad_data_{window_mark}_05.csv", ad_data, delimiter=",")

    # nrmse_data = np.array([window_list, nrmse_ad_list, nrmse_rc_list])
    # np.savetxt(f"./csvs/choose_param/alpha05/nrmse_05.csv", nrmse_data, delimiter=",")

    # window_list = []
    # nrmse_ad_list = []
    # nrmse_rc_list = []
    
    # exs = [361, 362, 363, 364, 365]
    # windows = [0.1, 0.25, 0.5, 0.75, 1]
    # windows_mark = ['01', '025', '05', '075', '1']
    # alpha = 0.9
    # for ex, window, window_mark in zip(exs, windows, windows_mark):
    #     predict_Y = torch.load(f'./datas/predict_Y_{ex}.pt').T.detach().numpy()
    #     ad_T = torch.load(f'./datas/T_{ex}.pt').detach().numpy()
    #     ad_Y = torch.load(f'./datas/ad_Y_{ex}.pt').T.detach().numpy()
    #     nrmse_rc = nrmse_std(predict_Y[:, start:end], acc_data[:, start:end])
    #     nrmse_ad = nrmse_std(ad_Y[:, start:end], acc_data[:, start:end])

    #     window_list.append(window)
    #     nrmse_ad_list.append(nrmse_ad)
    #     nrmse_rc_list.append(nrmse_rc)

    #     ad_data = np.concatenate((ad_T.reshape(1, -1), ad_Y), axis=0)
    #     np.savetxt(f"./csvs/choose_param/alpha09/ad_data_{window_mark}_09.csv", ad_data, delimiter=",")

    # nrmse_data = np.array([window_list, nrmse_ad_list, nrmse_rc_list])
    # np.savetxt(f"./csvs/choose_param/alpha09/nrmse_09.csv", nrmse_data, delimiter=",")

    acc_t, acc_data = read_data(f'./datas/lorenz_25.csv')
    start = 40 * 10 + 1
    end = 40 * 18 + 1

    window_list = []
    nrmse_ad_list = []
    nrmse_rc_list = []

    exs = [251, 252, 253, 254, 255]
    windows = [0.1, 0.25, 0.5, 0.75, 1]
    windows_mark = ['01', '025', '05', '075', '1']
    alpha = 0.1
    for ex, window, window_mark in zip(exs, windows, windows_mark):
        predict_Y = torch.load(f'./datas/predict_Y_{ex}.pt').T.detach().numpy()
        ad_T = torch.load(f'./datas/T_{ex}.pt').detach().numpy()
        ad_Y = torch.load(f'./datas/ad_Y_{ex}.pt').T.detach().numpy()
        nrmse_rc = nrmse_std(predict_Y[:, start:end], acc_data[:, start:end])
        nrmse_ad = nrmse_std(ad_Y[:, start:end], acc_data[:, start:end])

        window_list.append(window)
        nrmse_ad_list.append(nrmse_ad)
        nrmse_rc_list.append(nrmse_rc)

        ad_data = np.concatenate((ad_T.reshape(1, -1), ad_Y), axis=0)
        np.savetxt(f"./csvs/test_choose_param/alpha01/ad_data_{window_mark}_01.csv", ad_data, delimiter=",")

    nrmse_data = np.array([window_list, nrmse_ad_list, nrmse_rc_list])
    np.savetxt(f"./csvs/test_choose_param/alpha01/nrmse_01.csv", nrmse_data, delimiter=",")

    window_list = []
    nrmse_ad_list = []
    nrmse_rc_list = []

    exs = [261, 262, 263, 264, 265]
    windows = [0.1, 0.25, 0.5, 0.75, 1]
    windows_mark = ['01', '025', '05', '075', '1']
    alpha = 0.5
    for ex, window, window_mark in zip(exs, windows, windows_mark):
        predict_Y = torch.load(f'./datas/predict_Y_{ex}.pt').T.detach().numpy()
        ad_T = torch.load(f'./datas/T_{ex}.pt').detach().numpy()
        ad_Y = torch.load(f'./datas/ad_Y_{ex}.pt').T.detach().numpy()
        nrmse_rc = nrmse_std(predict_Y[:, start:end], acc_data[:, start:end])
        nrmse_ad = nrmse_std(ad_Y[:, start:end], acc_data[:, start:end])

        window_list.append(window)
        nrmse_ad_list.append(nrmse_ad)
        nrmse_rc_list.append(nrmse_rc)

        ad_data = np.concatenate((ad_T.reshape(1, -1), ad_Y), axis=0)
        np.savetxt(f"./csvs/test_choose_param/alpha05/ad_data_{window_mark}_05.csv", ad_data, delimiter=",")

    nrmse_data = np.array([window_list, nrmse_ad_list, nrmse_rc_list])
    np.savetxt(f"./csvs/test_choose_param/alpha05/nrmse_05.csv", nrmse_data, delimiter=",")

    window_list = []
    nrmse_ad_list = []
    nrmse_rc_list = []
    
    exs = [271, 272, 273, 274, 275]
    windows = [0.1, 0.25, 0.5, 0.75, 1]
    windows_mark = ['01', '025', '05', '075', '1']
    alpha = 0.9
    for ex, window, window_mark in zip(exs, windows, windows_mark):
        predict_Y = torch.load(f'./datas/predict_Y_{ex}.pt').T.detach().numpy()
        ad_T = torch.load(f'./datas/T_{ex}.pt').detach().numpy()
        ad_Y = torch.load(f'./datas/ad_Y_{ex}.pt').T.detach().numpy()
        nrmse_rc = nrmse_std(predict_Y[:, start:end], acc_data[:, start:end])
        nrmse_ad = nrmse_std(ad_Y[:, start:end], acc_data[:, start:end])

        window_list.append(window)
        nrmse_ad_list.append(nrmse_ad)
        nrmse_rc_list.append(nrmse_rc)

        ad_data = np.concatenate((ad_T.reshape(1, -1), ad_Y), axis=0)
        np.savetxt(f"./csvs/test_choose_param/alpha09/ad_data_{window_mark}_09.csv", ad_data, delimiter=",")

    nrmse_data = np.array([window_list, nrmse_ad_list, nrmse_rc_list])
    np.savetxt(f"./csvs/test_choose_param/alpha09/nrmse_09.csv", nrmse_data, delimiter=",")


if __name__ == '__main__':
    # init_data(25, 0.025)
    data_to_csv()
    # t, data = read_data(f'./datas/lorenz_{25}.csv')
    # print(data)
    # t, data = init_data_noise(40)
    # np.savetxt('./datas/lorenz_noise_snr40.csv', (t, data), delimiter=',')

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
