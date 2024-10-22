import torch
import numpy as np
import math
from torch.autograd import Variable as V
import matplotlib.pyplot as plt


def candidates(delay_t):

    cad = torch.cat(delay_t, dim=1)
    con = torch.ones((cad.shape[0], 1))

    th_list = []
    sin_th_list = []
    cos_th_list = []
    w_list = []
    for i in range(0, cad.shape[1], 4):
        th_list.append(cad[:, i].reshape(-1, 1))
        th_list.append(cad[:, i + 1].reshape(-1, 1))
        sin_th_list.append(torch.sin(cad[:, i].reshape(-1, 1)))
        sin_th_list.append(torch.sin(cad[:, i + 1].reshape(-1, 1)))
        # sin_th_list.append(torch.sin((cad[:, i] + cad[:, i + 1]).reshape(-1, 1)))
        # sin_th_list.append(torch.sin((cad[:, i] - cad[:, i + 1]).reshape(-1, 1)))
        cos_th_list.append(torch.cos(cad[:, i].reshape(-1, 1)))
        cos_th_list.append(torch.cos(cad[:, i + 1].reshape(-1, 1)))
        # cos_th_list.append(torch.cos((cad[:, i] + cad[:, i + 1]).reshape(-1, 1)))
        # cos_th_list.append(torch.cos((cad[:, i] - cad[:, i + 1]).reshape(-1, 1)))
        w_list.append(cad[:, i + 2].reshape(-1, 1))
        w_list.append(cad[:, i + 3].reshape(-1, 1))

    # cand = torch.cat(cand_list, dim=1)

    mix_list = []

    # for i in range(len(w_list)):
    #     for j in range(i, len(w_list)):
    #         mix_list.append((w_list[i] * w_list[j]).reshape(-1, 1))
    
    th = torch.cat(th_list, dim=1)
    sin_th = torch.cat(sin_th_list, dim=1)
    cos_th = torch.cat(cos_th_list, dim=1)
    w = torch.cat(w_list, dim=1)
    # mix = torch.cat(mix_list, dim=1)


    cand = torch.cat((con, cad, cad**2, cad**3, torch.sin(cad), torch.cos(cad)), dim=1)
    # cand = torch.cat((con, th, sin_th, cos_th, w), dim=1)
    # cand = torch.cat((con, cad, torch.sin(cad), torch.cos(cad)), dim=1)
    # mix_list = []

    # for i in range(cand.shape[1]):
    #     for j in range(i, cand.shape[1]):
    #         mix_list.append((cand[:, i] * cand[:, j]).reshape(-1, 1))
    #         for k in range(j, cand.shape[1]):
    #             for m in range(k, cand.shape[1]):
    #                 mix_list.append((cand[:, i] * cand[:, j] * cand[:, k] * cand[:, m]).reshape(-1, 1))
    # mix = torch.cat(mix_list, dim=1)
    # return mix
    return cand

def get_theta(y_train, num_d, ridge_param):
    
    train_num = len(y_train)
    X = y_train[num_d:, :]
    delay_t = [y_train[num_d-i:train_num-i, :] for i in range(num_d, 0, -1)]
    cad = candidates(delay_t)

    # ridge_param = 9e-9
    theta = torch.linalg.pinv(cad.T @ cad + ridge_param * torch.eye(cad.shape[1])) @ cad.T @ X

    print(theta)
    return theta


def predict(theta, num_d, dt, pre_t, predict_data, pre_start=10.025, pre_step=40):

    for i in range(0, pre_step):
        t = pre_start + dt * i
        
        delay_t = []
        for j in range(num_d, 0, -1):
            delay_t.append(predict_data[-j, :].reshape(1, -1))

        cad = candidates(delay_t)
        X = cad @ theta
        predict_data = torch.cat((predict_data, X), dim=0)
        pre_t = torch.cat((pre_t, torch.tensor([[t]])), dim=0)

    return pre_t, predict_data
    