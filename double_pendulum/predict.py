import torch
import numpy as np
import math
from torch.autograd import Variable as V
import matplotlib.pyplot as plt


def candidates(delay_t):

    cad = torch.cat(delay_t, axis=1)
    num_Lin = cad.shape[1]

    
    # for i in range(num_Lin):
    #     cad = torch.cat((cad, (np.sin(cad[:, i])).reshape(-1, 1)), axis=1)
    #     cad = torch.cat((cad, (np.cos(cad[:, i])).reshape(-1, 1)), axis=1)

    for i in range(num_Lin):
        for j in range(i, num_Lin):
            cad = torch.cat((cad, (cad[:, i] * cad[:, j]).reshape(-1, 1)), axis=1)
            # cad = torch.cat((cad, (np.sin(cad[:, i] - cad[:, j])).reshape(-1, 1)), axis=1)
            # cad = torch.cat((cad, (np.cos(cad[:, i] - cad[:, j])).reshape(-1, 1)), axis=1)

    for i in range(num_Lin):
        for j in range(i, num_Lin):
            for k in range(j, num_Lin):
                cad = torch.cat((cad, (cad[:, i] * cad[:, j] * cad[:, k]).reshape(-1, 1)), axis=1)
    
    Con = torch.ones((cad.shape[0], 1))

    cad = torch.cat((cad, Con), axis=1)

    return cad

def get_theta(y_train, num_d):
    
    train_num = len(y_train)
    X = y_train[num_d:, :]
    delay_t = [y_train[num_d-i:train_num-i, :] for i in range(num_d, 0, -1)]
    cad = candidates(delay_t)

    ridge_param = 1e-7
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
        predict_data = torch.cat((predict_data, X), axis=0)
        pre_t = torch.cat((pre_t, torch.tensor([[t]])), axis=0)

    return pre_t, predict_data
    