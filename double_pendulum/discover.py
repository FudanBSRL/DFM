import torch
import numpy as np
from jacobi import jacobi, jacobi_t

def num_candidates(num_Lin):
    cnt = num_Lin + 1
    # cnt = num_Lin
    for i in range(num_Lin):
        cnt += 2

    for i in range(num_Lin):
        for j in range(i, num_Lin):
            cnt += 3
            if i != j :
                cnt += 2

    for i in range(num_Lin):
        for j in range(i, num_Lin):
            for k in range(j, num_Lin):
                cnt += 1

    return cnt


def candidates(u):
    Con = torch.ones((u.shape[0], 1))
    cand = torch.cat((Con, u), axis=1)
    # cand = u
    num_Lin = u.shape[1]

    for i in range(num_Lin):
        cand = torch.cat((cand, (torch.sin(u[:, i])).reshape(-1, 1)), axis=1)
        cand = torch.cat((cand, (torch.cos(u[:, i])).reshape(-1, 1)), axis=1)

    for i in range(num_Lin):
        for j in range(i, num_Lin):
            cand = torch.cat((cand, (u[:, i] * u[:, j]).reshape(-1, 1)), axis=1)
            cand = torch.cat((cand, (torch.sin(u[:, i] + u[:, j])).reshape(-1, 1)), axis=1)
            cand = torch.cat((cand, (torch.cos(u[:, i] + u[:, j])).reshape(-1, 1)), axis=1)
            if i != j :
                cand = torch.cat((cand, (torch.sin(u[:, i] - u[:, j])).reshape(-1, 1)), axis=1)
                cand = torch.cat((cand, (torch.cos(u[:, i] - u[:, j])).reshape(-1, 1)), axis=1)

    for i in range(num_Lin):
        for j in range(i, num_Lin):
            for k in range(j, num_Lin):
                cand = torch.cat((cand, (u[:, i] * u[:, j] * u[:, k]).reshape(-1, 1)), axis=1)
    
    # cand = cand / (torch.sqrt(torch.sum(cand ** 2)))
    return cand


def candidates_t(u, u_t):
    Con = torch.zeros((u.shape[0], 1))
    cand = torch.cat((Con, u_t), axis=1)
    # cand = u_t
    num_Lin = u.shape[1]

    for i in range(num_Lin):
        for j in range(i, num_Lin):
            cand = torch.cat((cand, (u_t[:, i] * u[:, j] + u[:, i] * u_t[:, j]).reshape(-1, 1)), axis=1)

    for i in range(num_Lin):
        for j in range(i, num_Lin):
            for k in range(j, num_Lin):
                cand = torch.cat((cand, (u_t[:, i] * u[:, j] * u[:, k] + u[:, i] * u_t[:, j] * u[:, k] + u[:, i] * u[:, j] * u_t[:, k]).reshape(-1, 1)), axis=1)
    
    # cand = cand / (torch.sqrt(torch.sum(cand ** 2)))
    return cand


def discover_gpinn(net, T, ksi, Lambda):
    u, u_t, u_tt = jacobi_t(net, T)

    cand = candidates(u)
    cand_t = candidates_t(u, u_t)
    
    msed = torch.mean(1/2 * (u_t - cand @ (Lambda * ksi)) ** 2)
    msed_t = torch.mean(1/2 * (u_tt - cand_t @ (Lambda * ksi)) ** 2)
    # msed = torch.mean(torch.abs(u_t - cand @ (ksi)))
    # msed_t = torch.mean(torch.abs(u_tt - cand_t @ (ksi)))
    return msed, msed_t, u


def discover(net, T, ksi, Lambda):
    u, u_t = jacobi(net, T)

    cand = candidates(u)

    Lambda = torch.zeros(size=ksi.shape)
    Lambda[ksi != 0] = 1

    msed = torch.mean(1/2 * (u_t - cand @ (ksi * Lambda)) ** 2)
    return msed, u, u_t


def init_ksi_data(net, T):
    u, u_t = jacobi(net, T)
    cand = candidates(u)

    ridge_param = 0
    return torch.linalg.pinv(cand.T @ cand + ridge_param * torch.eye(cand.shape[1])) @ cand.T @ u_t

