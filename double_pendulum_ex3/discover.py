import torch
import numpy as np
from jacobi import jacobi, jacobi_t

def num_candidates(num_Lin):
    cnt = 1
    cnt += num_Lin
    cnt += 4

    num_Lin += 4

    for i in range(num_Lin):
        for j in range(i, num_Lin):
            cnt += 1
            if i < 4 and j < 4:
                cnt += 2
                if i != j:
                    cnt += 2

    for i in range(num_Lin):
        for j in range(i, num_Lin):
            for k in range(j, num_Lin):
                cnt += 1

    for i in range(num_Lin):
        for j in range(i, num_Lin):
            for k in range(j, num_Lin):
                for l in range(k, num_Lin):
                    cnt += 1

    return cnt


def candidates(u, device='cpu'):
    cand_list = []
    Con = torch.ones((u.shape[0], 1)).to(device)
    
    cand_list.append(Con)
    # cand = u
    base = torch.cat((u, torch.sin(u[:, 0:2]), torch.cos(u[:, 0:2])), dim=1)
    cand_list.append(base)
    num_Lin = base.shape[1]

    for i in range(num_Lin):
        for j in range(i, num_Lin):
            cand_list.append((base[:, i] * base[:, j]).reshape(-1, 1))
            if i < 4 and j < 4:
                cand_list.append(torch.sin(base[:, i] + base[:, j]).reshape(-1, 1))
                cand_list.append(torch.cos(base[:, i] + base[:, j]).reshape(-1, 1))
                if i != j:
                    cand_list.append(torch.sin(base[:, i] - base[:, j]).reshape(-1, 1))
                    cand_list.append(torch.cos(base[:, i] - base[:, j]).reshape(-1, 1))


    for i in range(num_Lin):
        for j in range(i, num_Lin):
            for k in range(j, num_Lin):
                cand_list.append((base[:, i] * base[:, j] * base[:, k]).reshape(-1, 1))

    for i in range(num_Lin):
        for j in range(i, num_Lin):
            for k in range(j, num_Lin):
                for l in range(k, num_Lin):
                    cand_list.append((base[:, i] * base[:, j] * base[:, k] * base[:, l]).reshape(-1, 1))
    
    cand = torch.cat(cand_list, dim=1)
    return cand


def discover(net, T, ksi, Lambda, train=False, device='cpu'):
    u, u_t = jacobi(net, T, device=device)

    cand = candidates(u, device=device)

    if not train:
        Lambda = torch.zeros(size=ksi.shape).to(device)
        Lambda[ksi != 0] = 1
        msed = torch.mean((u_t - cand @ (Lambda * ksi)) ** 2)
    else:
        msed = torch.mean((u_t - cand @ ksi) ** 2)
    return msed, u, u_t, cand


def init_ksi_data(net, T):
    u, u_t = jacobi(net, T)
    cand = candidates(u)

    ridge_param = 0
    return torch.linalg.pinv(cand.T @ cand + ridge_param * torch.eye(cand.shape[1])) @ cand.T @ u_t

