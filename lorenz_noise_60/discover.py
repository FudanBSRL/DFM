import torch
from jacobi import jacobi, jacobi_t, jacobi_diff

def num_candidates(num_Lin):
    cnt = num_Lin + 1
    # cnt = num_Lin
    for i in range(num_Lin):
        for j in range(i, num_Lin):
            cnt += 1

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
        for j in range(i, num_Lin):
            cand = torch.cat((cand, (u[:, i] * u[:, j]).reshape(-1, 1)), axis=1)

    for i in range(num_Lin):
        for j in range(i, num_Lin):
            for k in range(j, num_Lin):
                cand = torch.cat((cand, (u[:, i] * u[:, j] * u[:, k]).reshape(-1, 1)), axis=1)
    
    return cand


def candidates_t(u, u_t):
    Con = torch.zeros((u.shape[0], 1))
    cand = torch.cat((Con, u_t), axis=1)
    # cand = u_t
    num_Lin = u.shape[1]

    for i in range(num_Lin):
        for j in range(i, num_Lin):
            cand = torch.cat((cand, (u_t[:, i] * u[:, j] + u[:, i] * u_t[:, j]).reshape(-1, 1)), axis=1)
    
    return cand


def discover_gpinn(net, T, ksi):
    u, u_t, u_tt = jacobi_t(net, T)

    cand = candidates(u)
    cand_t = candidates_t(u, u_t)
    
    loss_fn = torch.nn.MSELoss(reduction='mean')
    msed = loss_fn(u_t, cand @ (ksi))
    msed_t = loss_fn(u_tt, cand_t @ (ksi))
    # msed = torch.mean(torch.abs(u_t - cand @ (ksi)))
    # msed_t = torch.mean(torch.abs(u_tt - cand_t @ (ksi)))
    return msed, msed_t


def discover(net, T, ksi, Lambda):
    u, u_t = jacobi(net, T)

    # PSRL = torch.mean(1 / 2 * (u[:, 0] - u[:, 2]) ** 2)

    cand = candidates(u)
    # msed = torch.sum(torch.abs(u_t - cand @ ksi)) / len(T)
    msed = torch.mean(1/2 * (u_t - cand @ (Lambda * ksi)) ** 2)
    return msed, u, u_t


def init_ksi_data(net, T):
    u, u_t = jacobi(net, T)
    cand = candidates(u)

    return torch.linalg.pinv(cand.T @ cand) @ cand.T @ u_t

