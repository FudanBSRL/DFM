import torch
from jacobi import jacobi, jacobi_t

def PDE_gpinn(net, t, params):
    t.requires_grad = True
    u, u_t, u_tt = jacobi_t(net, t)

    pde1 = u_t[:, 0] - params[0] * (u[:, 1] - u[:, 0])
    pde2 = u_t[:, 1] - u[:, 0] * (params[2] - u[:, 2]) + u[:, 1]
    pde3 = u_t[:, 2] - u[:, 0] * u[:, 1] + params[1] * u[:, 2]

    pde1_t = u_tt[:, 0] - params[0] * (u_t[:, 1] - u_t[:, 0])
    pde2_t = u_tt[:, 1] - u_t[:, 0] * params[2] + u_t[:, 0] * u[:, 2] + u[:, 0] * u_t[:, 2] + u_t[:, 1]
    pde3_t = u_tt[:, 2] - u_t[:, 0] * u[:, 1] - u[:, 0] * u_t[:, 1] + params[1] * u_t[:, 2]
    return pde1, pde2, pde3, pde1_t, pde2_t, pde3_t, u, u_t
    # return pde1, pde2, pde3


def PDE(net, t, params):
    t.requires_grad = True
    u, u_t = jacobi(net, t)

    pde1 = u_t[:, 0] - params[0] * (u[:, 1] - u[:, 0]) 
    pde2 = u_t[:, 1] - u[:, 0] * (params[2] - u[:, 2]) + u[:, 1]
    pde3 = u_t[:, 2] - u[:, 0] * u[:, 1] + params[1] * u[:, 2]
    return pde1, pde2, pde3, u, u_t


def pde_loss_gpinn(net, T, params):
    p1, p2, p3, p1_t, p2_t, p3_t, u, u_t = PDE_gpinn(net, T, params)
    mse_ac_p1 = torch.mean(1/2 * p1**2)
    mse_ac_p2 = torch.mean(1/2 * p2**2)
    mse_ac_p3 = torch.mean(1/2 * p3**2)
    mse_ac_p = (mse_ac_p1 + mse_ac_p2 + mse_ac_p3) / 3

    mse_ac_p1_t = torch.mean(1/2 * p1_t**2)
    mse_ac_p2_t = torch.mean(1/2 * p2_t**2)
    mse_ac_p3_t = torch.mean(1/2 * p3_t**2)
    mse_ac_p_t = (mse_ac_p1_t + mse_ac_p2_t + mse_ac_p3_t) / 3

    return mse_ac_p, mse_ac_p_t, u, u_t


def pde_loss(net, T, params):
    p1, p2, p3, u, u_t = PDE(net, T, params)
    mse_ac_p1 = torch.mean(1/2 * p1**2)
    mse_ac_p2 = torch.mean(1/2 * p2**2)
    mse_ac_p3 = torch.mean(1/2 * p3**2)
    mse_ac_p = (mse_ac_p1 + mse_ac_p2 + mse_ac_p3) / 3

    return mse_ac_p, u, u_t