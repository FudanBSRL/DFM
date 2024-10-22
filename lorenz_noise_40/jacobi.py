import torch
from torch.autograd import grad

def jacobi(net, t, device='cpu'):
    t.requires_grad = True
    u = net(t)

    u_t_list = []
    for i in range(u.shape[1]):
        temp = grad(u[:, i], t, grad_outputs=torch.ones_like(u[:, i]), create_graph=True, allow_unused=True)[0]
        u_t_list.append(temp)
    u_t = torch.cat(u_t_list, axis=1)
    return u, u_t


def jacobi_t(net, t, device='cpu'):
    t.requires_grad = True
    u, u_t = jacobi(net, t)

    u_tt_list = []
    for i in range(u_t.shape[1]):
        temp = grad(u_t[:, i], t, grad_outputs=torch.ones_like(u_t[:, i]), create_graph=True, allow_unused=True)[0]
        u_tt_list.append(temp)
    u_tt = torch.cat(u_tt_list, axis=1)
    return u, u_t, u_tt