import torch

def jacobi(net, t, device='cpu'):
    t.requires_grad = True
    u = net(t)
    u_t = torch.tensor([]).to(device)
    for i in range(u.shape[1]):
        p = torch.zeros(u.shape).to(device)
        p[:, i] = torch.ones(u.shape[0])
        temp = torch.autograd.grad(u, t, grad_outputs=p, create_graph=True, allow_unused=True)[0]
        u_t = torch.cat((u_t, temp), axis=1)
    return u, u_t


def jacobi_diff(net, t, device='cpu'):
    dt = 0.001
    u = net(t)
    u_before = net(t - dt)
    u_after = net(t + dt)
    u_t = (u_after - u_before) / (2 * dt)
    return u, u_t


def jacobi_t(net, t, device='cpu'):
    t.requires_grad = True
    u, u_t = jacobi(net, t)
    u_tt = torch.tensor([]).to(device)
    for i in range(u_t.shape[1]):
        p = torch.zeros(u_t.shape).to(device)
        p[:, i] = torch.ones(u_t.shape[0])
        temp = torch.autograd.grad(u_t, t, grad_outputs=p, create_graph=True, allow_unused=True)[0]
        u_tt = torch.cat((u_tt, temp), axis=1)
    return u, u_t, u_tt