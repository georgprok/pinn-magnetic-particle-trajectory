import torch


def gradients(y, x):
    """
    dy/dx trough autograd.
    """
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
    )[0]


def trajectory_and_velocity(model, t):
    """
    return:
    x, y, vx, vy
    """
    pred = model(t)

    x = pred[:, 0:1]
    y = pred[:, 1:2]

    vx = gradients(x, t)
    vy = gradients(y, t)

    return x, y, vx, vy


def lagrangian_density(x, y, vx, vy, q, m, Bz):
    """
    Lagrangian:
        L = (m/2)(vx^2 + vy^2) + (q Bz / 2)(x vy - y vx)
    """
    kinetic = 0.5 * m * (vx**2 + vy**2)
    magnetic = 0.5 * q * Bz * (x * vy - y * vx)

    return kinetic + magnetic


def euler_lagrange_residual(model, t, q, m, Bz):
    """
    Remains of the Euler–Lagrange equations:
        d/dt(∂L/∂vx) - ∂L/∂x = 0
        d/dt(∂L/∂vy) - ∂L/∂y = 0
    """
    x, y, vx, vy = trajectory_and_velocity(model, t)

    L = lagrangian_density(x, y, vx, vy, q=q, m=m, Bz=Bz)

    dL_dx = torch.autograd.grad(
        L, x, grad_outputs=torch.ones_like(L), create_graph=True, retain_graph=True
    )[0]

    dL_dy = torch.autograd.grad(
        L, y, grad_outputs=torch.ones_like(L), create_graph=True, retain_graph=True
    )[0]

    dL_dvx = torch.autograd.grad(
        L, vx, grad_outputs=torch.ones_like(L), create_graph=True, retain_graph=True
    )[0]

    dL_dvy = torch.autograd.grad(
        L, vy, grad_outputs=torch.ones_like(L), create_graph=True, retain_graph=True
    )[0]

    dt_dL_dvx = gradients(dL_dvx, t)
    dt_dL_dvy = gradients(dL_dvy, t)

    res_x = dt_dL_dvx - dL_dx
    res_y = dt_dL_dvy - dL_dy

    return res_x, res_y
