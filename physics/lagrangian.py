import torch


def gradients(y, x):
    """
    dy/dx trough autograd.
    """
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
    )[0]


def safe_grad(outputs, inputs):
    """
    safe grad:
    if inputs doesn't participate in graph(outputs),
    return zero tenzor same form
    """
    grad = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]

    if grad is None:
        grad = torch.zeros_like(inputs)

    return grad


def trajectory_and_velocity(model, t):
    """
    return:
    x, y, vx, vy
    """
    pred = model(t)

    x = pred[:, 0:1]
    y = pred[:, 1:2]
    z = pred[:, 2:3]

    vx = gradients(x, t)
    vy = gradients(y, t)
    vz = gradients(z, t)

    return x, y, z, vx, vy, vz


def lagrangian_density(x, y, z, vx, vy, vz, q, m, Bz):
    """
    Lagrangian:
        L = (m/2)(vx^2 + vy^2) + (q Bz / 2)(x vy - y vx)
    """
    kinetic = 0.5 * m * (vx**2 + vy**2 + vz**2)
    magnetic = 0.5 * q * Bz * (x * vy - y * vx)

    return kinetic + magnetic


def euler_lagrange_residual(model, t, q, m, Bz):
    """
    Remains of the Euler–Lagrange equations:
        d/dt(∂L/∂vx) - ∂L/∂x = 0
        d/dt(∂L/∂vy) - ∂L/∂y = 0
    """
    x, y, z, vx, vy, vz = trajectory_and_velocity(model, t)

    L = lagrangian_density(x, y, z, vx, vy, vz, q=q, m=m, Bz=Bz)

    dL_dx = safe_grad(L, x)
    dL_dy = safe_grad(L, y)
    dL_dz = safe_grad(L, z)

    dL_dvx = safe_grad(L, vx)
    dL_dvy = safe_grad(L, vy)
    dL_dvz = safe_grad(L, vz)

    dt_dL_dvx = gradients(dL_dvx, t)
    dt_dL_dvy = gradients(dL_dvy, t)
    dt_dL_dvz = gradients(dL_dvz, t)

    res_x = dt_dL_dvx - dL_dx
    res_y = dt_dL_dvy - dL_dy
    res_z = dt_dL_dvz - dL_dz

    return res_x, res_y, res_z
