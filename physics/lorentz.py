import torch


def gradients(y, x):

    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
    )[0]


def trajectory_derivatives(model, t):

    pred = model(t)

    x = pred[:, 0:1]
    y = pred[:, 1:2]
    z = pred[:, 2:3]

    vx = gradients(x, t)
    vy = gradients(y, t)
    vz = gradients(z, t)

    ax = gradients(vx, t)
    ay = gradients(vy, t)
    az = gradients(vz, t)

    return x, y, z, vx, vy, vz, ax, ay, az


def physics_residual(model, t, q, m, Bz):

    x, y, z, vx, vy, vz, ax, ay, az = trajectory_derivatives(model, t)

    res_x = m * ax - q * Bz * vy
    res_y = m * ay + q * Bz * vx
    res_z = m * az

    return res_x, res_y, res_z
