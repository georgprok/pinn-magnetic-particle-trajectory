import torch


def gradients(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    dy/dx через autograd.
    """
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
    )[0]


def trajectory_derivatives(model, t: torch.Tensor):
    """
    Возвращает:
    x, y, vx, vy, ax, ay
    """
    pred = model(t)  # shape: [N, 2]
    x = pred[:, 0:1]
    y = pred[:, 1:2]

    vx = gradients(x, t)
    vy = gradients(y, t)

    ax = gradients(vx, t)
    ay = gradients(vy, t)

    return x, y, vx, vy, ax, ay


def physics_residuals(model, t: torch.Tensor, q: float, m: float, bz: float):
    """
    Уравнения движения в постоянном магнитном поле B=(0,0,Bz):
        m * x'' = q * Bz * y'
        m * y'' = -q * Bz * x'
    """
    x, y, vx, vy, ax, ay = trajectory_derivatives(model, t)

    res_x = m * ax - q * bz * vy
    res_y = m * ay + q * bz * vx

    return res_x, res_y
