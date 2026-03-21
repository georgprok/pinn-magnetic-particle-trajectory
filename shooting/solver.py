import os
import torch


def euler_integrate_trajectory(v0, config, device):
    """
    Integrates the particle trajectory using the Euler method.

    State:
        x, y, vx, vy

    Equations:
        dx/dt  = vx
        dy/dt  = vy
        dvx/dt = (q * Bz / m) * vy
        dvy/dt = -(q * Bz / m) * vx
    """
    q = config.q
    m = config.m
    Bz = config.Bz

    x0, y0, z0 = config.A
    T = config.T
    n_steps = config.shooting_steps
    dt = T / n_steps

    # initial state
    x = torch.tensor(x0, dtype=torch.float32, device=device)
    y = torch.tensor(y0, dtype=torch.float32, device=device)
    z = torch.tensor(z0, dtype=torch.float32, device=device)

    vx = v0[0]
    vy = v0[1]
    vz = v0[2]

    xs = [x]
    ys = [y]
    zs = [z]
    omega = q * Bz / m

    for _ in range(n_steps):
        ax = omega * vy
        ay = -omega * vx
        az = torch.tensor(0.0, dtype=torch.float32, device=device)

        x = x + dt * vx
        y = y + dt * vy
        z = z + dt * vz
        vx = vx + dt * ax
        vy = vy + dt * ay
        vz = vz + dt * az
        xs.append(x)
        ys.append(y)
        zs.append(z)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    zs = torch.stack(zs)
    return xs, ys, zs


def shooting_loss(v0, config, device):
    """
    Shooting method loss:
    distance between the trajectory endpoint and point B.
    """
    xs, ys, zs = euler_integrate_trajectory(v0, config, device)

    xT = xs[-1]
    yT = ys[-1]
    zT = zs[-1]
    Bx, By, Bz_target = config.B
    target = torch.tensor([Bx, By], dtype=torch.float32, device=device)
    pred = torch.stack([xT, yT])

    loss = torch.mean((pred - target) ** 2)
    return loss, xs, ys, zs


def solve_shooting(config, device, log_path="results/shooting_log.txt"):
    """
    Finds the initial velocity v0 = (vx0, vy0) via optimization.
    """
    os.makedirs("results", exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Shooting log\n")
        f.write("=" * 80 + "\n")

    # trainable shooting parameters: initial velocity
    v0 = torch.nn.Parameter(
        torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device=device)
    )

    optimizer = torch.optim.Adam([v0], lr=config.shooting_lr)

    history = []

    for epoch in range(1, config.shooting_epochs + 1):
        optimizer.zero_grad()

        loss, xs, ys, zs = shooting_loss(v0, config, device)
        loss.backward()
        optimizer.step()

        record = {
            "epoch": epoch,
            "loss": loss.item(),
            "vx0": v0[0].item(),
            "vy0": v0[1].item(),
            "vz0": v0[2].item(),
        }
        history.append(record)

        if epoch == 1 or epoch % config.shooting_print_every == 0:
            msg = (
                f"Shooting Epoch {epoch:5d} | "
                f"Loss = {loss.item():.6f} | "
                f"vx0 = {v0[0].item():.6f} | "
                f"vy0 = {v0[1].item():.6f} | "
                f"vz0 = {v0[2].item():.6f}"
            )
            print(msg)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    # final trajectory after optimization
    with torch.no_grad():
        _, xs, ys, zs = shooting_loss(v0, config, device)

    return {
        "vx0": v0[0].item(),
        "vy0": v0[1].item(),
        "vz0": v0[2].item(),
        "x": xs.detach().cpu().numpy(),
        "y": ys.detach().cpu().numpy(),
        "z": zs.detach().cpu().numpy(),
        "history": history,
    }
