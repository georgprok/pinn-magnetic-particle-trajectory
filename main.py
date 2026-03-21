import os
import torch

from config import Config
from models.mlp import MLP
from training.trainer import Trainer
from shooting.solver import solve_shooting
from utils.plotting import (
    evaluate,
    plot_trajectory,
    plot_loss,
    plot_shooting_loss,
    plot_comparison,
)
from utils.seed import set_seed


def estimate_initial_velocity(model, device):
    """
    Extracts the PINN initial velocity from derivatives at t=0.
    """
    t0 = torch.tensor([[0.0]], dtype=torch.float32, device=device, requires_grad=True)
    pred0 = model(t0)

    x0 = pred0[:, 0:1]
    y0 = pred0[:, 1:2]

    vx0 = torch.autograd.grad(
        x0, t0, grad_outputs=torch.ones_like(x0), create_graph=False, retain_graph=True
    )[0].item()

    vy0 = torch.autograd.grad(
        y0, t0, grad_outputs=torch.ones_like(y0), create_graph=False
    )[0].item()

    return vx0, vy0


def main():
    cfg = Config()

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = device

    print(f"device: {device}")
    print(f"physics_mode: {cfg.physics_mode}")

    os.makedirs("results", exist_ok=True)

    # -----------------------------
    # 1. PINN training
    # -----------------------------
    model = MLP(hidden_dim=cfg.hidden_dim, hidden_layers=cfg.hidden_layers).to(device)

    trainer = Trainer(model, cfg)
    history = trainer.train()

    x_pinn, y_pinn = evaluate(model, cfg.T, device)
    plot_trajectory(x_pinn, y_pinn, cfg.A, cfg.B)
    plot_loss(history)

    torch.save(model.state_dict(), "results/model.pt")

    vx0_pinn, vy0_pinn = estimate_initial_velocity(model, device)

    print("\nPINN initial velocity:")
    print(f"vx0 = {vx0_pinn:.6f}, vy0 = {vy0_pinn:.6f}")

    # -----------------------------
    # 2. Shooting method
    # -----------------------------
    shooting_result = solve_shooting(cfg, device)

    x_shoot = shooting_result["x"]
    y_shoot = shooting_result["y"]
    plot_shooting_loss(shooting_result["history"])

    print("\nShooting initial velocity:")
    print(f"vx0 = {shooting_result['vx0']:.6f}, " f"vy0 = {shooting_result['vy0']:.6f}")

    # -----------------------------
    # 3. Comparison
    # -----------------------------
    plot_comparison(x_pinn, y_pinn, x_shoot, y_shoot, cfg.A, cfg.B)

    pinn_end_error = (
        (x_pinn[-1] - cfg.B[0]) ** 2 + (y_pinn[-1] - cfg.B[1]) ** 2
    ) ** 0.5
    shoot_end_error = (
        (x_shoot[-1] - cfg.B[0]) ** 2 + (y_shoot[-1] - cfg.B[1]) ** 2
    ) ** 0.5

    print("\nFinal endpoint errors:")
    print(f"PINN     error = {pinn_end_error:.6e}")
    print(f"Shooting error = {shoot_end_error:.6e}")

    with open("results/comparison.txt", "w", encoding="utf-8") as f:
        f.write("PINN vs Shooting comparison\n")
        f.write("=" * 80 + "\n")
        f.write(f"PINN initial velocity: vx0 = {vx0_pinn:.6f}, vy0 = {vy0_pinn:.6f}\n")
        f.write(
            f"Shooting initial velocity: "
            f"vx0 = {shooting_result['vx0']:.6f}, "
            f"vy0 = {shooting_result['vy0']:.6f}\n"
        )
        f.write(f"PINN endpoint error: {pinn_end_error:.6e}\n")
        f.write(f"Shooting endpoint error: {shoot_end_error:.6e}\n")

    print("\nResults saved to results/")


if __name__ == "__main__":
    main()
