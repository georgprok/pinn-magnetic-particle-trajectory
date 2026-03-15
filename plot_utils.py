import numpy as np
import matplotlib.pyplot as plt
import torch


def evaluate_trajectory(model, T, n_points=300, device="cpu"):
    """
    Вычисляет траекторию на равномерной сетке по времени.
    """
    t = torch.linspace(0.0, T, n_points, device=device).view(-1, 1)
    t.requires_grad_(True)

    with torch.no_grad():
        pred = model(t)

    x = pred[:, 0].detach().cpu().numpy()
    y = pred[:, 1].detach().cpu().numpy()
    t_np = t[:, 0].detach().cpu().numpy()

    return t_np, x, y


def plot_trajectory(x, y, A, B):
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, label="PINN trajectory")
    plt.scatter([A[0], B[0]], [A[1], B[1]], label="A, B", s=80)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2D trajectory in constant magnetic field")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_losses(history):
    epochs = [item["epoch"] for item in history]
    total = [item["loss"] for item in history]
    bc = [item["loss_bc"] for item in history]
    phys = [item["loss_phys"] for item in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, total, label="total loss")
    plt.plot(epochs, bc, label="boundary loss")
    plt.plot(epochs, phys, label="physics loss")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training losses")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
