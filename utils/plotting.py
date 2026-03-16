import os
import torch
import matplotlib.pyplot as plt


def evaluate(model, T, device):
    t = torch.linspace(0.0, T, 300, device=device).view(-1, 1)

    with torch.no_grad():
        pred = model(t)

    x = pred[:, 0].detach().cpu().numpy()
    y = pred[:, 1].detach().cpu().numpy()

    return x, y


def plot_trajectory(x, y, A, B):
    plt.figure(figsize=(6, 6))

    plt.plot(x, y, label="PINN trajectory")
    plt.scatter([A[0], B[0]], [A[1], B[1]], s=80, label="A, B")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    os.makedirs("results", exist_ok=True)
    plt.tight_layout()
    plt.savefig("results/trajectory.png")
    plt.close()


def plot_loss(history):
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

    os.makedirs("results", exist_ok=True)
    plt.tight_layout()
    plt.savefig("results/loss.png")
    plt.close()


def plot_shooting_loss(history):
    epochs = [item["epoch"] for item in history]
    loss = [item["loss"] for item in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, label="shooting loss")

    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Shooting optimization loss")
    plt.grid(True)
    plt.legend()

    os.makedirs("results", exist_ok=True)
    plt.tight_layout()
    plt.savefig("results/shooting_loss.png")
    plt.close()


def plot_comparison(x_pinn, y_pinn, x_shoot, y_shoot, A, B):
    plt.figure(figsize=(6, 6))

    plt.plot(x_pinn, y_pinn, label="PINN")
    plt.plot(x_shoot, y_shoot, "--", label="Shooting")
    plt.scatter([A[0], B[0]], [A[1], B[1]], s=80, label="A, B")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("PINN vs Shooting")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    os.makedirs("results", exist_ok=True)
    plt.tight_layout()
    plt.savefig("results/trajectory_comparison.png")
    plt.close()
