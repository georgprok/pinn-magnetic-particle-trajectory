import os
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def evaluate(model, T, device):
    t = torch.linspace(0.0, T, 300, device=device).view(-1, 1)

    with torch.no_grad():
        pred = model(t)

    x = pred[:, 0].detach().cpu().numpy()
    y = pred[:, 1].detach().cpu().numpy()
    z = pred[:, 2].detach().cpu().numpy()

    return x, y, z


def plot_trajectory(x, y, z, A, B):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x, y, z, label="PINN trajectory")
    ax.scatter([A[0], B[0]], [A[1], B[1]], [A[2], B[2]], s=80, label="A, B")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("3D Trajectory")
    ax.legend()

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


def plot_comparison(x_pinn, y_pinn, z_pinn, x_shoot, y_shoot, z_shoot, A, B):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x_pinn, y_pinn, z_pinn, label="PINN")
    ax.plot(x_shoot, y_shoot, z_shoot, "--", label="Shooting")
    ax.scatter([A[0], B[0]], [A[1], B[1]], [A[2], B[2]], s=80, label="A, B")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("PINN vs Shooting (3D)")
    ax.legend()

    os.makedirs("results", exist_ok=True)
    plt.tight_layout()
    plt.savefig("results/trajectory_comparison.png")
    plt.close()
