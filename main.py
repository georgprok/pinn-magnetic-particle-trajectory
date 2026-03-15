import math
import torch

from model import MLP
from train import train_pinn
from plot_utils import evaluate_trajectory, plot_trajectory, plot_losses


def main():
    # -----------------------------
    # Базовые параметры задачи
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Физика
    q = 1.0
    m = 1.0
    bz = 1.0

    # Время движения
    T = math.pi / 2.0

    # Граничные точки
    A = (0.0, 0.0)
    B = (1.0, 1.0)

    # -----------------------------
    # Параметры модели и обучения
    # -----------------------------
    model = MLP(in_dim=1, hidden_dim=64, hidden_layers=3, out_dim=2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 8000
    n_collocation = 256

    # -----------------------------
    # Обучение
    # -----------------------------
    history = train_pinn(
        model=model,
        optimizer=optimizer,
        epochs=epochs,
        n_collocation=n_collocation,
        T=T,
        A=A,
        B=B,
        q=q,
        m=m,
        bz=bz,
        bc_weight=100.0,
        phys_weight=1.0,
        device=device,
    )

    # -----------------------------
    # Оценка и визуализация
    # -----------------------------
    t, x, y = evaluate_trajectory(model, T=T, n_points=300, device=device)

    print("\nPredicted endpoints:")
    print(f"Start approx: ({x[0]:.4f}, {y[0]:.4f})")
    print(f"End   approx: ({x[-1]:.4f}, {y[-1]:.4f})")

    # Начальная скорость как производная траектории в t=0
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

    print(f"Estimated initial velocity: vx(0)={vx0:.4f}, vy(0)={vy0:.4f}")

    plot_trajectory(x, y, A, B)
    plot_losses(history)


if __name__ == "__main__":
    main()
