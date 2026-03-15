import torch
from physics import physics_residuals


def boundary_loss(model, t0, tT, A, B):
    """
    Штраф за граничные условия:
    r(0)=A, r(T)=B
    """
    pred0 = model(t0)
    predT = model(tT)

    loss_start = torch.mean((pred0 - A) ** 2)
    loss_end = torch.mean((predT - B) ** 2)

    return loss_start + loss_end


def physics_loss(model, t_collocation, q, m, bz):
    """
    Штраф за невязку уравнений движения.
    """
    res_x, res_y = physics_residuals(model, t_collocation, q, m, bz)
    return torch.mean(res_x**2) + torch.mean(res_y**2)


def train_pinn(
    model,
    optimizer,
    epochs,
    n_collocation,
    T,
    A,
    B,
    q,
    m,
    bz,
    bc_weight=100.0,
    phys_weight=1.0,
    device="cpu",
):
    """
    Обучение PINN.
    """
    history = []

    # Точки для граничных условий
    t0 = torch.tensor([[0.0]], dtype=torch.float32, device=device, requires_grad=True)
    tT = torch.tensor([[T]], dtype=torch.float32, device=device, requires_grad=True)

    A_tensor = torch.tensor([A], dtype=torch.float32, device=device)
    B_tensor = torch.tensor([B], dtype=torch.float32, device=device)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # Коллокационные точки внутри [0, T]
        t_col = torch.rand(n_collocation, 1, device=device) * T
        t_col.requires_grad_(True)

        loss_bc = boundary_loss(model, t0, tT, A_tensor, B_tensor)
        loss_phys = physics_loss(model, t_col, q=q, m=m, bz=bz)

        loss = bc_weight * loss_bc + phys_weight * loss_phys
        loss.backward()
        optimizer.step()

        history.append(
            {
                "epoch": epoch,
                "loss": loss.item(),
                "loss_bc": loss_bc.item(),
                "loss_phys": loss_phys.item(),
            }
        )

        if epoch % 500 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:5d} | "
                f"Loss = {loss.item():.6f} | "
                f"BC = {loss_bc.item():.6f} | "
                f"Phys = {loss_phys.item():.6f}"
            )

    return history
