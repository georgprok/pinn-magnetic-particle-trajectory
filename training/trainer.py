import os
import torch
from physics.lorentz import physics_residual


class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.cfg = config

        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        self.history = []

        os.makedirs("results", exist_ok=True)
        self.log_path = os.path.join("results", "train_log.txt")

        # очищаем лог в начале нового запуска
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("Training log\n")
            f.write("=" * 80 + "\n")

    def boundary_loss(self, t0, tT):
        device = self.cfg.device

        A = torch.tensor([self.cfg.A], dtype=torch.float32, device=device)
        B = torch.tensor([self.cfg.B], dtype=torch.float32, device=device)

        pred0 = self.model(t0)
        predT = self.model(tT)

        loss_start = torch.mean((pred0 - A) ** 2)
        loss_end = torch.mean((predT - B) ** 2)

        return loss_start + loss_end

    def physics_loss(self, t):
        res_x, res_y = physics_residual(
            self.model, t, self.cfg.q, self.cfg.m, self.cfg.Bz
        )

        return torch.mean(res_x**2) + torch.mean(res_y**2)

    def _log(self, message: str):
        print(message)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def train(self):
        device = self.cfg.device

        t0 = torch.tensor(
            [[0.0]], dtype=torch.float32, device=device, requires_grad=True
        )
        tT = torch.tensor(
            [[self.cfg.T]], dtype=torch.float32, device=device, requires_grad=True
        )

        for epoch in range(1, self.cfg.epochs + 1):
            self.optimizer.zero_grad()

            t = torch.rand(self.cfg.collocation_points, 1, device=device) * self.cfg.T
            t.requires_grad_(True)

            loss_bc = self.boundary_loss(t0, tT)
            loss_phys = self.physics_loss(t)

            loss = self.cfg.bc_weight * loss_bc + self.cfg.phys_weight * loss_phys

            loss.backward()
            self.optimizer.step()

            record = {
                "epoch": epoch,
                "loss": loss.item(),
                "loss_bc": loss_bc.item(),
                "loss_phys": loss_phys.item(),
            }
            self.history.append(record)

            if epoch == 1 or epoch % self.cfg.print_every == 0:
                msg = (
                    f"Epoch {epoch:5d} | "
                    f"Loss = {loss.item():.6f} | "
                    f"BC = {loss_bc.item():.6f} | "
                    f"Phys = {loss_phys.item():.6f}"
                )
                self._log(msg)

        return self.history
