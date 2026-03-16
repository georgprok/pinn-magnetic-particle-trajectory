import os
import torch

from config import Config
from models.mlp import MLP
from training.trainer import Trainer
from utils.plotting import evaluate, plot_trajectory, plot_loss
from utils.seed import set_seed


def main():
    cfg = Config()

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = device

    print(f"device: {device}")

    os.makedirs("results", exist_ok=True)

    model = MLP(hidden_dim=cfg.hidden_dim, hidden_layers=cfg.hidden_layers).to(device)

    trainer = Trainer(model, cfg)
    history = trainer.train()

    x, y = evaluate(model, cfg.T, device)
    plot_trajectory(x, y, cfg.A, cfg.B)
    plot_loss(history)

    torch.save(model.state_dict(), "results/model.pt")

    print("Results saved to results/")


if __name__ == "__main__":
    main()
