import math


class Config:
    # устройство
    device = "cuda"

    # seed
    seed = 42

    # физика
    q = 1.0
    m = 1.0
    Bz = 1.0

    # граничные условия
    A = (0.0, 0.0)
    B = (1.0, 1.0)

    # время
    T = math.pi / 2

    # сеть
    hidden_dim = 64
    hidden_layers = 3

    # обучение PINN
    lr = 1e-3
    epochs = 5000
    collocation_points = 256

    # веса loss
    bc_weight = 100.0
    phys_weight = 1.0

    # логирование
    print_every = 500

    # shooting method
    shooting_lr = 1e-2
    shooting_epochs = 3000
    shooting_steps = 400
    shooting_print_every = 300
