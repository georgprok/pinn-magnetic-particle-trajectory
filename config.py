import math


class Config:
    # device
    device = "cuda"

    # seed
    seed = 42

    # physics
    q = 1.0
    m = 1.0
    Bz = 1.0

    # boundary conditions
    A = (0.0, 0.0)
    B = (1.0, 1.0)

    # time
    T = math.pi / 2

    # network
    hidden_dim = 64
    hidden_layers = 3

    # PINN training
    lr = 1e-3
    epochs = 7500
    collocation_points = 256

    # loss weights
    bc_weight = 100.0
    phys_weight = 1.0

    # logging
    print_every = 500

    # shooting method
    shooting_lr = 1e-2
    shooting_epochs = 1800
    shooting_steps = 400
    shooting_print_every = 300
