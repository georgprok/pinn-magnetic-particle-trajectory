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

    # обучение
    lr = 1e-3
    epochs = 7000
    collocation_points = 256

    # веса loss
    bc_weight = 100.0
    phys_weight = 1.0

    # вывод
    print_every = 500
