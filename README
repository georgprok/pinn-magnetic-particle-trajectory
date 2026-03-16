# PINN Trajectory Optimization in a Magnetic Field

This project investigates **Physics-Informed Neural Networks (PINNs)** for computing the trajectory of a charged particle moving in a magnetic field between two fixed spatial points.

The PINN approach is compared with a classical **shooting method**, which solves the same boundary value problem by optimizing the initial velocity.

The goal is to study how well PINNs reproduce physically consistent trajectories compared to traditional numerical methods.

---

# Example Result

The trajectories produced by the PINN and the shooting method are visually almost identical.

![Trajectory comparison](results/trajectory_comparison.png)

PINN training loss:

![Training loss](results/loss.png)

Shooting optimization loss:

![Shooting loss](results/shooting_loss.png)

Numerical values depend on the specific experiment configuration and random seed.

---

# Problem Description

We consider the motion of a charged particle in a constant magnetic field

B = (0, 0, Bz)

The equations of motion follow the Lorentz force

m r'' = q (v × B)

In 2D this becomes

m x'' = q Bz y'  
m y'' = − q Bz x'

The task is to find a trajectory

(x(t), y(t))

such that

r(0) = A  
r(T) = B

with unknown initial velocity.

---

# Methods

Two approaches are implemented.

## Physics-Informed Neural Network (PINN)

The neural network predicts the trajectory

t → (x(t), y(t))

The loss function consists of

- boundary constraints
- physics residual of the differential equation

Automatic differentiation is used to compute velocities and accelerations.

---

## Shooting Method

The shooting method solves the same boundary value problem by optimizing the initial velocity.

Steps:

1. Guess initial velocity  
2. Integrate the equations of motion  
3. Minimize final position error

---

# Installation

```bash
pip install torch matplotlib numpy
```

# Run Experiment

```bash
python main.py
```

Results are saved in

```bash
results/
```
