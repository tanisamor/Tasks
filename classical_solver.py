import numpy as np
import matplotlib.pyplot as plt
# Parameters( Please note, some values are supposed according to the situation)

c = 1.0          #speed
L = 1.0           # length along the x-direction
T = 0.1           # total time
N = 10            # number of spatial grid points (excluding boundaries)
dx = L / (N + 1)  # spatial step size
dt = 0.01         # time step
lambda_val = c * dt / dx
print("λ = ", lambda_val)

A = np.zeros((N, N))
for i in range(N):
   A[i, i] = 1.0
   if i > 0:
       A[i, i-1] = -lambda_val / 2
   if i < N - 1:
       A[i, i+1] = lambda_val / 2
print("Matrix A:")
print(A)

x = np.linspace(dx, L - dx, N)
u_j = np.sin(np.pi * x)
print("Initial u^j:")
print(u_j)

u_next = np.linalg.solve(A, u_j)

plt.plot(x, u_j, label="u^j (initial)")
plt.plot(x, u_next, label="u^{j+1} (after one step)")
plt.legend()
plt.title("Advection PDE: Classical Time Step")
plt.xlabel("x")
plt.ylabel("u")
plt.grid(True)
plt.show()
