import pennylane as qml
from pennylane import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

A = np.array([[1.0, 0.25],
              [-0.25, 1.0]])

u = np.array([1.0, 0.0])
steps = 3
all_u = [u]

dev = qml.device("default.qubit", wires=1)

def quantum_solve(A, b):
    def ansatz(theta):
        qml.RY(theta[0], wires=0)

    @qml.qnode(dev)
    def quantum_state(theta):
        ansatz(theta)
        return qml.state()

    def cost(theta):
        psi = quantum_state(theta)
        Apsi = A @ psi
        diff = Apsi - b
        return np.real(np.dot(diff.conj(), diff))

    theta0 = np.array([0.1], requires_grad=False)
    result = minimize(cost, theta0, method='Nelder-Mead')
    return quantum_state(result.x)

for j in range(steps):
    u = quantum_solve(A, u)
    all_u.append(u)

u0 = [vec[0].real for vec in all_u]
u1 = [vec[1].real for vec in all_u]
time = range(len(all_u))

plt.plot(time, u0, marker='o', label='u[0]')
plt.plot(time, u1, marker='s', label='u[1]')
plt.xlabel('Time step')
plt.ylabel('Value')
plt.title('Hybrid Quantum-Classical Solver')
plt.grid(True)
plt.legend()
plt.show()
