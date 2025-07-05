import pennylane as qml
from pennylane import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

A = np.array([[1.0, 0.25],
              [-0.25, 1.0]])

b = np.array([1.0, 0.0])

dev = qml.device("default.qubit", wires=1)

def ansatz(theta):
    qml.RY(theta[0], wires=0)

@qml.qnode(dev)
def quantum_state(theta):
    ansatz(theta)
    return qml.state()

def cost(theta):
    psi = quantum_state(theta)
    Apsi = A @ psi
    error = Apsi - b
    return np.real(np.dot(error.conj(), error))

theta0 = np.array([0.1], requires_grad=False)
res = minimize(cost, theta0, method='Nelder-Mead')

theta_final = res.x
quantum_result = quantum_state(theta_final)
classical_result = np.linalg.solve(A, b)

print("Final theta:", theta_final)
print("Quantum solution:", quantum_result)
print("Classical solution:", classical_result)

angles = np.linspace(0, 2 * np.pi, 100)
cost_values = [cost([a]) for a in angles]

plt.plot(angles, cost_values)
plt.xlabel("theta")
plt.ylabel("cost")
plt.title("Cost vs theta")
plt.grid(True)
plt.show()


