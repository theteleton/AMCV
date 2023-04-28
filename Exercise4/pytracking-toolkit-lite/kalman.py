import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt


from ex4_utils import kalman_step


def function():
    N = 40
    v = np.linspace(5 * math.pi, 0, N)
    x = np.cos(v) * v
    y = np.sin(v) * v
    return x, y

def kalman_filter(x, y, A, C, Q_1, R_1):
    sx = np.zeros((x.size, 1), dtype=np.float32).flatten()
    sy = np.zeros((y.size, 1), dtype=np.float32).flatten()

    sx[0] = x[0]
    sy[0] = y[0]

    state = np.zeros((A.shape[0], 1), dtype=np.float32).flatten()
    state[0] = x[0]
    state[1] = y[0]
    covariance = np.eye(A.shape[0], dtype=np.float32)

    for j in range(1, x.size):
        state, covariance, _, _ = kalman_step(A, C, Q_1, R_1, np.reshape(np.array([x[j], y[j]]), (-1 , 1)), np.reshape(state, (-1, 1)), covariance)
        sx[j] = state[0]
        sy[j] = state[1]

    return sx, sy

def find_matrices(type_movement):
    if type_movement == "RW":
        T, q = sp.symbols("T q")
        F = sp.Matrix([[0, 0], [0, 0]])
        Fi = sp.exp(F*T)
        L = sp.Matrix([[1, 0], [0, 1]])
        Q = sp.integrate((Fi * L) * q * (Fi * L).T, (T, 0, T))
        print(Fi)
        print(Q)
    elif type_movement == "NCV":
        T, q = sp.symbols("T q")
        F = sp.Matrix([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        Fi = sp.exp(F*T)
        L = sp.Matrix([[0, 0], [0, 0], [1, 0], [0, 1]])
        Q = sp.integrate((Fi * L) * q * (Fi * L).T, (T, 0, T))
        print(Fi)
        print(Q)
    elif type_movement == "NCA":
        T, q = sp.symbols("T q")
        F = sp.Matrix([[0, 0, 1, 0, T, 0], [0, 0, 0, 1, 0, T], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        Fi = sp.exp(F*T)
        L = sp.Matrix([[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 1]])
        Q = sp.integrate((Fi * L) * q * (Fi * L).T, (T, 0, T))
        print(Fi)
        print(Q)

def compute_matrices(movement, T, q, r):
    if movement == "RW":
        Fi = np.array([[1, 0], [0, 1]])
        Q = np.array([[T*q, 0], [0, T*q]])
        H = np.array([[1, 0], [0, 1]])
        R = np.array([[r, 0], [0, r]])
    elif movement == "NCV":
        Fi = np.array([[1, 0, T, 0], [0, 1, 0, T], [0, 0, 1, 0], [0, 0, 0, 1]])
        Q = np.array([[T**3*q/3, 0, T**2*q/2, 0], [0, T**3*q/3, 0, T**2*q/2], [T**2*q/2, 0, T*q, 0], [0, T**2*q/2, 0, T*q]])
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        R = np.array([[r, 0], [0, r]])
    elif movement == "NCA":
        Fi = np.array([[1, 0, T, 0, 3*T**2/2, 0], [0, 1, 0, T, 0, 3*T**2/2], [0, 0, 1, 0, T, 0], [0, 0, 0, 1, 0, T], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
        Q = np.array([[9*T**5*q/20, 0, 3*T**4*q/8, 0, T**3*q/2, 0], [0, 9*T**5*q/20, 0, 3*T**4*q/8, 0, T**3*q/2], [3*T**4*q/8, 0, T**3*q/3, 0, T**2*q/2, 0], [0, 3*T**4*q/8, 0, T**3*q/3, 0, T**2*q/2], [T**3*q/2, 0, T**2*q/2, 0, T*q, 0], [0, T**3*q/2, 0, T**2*q/2, 0, T*q]])
        H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
        R = np.array([[r, 0], [0, r]])
    else:
        return "Nothing", 0, 0, 0

    return Fi, Q, H, R

if __name__ == "__main__":
    x, y = function()
    x = np.array([0, 0, 1, 1, 0])
    y = np.array([0, 1, 1, 0, 0])
    x = np.array([0, 1, 4, 6, 5, 5, 3, 2, 0])
    y = np.array([0, 3, 3, 2, 1, -1, -2, -4, 0])

    fig, ax = plt.subplots(3, 5, figsize=(15, 10))
    i = 0
    j = 0
    for move in ["RW", "NCV", "NCA"]:
        j = 0
        for (q, r) in [(100, 1), (5, 1), (1, 1), (1, 5), (1, 100)]:
            Fi, Q, H, R = compute_matrices(move, 1, q, r)
            sx, sy = kalman_filter(x, y, Fi, H, Q, R)
            ax[i, j].plot(x, y, "-o", label="Measurements (observations)")
            ax[i, j].plot(sx, sy, "-o", label="Filtered measurements")
            ax[i, j].set_title(f"{move}: q={q}, r={r}", fontsize=13)
            j += 1
        i += 1

    fig.savefig("./sol.png")