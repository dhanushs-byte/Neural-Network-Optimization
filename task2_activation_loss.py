import numpy as np
import matplotlib.pyplot as plt

def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)
def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()
def mse(y_true, y_pred):
    return np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)
def cross_entropy(y_true, y_prob):
    eps = 1e-15
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1-eps)
    return -np.mean(y_true*np.log(p) + (1-y_true)*np.log(1-p))

p = softmax(np.array([2.0, 1.0, 0.1]))
print("Softmax:", np.round(p, 3), "sum:", round(p.sum(), 3))
x = np.linspace(-6, 6, 200)
plt.plot(x, relu(x), label="ReLU")
plt.plot(x, sigmoid(x), label="Sigmoid")
plt.plot(x, tanh(x), label="Tanh")
plt.title("Activation Functions"); plt.xlabel("Input"); plt.ylabel("Output")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("task2_activation_functions.png"); plt.show()

yt, yp = [1, 0, 1], [0.9, 0.2, 0.7]
print("MSE:", mse(yt, yp))
print("Binary cross-entropy:", cross_entropy(yt, yp))
print("Confident wrong:", cross_entropy([1], [0.01]))
print("Uncertain:", cross_entropy([1], [0.40]))
