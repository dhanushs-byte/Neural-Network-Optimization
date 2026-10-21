import numpy as np
np.random.seed(1)
W1, b1 = np.random.randn(3, 4), np.zeros(4)
W2, b2 = np.random.randn(4, 1), np.zeros(1)

def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-x))
def forward(x):
    hidden = relu(x @ W1 + b1)
    output = sigmoid(hidden @ W2 + b2)
    return hidden, output

x = np.array([0.5, -1.2, 3.0])
hidden, output = forward(x)
print("W1:", W1.shape, "b1:", b1.shape, "W2:", W2.shape, "b2:", b2.shape)
print("Hidden values:", hidden)
print("Output probability:", output[0])
