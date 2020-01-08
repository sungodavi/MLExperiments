import numpy as np


class Dense:
    def __init__(self, input_size, output_size):
        self.W = np.random.random((input_size, output_size))
        self.b = np.ones((1, output_size)) * 0.01

        self.X = None
        self.dW = None
        self.db = None
        self.reset_grads()

    def reset_grads(self):
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, grad):
        self.dW = self.X.T @ grad
        self.db = np.sum(grad, axis=0)

        return grad @ self.W.T

    def apply_grads(self, lr):  # TODO: Multiple optimizers
        self.W -= lr * self.dW
        self.b -= lr * self.db
