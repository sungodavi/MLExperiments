import numpy as np


class Tanh:
    def __init__(self):
        self.sig = None

    def reset_grads(self):
        pass

    def apply_grads(self, lr):
        pass

    def forward(self, X):
        self.sig = np.tanh(X)
        return self.sig

    def backward(self, grad):
        gradient = 1 - self.sig ** 2
        return grad * gradient
