import numpy as np


class Sigmoid:
    def __init__(self):
        self.sig = None

    def reset_grads(self):
        pass

    def apply_grads(self, lr):
        pass

    def forward(self, X):
        self.sig = 1.0 / (1 + np.exp(-X))
        return self.sig

    def backward(self, grad):
        gradient = self.sig * (1 - self.sig)
        return grad * gradient
