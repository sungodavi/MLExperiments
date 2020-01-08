import numpy as np


class Relu:
    def __init__(self):
        self.X = None

    def reset_grads(self):
        pass

    def forward(self, X):
        self.X = X
        return np.maximum(X, 0)

    def backward(self, grad):
        return np.greater(self.X, 0).astype(int) * grad

    def apply_grads(self, lr):
        pass


if __name__ == '__main__':
    X = np.array([1, -1])
    grad = np.array([-1, 100])
    layer = Relu()
    print(layer.forward(X))
    print(layer.backward(grad))
