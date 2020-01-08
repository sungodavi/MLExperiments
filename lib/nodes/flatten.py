import numpy as np


class Flatten:
    def __init__(self):
        self.X = None

    def reset_grads(self):
        pass

    def forward(self, X):
        self.X = X
        return X.reshape(X.shape[0], -1)

    def backward(self, grad):
        return np.reshape(grad, self.X.shape)

    def apply_grads(self, lr):
        pass


if __name__ == '__main__':
    layer = Flatten()

    X = np.empty((100, 16, 16, 16))

    result = layer.forward(X)
    assert result.shape == (100, 16 ** 3)

    grad = np.empty((100, 16 ** 3))
    back_result = layer.backward(grad)

    assert back_result.shape == X.shape
