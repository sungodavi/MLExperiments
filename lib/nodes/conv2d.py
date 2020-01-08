import numpy as np
from .utils import get_dim, im2col_indices, col2im_indices


class Conv2d:
    def __init__(self, n_channels, input_size, kernel_size, stride, padding='same'):
        assert kernel_size % 2 != 0
        self.padding = kernel_size // 2 if padding == 'same' else 0
        self.stride = stride
        self.in_channels = input_size[0]
        self.kernel_size = kernel_size
        self.n_channels = n_channels
        self.out_h = get_dim(input_size[1], kernel_size, self.padding, stride)
        self.out_w = get_dim(input_size[2], kernel_size, self.padding, stride)

        self.W_col = np.random.uniform(-1, 1, (n_channels, self.in_channels * self.kernel_size ** 2))
        self.b = np.zeros((1, n_channels))

        self.X = None
        self.X_col = None
        self.dW = None
        self.db = None
        self.reset_grads()

    def reset_grads(self):
        self.dW = np.zeros(self.W_col.shape)
        self.db = np.zeros(self.b.shape)

    def forward(self, X):
        n = X.shape[0]

        self.X = X
        self.X_col = im2col_indices(X, self.kernel_size, self.kernel_size, self.padding, self.stride)

        out = self.W_col @ self.X_col + self.b
        return out.reshape(n, self.out_h, self.out_w, self.n_channels).transpose(3, 0, 1, 2)

    def backward(self, grad):
        self.db = np.sum(grad, (0, 2, 3)).reshape(self.n_channels, -1)

        grad_reshaped = grad.transpose(1, 2, 3, 0).reshape(self.n_channels, -1)
        self.dW = np.dot(grad_reshaped, self.X_col.T)

        dX_col = self.W_col.T @ grad_reshaped
        return col2im_indices(dX_col, self.X.shape, self.kernel_size, self.kernel_size, self.padding, self.stride)

    def apply_grads(self, lr):
        self.W_col -= lr * self.dW
        self.b -= lr * self.db


if __name__ == '__main__':
    layer = Conv2d(1, input_size=(1, 6, 6), kernel_size=3, stride=1, padding='valid')
    assert layer.out_w == 4
    assert layer.out_h == 4
    layer.W_col = np.array([1, 0, -1, 1, 0, -1, 1, 0, -1]).reshape([1, -1])
    layer.b = np.array([[1]])

    X = np.array([[3, 0, 1, 2, 7, 4],
                  [1, 5, 8, 9, 3, 1],
                  [2, 7, 2, 5, 1, 3],
                  [0, 1, 3, 1, 7, 8],
                  [4, 2, 1, 6, 2, 8],
                  [2, 4, 5, 2, 3, 9]]).reshape(1, 1, 6, 6)

    result = layer.forward(X)
    target = np.array([[-5, -4, 0, 8],
                      [-10, -2, 2, 3],
                      [0, -2, -4, -7],
                      [-3, -2, -3, -16]]).reshape(1, 1, 4, 4) + 1

    assert np.array_equal(result, target)
