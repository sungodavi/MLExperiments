import numpy as np
from utils import get_dim


class MaxPool:
    def __init__(self, input_size, kernel_size, stride):
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.indices = None
        self.X = None

        self.out_h = get_dim(input_size[1], kernel_size[0], 0, stride)
        self.out_w = get_dim(input_size[2], kernel_size[1], 0, stride)

    def forward(self, X):
        self.X = X
        self.indices = []

        result = np.empty((self.out_h, self.out_w))
        for channel in range(self.input_size[0]):
            channel_indices = []
            for r in range(self.out_h):
                row_indices = []
                for c in range(self.out_w):
                    beg_h = r * self.stride
                    end_h = beg_h + self.kernel_size[0]
                    beg_w = c * self.stride
                    end_w = beg_w + self.kernel_size[1]

                    X_slice = X[channel, beg_h:end_h, beg_w:end_w]
                    index = np.unravel_index(np.argmax(X_slice), X_slice.shape)
                    row_indices.append((channel, index[0] + beg_h, index[1] + beg_w))
                    result[r, c] = X_slice[index]
                channel_indices.append(row_indices)
            self.indices.append(channel_indices)

        return result

    def backward(self, grad):
        result = np.zeros_like(self.X)
        for i, channel in enumerate(self.indices):
            for j, row in enumerate(channel):
                for k, index in enumerate(row):
                    result[index[0], index[1], index[2]] += grad[i][j][k]

        return result

    def apply_grads(self, lr):
        pass


if __name__ == '__main__':
    layer = MaxPool((1, 4, 4), (2, 2), 1)

    X = np.array([[-5, -4, 0, 8],
              [-10, -2, 2, 3],
              [0, -2, -4, -7],
              [-3, -2, -3, -16]]).reshape(1, 4, 4)


    print(layer.forward(X))

    grad = np.ones((1, 3, 3)) * 3
    print(layer.backward(grad))

