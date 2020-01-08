import numpy as np


class MSE:
    def __init__(self):
        self.pred = None
        self.y = None

    def forward(self, pred, y):
        self.pred = pred
        self.y = y
        return np.square(pred - y).mean()

    def backward(self):
        return 2 / self.pred.shape[0] * (self.pred - self.y)
