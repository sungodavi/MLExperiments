import numpy as np


class BinaryCrossEntropy:
    def __init__(self):
        self.pred = None
        self.y = None

    def forward(self, pred, y):
        self.pred = pred
        self.y = y

        L = y * np.log(pred) + (1 - y) * np.log(pred)
        return -np.mean(L)

    def backward(self):
        diff = self.y / self.pred - (1 - self.y) / (1 - self.pred)
        return -diff / diff.shape[0]
