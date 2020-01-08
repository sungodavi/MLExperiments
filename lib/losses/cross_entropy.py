import numpy as np


class CrossEntropy:
    def __init__(self):
        self.pred = None
        self.y = None

    def forward(self, pred, y):
        self.pred = pred
        self.y = y

        probs = self.softmax(pred)

        L = y * np.log(probs)
        return -np.mean(L)

    def backward(self):
        return self.pred - self.y

    def softmax(self, logits):
        exps = np.exp(logits - np.max(logits))
        return exps / np.sum(exps)


if __name__ == '__main__':
    pred = np.array([1, 1, 1])
    y = np.array([1, 0, 0])

    layer = CrossEntropy()

    print(layer.softmax(pred))
    print(layer.forward(pred, y))
    print(layer.backward())
