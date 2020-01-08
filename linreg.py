import numpy as np
import random
import matplotlib.pyplot as plt

from lib import Model
from lib.nodes import Dense
from lib.losses import MSE


def get_data():
    m = random.uniform(-3, 3)
    b = random.uniform(-5, 5)

    points = []
    for _ in range(300):
        x = random.uniform(0, 100)
        y = (m * x + b) + random.gauss(0, 25)
        points.append((x, y))

    return points, m, b


def plot(data, line=None):
    x = [p[0] for p in data]
    y = [p[1] for p in data]

    plt.scatter(x, y)

    if line:
        w, b, losses = line
        h = [w * _x + b for _x in x]
        plt.plot(x, h)

        plt.figure()

        plt.plot(np.arange(len(losses)), np.array(losses))

    plt.show()


def get_line(data, lr=0.0001):
    X = np.array([p[0] for p in data]).reshape(-1, 1)
    y = np.array([p[1] for p in data]).reshape(-1, 1)

    losses = []
    model = Model([Dense(1, 1)], MSE())

    for _ in range(50):
        h = model.forward(X)
        losses.append(model.backward(h, y, lr))

    layer = model.layers[0]
    return layer.W[0, 0], layer.b[0, 0], losses


if __name__ == '__main__':
    data, true_m, true_b = get_data()
    w, b, losses = get_line(data)
    print(w, b, true_m, true_b)
    plot(data, (w, b, losses))

