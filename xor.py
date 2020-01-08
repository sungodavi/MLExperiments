import numpy as np
import matplotlib.pyplot as plt
from lib import Model
from lib.nodes import Dense, Tanh
from lib.losses import MSE

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0]).reshape(-1, 1)

lr = 0.1

errs = []
for _ in range(20):
    model = Model([Dense(2, 4), Tanh(), Dense(4, 1)], MSE())
    for _ in range(1000):
        pred = model.forward(X)
        model.backward(pred, y, lr)

    err = np.abs(y - model.forward(X)).sum()
    errs.append(err)


plt.hist(errs)
plt.show()


