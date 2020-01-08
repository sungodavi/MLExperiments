class Model:
    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss
        self.pred = None

    def forward(self, X):
        h = X
        for layer in self.layers:
            h = layer.forward(h)

        return h

    def backward(self, h, y, lr):
        curr_loss = self.loss.forward(h, y)

        for layer in self.layers:
            layer.reset_grads()

        grad = self.loss.backward()

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        for layer in self.layers:
            layer.apply_grads(lr)

        return curr_loss
