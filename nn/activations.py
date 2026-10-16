import numpy as np

class ReLU:
    def forward(self, x, training=True):
        self.mask = x > 0
        return np.maximum(x, 0)

    def backward(self, gradient):
        return gradient*self.mask

class Sigmoid:
    def forward(self, x, training=True):
        # exp(-abs(x)) avoids overflow on either side of zero.
        e = np.exp(-np.abs(x))
        self.output = np.where(x >= 0, 1/(1+e), e/(1+e))
        return self.output

    def backward(self, gradient):
        return gradient*self.output*(1-self.output)

class Softmax:
    def forward(self, x, training=True):
        e = np.exp(x-x.max(axis=1, keepdims=True))
        self.output = e/e.sum(axis=1, keepdims=True)
        return self.output

    def backward(self, gradient):
        # Jacobian-vector product, without constructing (classes, classes).
        return self.output*(gradient-(gradient*self.output).sum(axis=1, keepdims=True))
