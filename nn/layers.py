import numpy as np

class Dense:
    """X: (batch, inputs), W: (inputs, outputs), b: (outputs,)."""
    def __init__(self, inputs, outputs, rng, initialization="he"):
        scale = np.sqrt(2 / inputs) if initialization == "he" else np.sqrt(2 / (inputs + outputs))
        self.params = {"W": rng.normal(0, scale, (inputs, outputs)), "b": np.zeros(outputs)}
        self.grads = {}

    def forward(self, x, training=True):
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, gradient):
        # Chain rule: sum parameter contributions across the batch.
        self.grads = {"W": self.x.T @ gradient, "b": gradient.sum(axis=0)}
        return gradient @ self.params["W"].T

class Dropout:
    """Inverted dropout preserves the expected activation during training."""
    def __init__(self, probability, rng):
        if not 0 <= probability < 1:
            raise ValueError("Dropout probability must be in [0, 1)")
        self.probability, self.rng = probability, rng

    def forward(self, x, training=True):
        self.mask = (self.rng.random(x.shape) >= self.probability) / (1-self.probability) if training else np.ones_like(x)
        return x * self.mask

    def backward(self, gradient):
        return gradient * self.mask

class BatchNorm:
    """Normalize each feature; running statistics serve inference."""
    def __init__(self, features, momentum=0.9, epsilon=1e-5):
        self.params = {"gamma": np.ones(features), "beta": np.zeros(features)}
        self.grads = {}
        self.mean, self.var = np.zeros(features), np.ones(features)
        self.momentum, self.epsilon = momentum, epsilon

    def forward(self, x, training=True):
        if training:
            mean, var = x.mean(axis=0), x.var(axis=0)
            self.mean = self.momentum*self.mean + (1-self.momentum)*mean
            self.var = self.momentum*self.var + (1-self.momentum)*var
        else:
            mean, var = self.mean, self.var
        self.inv = 1 / np.sqrt(var+self.epsilon)
        self.normalized = (x-mean)*self.inv
        return self.params["gamma"]*self.normalized+self.params["beta"]

    def backward(self, gradient):
        z = self.normalized
        self.grads = {"gamma": (gradient*z).sum(axis=0), "beta": gradient.sum(axis=0)}
        g = gradient*self.params["gamma"]
        # Derivative includes both batch mean and batch variance paths.
        return self.inv*(g-g.mean(axis=0)-z*(g*z).mean(axis=0))
