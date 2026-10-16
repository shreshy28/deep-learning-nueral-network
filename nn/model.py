import numpy as np
from .layers import Dense, BatchNorm

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x, training=False):
        for layer in self.layers:
            x = layer.forward(x, training)
        return x

    def backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def parameters(self, l2=0):
        for i, layer in enumerate(self.layers):
            for name, value in getattr(layer, "params", {}).items():
                gradient = layer.grads[name]
                if isinstance(layer, Dense) and name == "W":
                    gradient = gradient+l2*value
                yield f"{i}.{name}", value, gradient

    def penalty(self, l2):
        return 0.5*l2*sum(np.sum(l.params["W"]**2) for l in self.layers if isinstance(l, Dense))

    def state(self):
        state = {}
        for i, layer in enumerate(self.layers):
            for name, value in getattr(layer, "params", {}).items():
                state[f"{i}.{name}"] = value.copy()
            if isinstance(layer, BatchNorm):
                state[f"{i}.mean"], state[f"{i}.var"] = layer.mean.copy(), layer.var.copy()
        return state

    def restore(self, state):
        expected = self.state()
        if set(expected) != set(state) or any(expected[k].shape != state[k].shape for k in expected):
            raise ValueError("Checkpoint does not match architecture")
        for i, layer in enumerate(self.layers):
            for name, value in getattr(layer, "params", {}).items():
                value[...] = state[f"{i}.{name}"]
            if isinstance(layer, BatchNorm):
                layer.mean, layer.var = state[f"{i}.mean"].copy(), state[f"{i}.var"].copy()

    def save(self, path):
        np.savez_compressed(path, **self.state())

    def load(self, path):
        with np.load(path, allow_pickle=False) as archive:
            self.restore(dict(archive))
