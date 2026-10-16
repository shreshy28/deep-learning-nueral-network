import numpy as np
from .activations import Softmax

class CrossEntropy:
    """Fused softmax cross-entropy accepts logits and integer labels."""
    def forward(self, logits, targets):
        self.targets = np.asarray(targets, dtype=int)
        shifted = logits-logits.max(axis=1, keepdims=True)
        self.probabilities = Softmax().forward(shifted)
        return np.mean(np.log(np.exp(shifted).sum(axis=1))-shifted[np.arange(len(targets)), self.targets])

    def backward(self):
        g = self.probabilities.copy()
        g[np.arange(len(g)), self.targets] -= 1
        return g/len(g)

class MSE:
    def forward(self, prediction, targets):
        self.difference = prediction-targets
        return np.mean(self.difference**2)

    def backward(self):
        return 2*self.difference/self.difference.size
