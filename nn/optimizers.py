import numpy as np

class SGD:
    def __init__(self, lr=0.01, momentum=0.0):
        self.lr, self.momentum, self.velocity = lr, momentum, {}

    def step(self, parameters):
        for key, value, gradient in parameters:
            self.velocity[key] = self.momentum*self.velocity.get(key, np.zeros_like(value))+gradient
            value -= self.lr*self.velocity[key]

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr, self.beta1, self.beta2, self.epsilon = lr, beta1, beta2, epsilon
        self.m, self.v, self.t = {}, {}, 0

    def step(self, parameters):
        self.t += 1
        for key, value, gradient in parameters:
            self.m[key] = self.beta1*self.m.get(key, np.zeros_like(value))+(1-self.beta1)*gradient
            self.v[key] = self.beta2*self.v.get(key, np.zeros_like(value))+(1-self.beta2)*gradient**2
            m = self.m[key]/(1-self.beta1**self.t)
            v = self.v[key]/(1-self.beta2**self.t)
            value -= self.lr*m/(np.sqrt(v)+self.epsilon)
