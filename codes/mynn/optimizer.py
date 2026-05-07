from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.grads[key] is None:
                        continue
                    if layer.weight_decay and key == 'W':
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] -= self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu=0.9):
        super().__init__(init_lr, model)
        self.mu = mu
        self.velocity = {}

    def step(self):
        for layer_idx, layer in enumerate(self.model.layers):
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.grads[key] is None:
                        continue
                    state_key = (layer_idx, key)
                    if state_key not in self.velocity:
                        self.velocity[state_key] = np.zeros_like(layer.params[key])
                    if layer.weight_decay and key == 'W':
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    self.velocity[state_key] = (
                        self.mu * self.velocity[state_key]
                        - self.init_lr * layer.grads[key]
                    )
                    layer.params[key] += self.velocity[state_key]


class AdaGrad(Optimizer):
    def __init__(self, init_lr, model, eps=1e-8):
        super().__init__(init_lr, model)
        self.eps = eps
        self.accumulator = {}

    def step(self):
        for layer_idx, layer in enumerate(self.model.layers):
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.grads[key] is None:
                        continue
                    state_key = (layer_idx, key)
                    if state_key not in self.accumulator:
                        self.accumulator[state_key] = np.zeros_like(layer.params[key])
                    if layer.weight_decay and key == 'W':
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    self.accumulator[state_key] += layer.grads[key] ** 2
                    layer.params[key] -= (
                        self.init_lr
                        * layer.grads[key]
                        / (np.sqrt(self.accumulator[state_key]) + self.eps)
                    )


class RMSProp(Optimizer):
    def __init__(self, init_lr, model, rho=0.9, eps=1e-8):
        super().__init__(init_lr, model)
        self.rho = rho
        self.eps = eps
        self.square_avg = {}

    def step(self):
        for layer_idx, layer in enumerate(self.model.layers):
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.grads[key] is None:
                        continue
                    state_key = (layer_idx, key)
                    if state_key not in self.square_avg:
                        self.square_avg[state_key] = np.zeros_like(layer.params[key])
                    if layer.weight_decay and key == 'W':
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    self.square_avg[state_key] = (
                        self.rho * self.square_avg[state_key]
                        + (1 - self.rho) * layer.grads[key] ** 2
                    )
                    layer.params[key] -= (
                        self.init_lr
                        * layer.grads[key]
                        / (np.sqrt(self.square_avg[state_key]) + self.eps)
                    )


class Adam(Optimizer):
    def __init__(self, init_lr, model, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(init_lr, model)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self):
        self.t += 1
        for layer_idx, layer in enumerate(self.model.layers):
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.grads[key] is None:
                        continue
                    state_key = (layer_idx, key)
                    if state_key not in self.m:
                        self.m[state_key] = np.zeros_like(layer.params[key])
                        self.v[state_key] = np.zeros_like(layer.params[key])
                    if layer.weight_decay and key == 'W':
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    self.m[state_key] = (
                        self.beta1 * self.m[state_key]
                        + (1 - self.beta1) * layer.grads[key]
                    )
                    self.v[state_key] = (
                        self.beta2 * self.v[state_key]
                        + (1 - self.beta2) * layer.grads[key] ** 2
                    )
                    m_hat = self.m[state_key] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[state_key] / (1 - self.beta2 ** self.t)
                    layer.params[key] -= self.init_lr * m_hat / (np.sqrt(v_hat) + self.eps)
