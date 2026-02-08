import numpy as np
from core.optimizer import Optimizer


class GradientDescent(Optimizer):
    def __init__(self, lr: float =1e-3):
        super().__init__()
        self.lr = lr

    @property
    def name(self):
        return "GradientDescent"
    
    def step(self, x, grad, hess = None):
        self.iterations += 1
        return x - self.lr * grad
    

