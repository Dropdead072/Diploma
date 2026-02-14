import numpy as np
from core.optimizer import Optimizer
from typing import Callable


class GradientDescent(Optimizer):
    def __init__(self, lr: float =1e-3):
        '''
        Docstring for __init__
        
        :param lr: learning rate
        :type lr: float
        '''
        super().__init__()
        self.lr = lr

    @property
    def name(self):
        return "Gradient Descent"
    
    def step(self, x, grad_fn: Callable[[np.ndarray], np.ndarray], hess = None):
        grad = grad_fn(x)
        self.iterations += 1
        return x - self.lr * grad
    

class GradientDescentWithMomentum(Optimizer):
    def __init__(self, lr: float =1e-3, momentum_coef: float =0.9):
        '''
        Docstring for __init__
        
        :param lr: learning rate
        :type lr: float
        :param momentum: momentum coeficient
        :type momentum: float
        '''
        super().__init__()
        self.lr = lr
        self.momentum_coef = momentum_coef
        self.momentum = None

    @property
    def name(self):
        return "Gradient Descent With Momentum"

    def step(self, x, grad_fn: Callable[[np.ndarray], np.ndarray], hess=None):
        if self.momentum is None:
            self.momentum = np.zeros_like(x)

        grad = grad_fn(x)

        self.momentum = self.momentum_coef * self.momentum + grad
        return x - self.lr * self.momentum
    

class NesterovMomentum(Optimizer):
    def __init__(self, lr: float=1e-3, momentum_coef: float =0.9):
        super().__init__()
        self.lr = lr
        self.momentum_coef = momentum_coef
        self.momentum = None

    @property
    def name(self):
        return "Nesterov Momentum"
    
    def step(self, x, grad_fn: Callable[[np.ndarray], np.ndarray], hess=None):
        if self.momentum is None:
            self.momentum = np.zeros_like(x)

        self.momentum = self.momentum_coef * self.momentum + grad_fn(x - self.momentum_coef * self.momentum)
        return x - self.lr * self.momentum

