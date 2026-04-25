import numpy as np
from core.objective_function import ObjectiveFunction


class Himmelblau(ObjectiveFunction):

    @property
    def name(self):
        return "Himmelblau"
    
    @property
    def bounds(self):
        return ((-10, 10), (-10, 10))
    
    def value(self, x):
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
    
    def grad(self, x):
        dx = 4*x[0] * (x[0]**2 + x[1] - 11) + 2*(x[0] + x[1]**2 - 7)
        dy = 2*( x[0]**2 + x[1] - 11 ) + 4*x[1]*(x[0] + x[1]**2 - 7)
        return np.array([dx, dy])
    
    def hess(self, x):
        return np.array([
            [12*x[0]**2 + 4*x[1] -42, 4*x[0]+4*x[1]],
            [4*x[0]+4*x[1], 4*x[0] + 12*x[1]**2 - 26]
        ])



class Rosenbrock(ObjectiveFunction):

    @property
    def name(self):
        return "Rosenbrock"

    @property
    def bounds(self):
        return ((-5, 10), (-5, 10))

    def value(self, x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

    def grad(self, x):
        dx = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
        dy = 200*(x[1] - x[0]**2)
        return np.array([dx, dy])

    def hess(self, x):
        return np.array([
            [2 - 400*x[1] + 1200*x[0]**2, -400*x[0]],
            [-400*x[0], 200]
        ])
    


class Rastrigin(ObjectiveFunction):

    @property
    def name(self):
        return "Rastrigin"

    @property
    def bounds(self):
        return ((-5.12, 5.12), (-5.12, 5.12))

    def value(self, x):
        return (
            20
            + x[0]**2 + x[1]**2
            - 10*(np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))
        )

    def grad(self, x):
        dx = 2*x[0] + 20*np.pi*np.sin(2*np.pi*x[0])
        dy = 2*x[1] + 20*np.pi*np.sin(2*np.pi*x[1])
        return np.array([dx, dy])

    def hess(self, x):
        dxx = 2 + 40*np.pi**2*np.cos(2*np.pi*x[0])
        dyy = 2 + 40*np.pi**2*np.cos(2*np.pi*x[1])
        return np.array([
            [dxx, 0.0],
            [0.0, dyy]
        ])


class LevyN13(ObjectiveFunction):

    @property
    def name(self):
        return "Levy_N13"

    @property
    def bounds(self):
        return ((-10, 10), (-10, 10))

    def value(self, x):
        x1, x2 = x
        return (
            np.sin(3*np.pi*x1)**2
            + (x1 - 1)**2 * (1 + np.sin(3*np.pi*x2)**2)
            + (x2 - 1)**2 * (1 + np.sin(2*np.pi*x2)**2)
        )

    def grad(self, x):
        # Analytical gradient is cumbersome -> ?????????
        eps = 1e-6
        grad = np.zeros_like(x)
        for i in range(2):
            e = np.zeros_like(x)
            e[i] = eps
            grad[i] = (self.value(x + e) - self.value(x - e)) / (2 * eps)
        return grad

    def hess(self, x):
        raise NotImplementedError("Hessian is not used for Levy N.13")



class Sphere(ObjectiveFunction):

    @property
    def name(self):
        return "Sphere"

    @property
    def bounds(self):
        return ((-5, 5), (-5, 5))

    def value(self, x):
        return np.sum(x**2)

    def grad(self, x):
        return 2 * x

    def hess(self, x):
        return 2 * np.eye(2)


class Ackley(ObjectiveFunction):

    @property
    def name(self):
        return "Ackley"

    @property
    def bounds(self):
        return ((-5, 5), (-5, 5))

    def value(self, x):
        a, b, c = 20, 0.2, 2*np.pi
        x1, x2 = x
        return (
            -a * np.exp(-b * np.sqrt(0.5 * (x1**2 + x2**2)))
            - np.exp(0.5 * (np.cos(c*x1) + np.cos(c*x2)))
            + a + np.e
        )

    def grad(self, x):
        # Numerical gradient (standard for Ackley)
        eps = 1e-6
        grad = np.zeros_like(x)
        for i in range(2):
            e = np.zeros_like(x)
            e[i] = eps
            grad[i] = (self.value(x + e) - self.value(x - e)) / (2 * eps)
        return grad

    def hess(self, x):
        raise NotImplementedError("Hessian is not defined for Ackley")
