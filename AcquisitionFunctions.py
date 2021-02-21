import numpy as np
import math

def cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def pdf(x):
    return np.exp((-x**2)/2) / np.sqrt(2 * np.pi)



class LCB:
    def __init__(self, gaussian_process, beta):
        self.gp = gaussian_process
        self.beta = beta

    def __call__(self, input):
        input = np.asarray([input])
        mean, var = self.gp.predict(input)
        mean, var = float(mean), float(var)

        return mean - np.sqrt(self.beta * var)





class nExpectedImprovement:

    def __init__(self, gaussian_process, best_val, eps):
        self.kernel = gaussian_process.kernel
        self.gp = gaussian_process
        self.best_val = best_val
        self.eps = eps


    def derivative(self, input):
        raise NotImplementedError("Not Currently Implemented")



    def __call__(self, input):
        input = np.asarray([input])
        mean, var = self.gp.predict(input)
        mean, var = float(mean), float(var)

        if var == 0:
            return 0
        else:
            Z = (self.best_val - mean - self.eps)/np.sqrt(var)
            ei = (self.best_val - mean) * cdf(Z) + np.sqrt(var) * pdf(Z)
            return -ei


class nProbabilityofImprovement:
    def __init__(self, gaussian_process, best_val, eps):
        self.kernel = kernel
        self.gp = gaussian_process
        self.best_val = best_val
        self.eps = eps


    def derivative(self, input):
        raise NotImplementedError("Not Currently Implemented")


    def __call__(self, input):
        input = np.asarray([input])
        mean, var = self.gp.predict(input)

        mean, var = float(mean), float(var)

        return -cdf((self.best_val - mean - self.eps) / np.sqrt(var))


