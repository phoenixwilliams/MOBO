import numpy as np
import time
import Kernels
import Likelihood
import GradientOptimizers
import AcquisitionFunctions
import EvolutionaryOptimizers



class SingleTaskGP:
    def __init__(self, train_x, train_y, kernel):

        self.train_x = train_x
        self.train_y = train_y
        self.kernel = kernel #assumes to be conditioned on training data

        self.covar = kernel(kernel.theta)
        self.L = np.linalg.cholesky(self.covar + 1e-5*np.eye(self.covar.shape[0]))
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.train_y))

    def predict(self, x_test):
        k_star = self.kernel.covar(self.train_x, x_test)
        k_star_star = self.kernel.covar(x_test, x_test)
        mean = k_star.T.dot(self.alpha)

        v = np.linalg.solve(self.L, k_star).T
        var = k_star_star - np.dot(v, v.T)

        return mean, var









