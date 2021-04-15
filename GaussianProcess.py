import numpy as np
from scipy.linalg import cholesky


class SingleTaskGP:
    def __init__(self, train_x, train_y, kernel):

        self.train_x = train_x
        self.train_y = train_y
        self.kernel = kernel #assumed to be conditioned on training data

        self.covar = None
        self.L = None
        self.alpha = None

    def initialize(self):
        self.covar = self.kernel(self.train_x, self.train_x)
        self.L = cholesky(self.covar + 1e-5 * np.eye(self.covar.shape[0]), lower=True)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.train_y))

    def set_params(self, params):
        self.kernel.set_params(params)

    def predict(self, x_test):
        k_star = self.kernel(self.train_x, x_test)
        k_star_star = self.kernel(x_test, x_test)
        mean = k_star.T.dot(self.alpha)

        v = np.linalg.solve(self.L, k_star).T
        var = k_star_star - np.dot(v, v.T)

        return mean, var


class HetertopicMultiTaskGP:

    def __init__(self, train_x, train_y, train_i, kernel):

        self.train_x = train_x
        self.train_i = train_i
        self.train_y = train_y
        self.kernel = kernel

        self.covar = None
        self.L = None
        self.alpha = None


    def initialize(self):
        self.covar = self.kernel(self.train_x, self.train_x ,self.train_i, self.train_i)
        self.L = cholesky(self.covar + 1e-5 * np.eye(self.covar.shape[0]), lower=True)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.train_y))

    def set_params(self, params):
        self.kernel.set_params(params)

    def predict(self, x_test, i_test):
        k_star = self.kernel(self.train_x, x_test, self.train_i, i_test)
        k_star_star = self.kernel(x_test, x_test, i_test, i_test)
        mean = k_star.T.dot(self.alpha)

        v = np.linalg.solve(self.L, k_star).T
        var = k_star_star - np.dot(v, v.T)

        return mean, var





