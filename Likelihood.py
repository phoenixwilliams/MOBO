import numpy as np
from scipy.linalg import cholesky
from Kernels import IndexKernel


class negativeGaussianLogLiklihood:
    def __init__(self, model):
        self.model = model

    def __call__(self, theta):
        self.model.set_params(theta)

        if isinstance(self.model.kernel, IndexKernel):
            k = self.model.kernel(self.model.train_x, self.model.train_x, self.model.train_i, self.model.train_i)
            k = k + 1e-3 * np.eye(k.shape[0])
            L = cholesky(k)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.model.train_y))

            logp = -0.5 * self.model.train_y.T.dot(alpha) - sum(np.log(np.diag(L))) \
                   - 0.5 * k.shape[0] * np.log(2 * np.pi)

            return -logp

        else:
            k = self.model.kernel(self.model.train_x, self.model.train_x)
            k = k + 1e-5*np.eye(k.shape[0])
            L = cholesky(k)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.model.train_y))

            logp = -0.5*self.model.train_y.T.dot(alpha) - sum(np.log(np.diag(L))) \
                   - 0.5*k.shape[0]*np.log(2 * np.pi)

            return -logp

    def derivative(self, theta):
        k = self.model.kernel(self.model.train_x, self.model.train_x)
        k = k + 1e-5*np.eye(k.shape[0])
        L = cholesky(k)
        ink = np.linalg.solve(L.T, np.eye(L.shape[0]))
        d = len(theta)

        kernel_derivs = self.model.kernel.derivative(theta)
        mll_derivs = np.zeros(d)

        for i in range(d):
            ## Equation from Gaussian Processes for Machine Learning Book
            mll_derivs[i] = 0.5 * np.dot(self.model.train_y.T, ink).dot(kernel_derivs[i]).dot(np.dot(ink, self.model.train_y)) \
                                    - 0.5 * sum(np.diag(np.dot(ink, kernel_derivs[i])))

        return -mll_derivs

