import numpy as np
from scipy.linalg import cholesky


class negativeGaussianLogLiklihood:
    def __init__(self, kernel, y_train):
        self.kernel = kernel
        self.y_train = y_train



    def __call__(self, theta):
        k = self.kernel(theta)
        k = k + 1e-5*np.eye(k.shape[0])
        L = cholesky(k)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))

        logp = -0.5*self.y_train.T.dot(alpha) - sum(np.log(np.diag(L))) \
               - 0.5*k.shape[0]*np.log(2 * np.pi)

        return -logp


    def derivative(self, theta):
        k = self.kernel(theta)
        k = k + 1e-5*np.eye(k.shape[0])
        L = cholesky(k)
        ink = np.linalg.solve(L.T, np.eye(L.shape[0]))
        d = len(theta)

        kernel_derivs = self.kernel.derivative(theta)
        mll_derivs = np.zeros(d)

        for i in range(d):
            ## Equation from Gaussian Processes for Machine Learning Book
            mll_derivs[i] = 0.5 * np.dot(self.y_train.T, ink).dot(kernel_derivs[i]).dot(np.dot(ink, self.y_train)) \
                                    - 0.5 * sum(np.diag(np.dot(ink, kernel_derivs[i])))

        return -mll_derivs



