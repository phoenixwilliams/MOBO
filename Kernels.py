import numpy as np

class RBF:

    def __init__(self, x, x_):
        self.x = x
        self.x_ = x_
        self.theta = None

    def set_theta(self, theta):
        self.theta = theta


    def covar(self, x1, x2):
        sigma_f, sigma_l, sigma_n = self.theta[0], self.theta[1], self.theta[2]
        sqdist = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * np.dot(x1, x2.T)
        k = (sigma_f ** 2) * np.exp(((-1 / (2*(sigma_l ** 2)))) * sqdist)

        return k

    def predict(self, x_):
        sigma_f, sigma_l, sigma_n = self.theta[0], self.theta[1], self.theta[2]
        sqdist = np.sum(self.x ** 2, 1).reshape(-1, 1) + np.sum(x_ ** 2, 1) - 2 * np.dot(self.x, x_.T)
        k = (sigma_f ** 2) * np.exp(((-1 / (2*(sigma_l ** 2)))) * sqdist)
        k += (sigma_n**2) * np.eye(k.shape[0])

        return k

    def __call__(self, theta):
        sigma_f, sigma_l, sigma_n = theta[0], theta[1], theta[2]
        sqdist = np.sum(self.x ** 2, 1).reshape(-1, 1) + np.sum(self.x_ ** 2, 1) - 2 * np.dot(self.x, self.x_.T)
        k = (sigma_f ** 2) * np.exp(((-1 / (2*(sigma_l ** 2)))) * sqdist)
        k += (sigma_n**2)*np.eye(k.shape[0])

        return k

    def derivative(self, theta):
        sigma_f, sigma_l, sigma_n = theta[0], theta[1], theta[2]
        sqdist = np.sum(self.x ** 2, 1).reshape(-1, 1) + np.sum(self.x_ ** 2, 1) - 2 * np.dot(self.x, self.x_.T)

        k_ = np.exp(((-1 / (2*(sigma_l ** 2)))) * sqdist)

        return 2*sigma_f*k_, (sigma_f ** 2) * k_ * sqdist*(1./sigma_l**3), 2*sigma_n*np.eye(k_.shape[0])


    def input_test_deriv(self, x_):
        sigma_f, sigma_l, sigma_n = self.theta[0], self.theta[1], self.theta[2]
        sqdist = np.sum(self.x ** 2, 1).reshape(-1, 1) + np.sum(x_ ** 2, 1) - 2 * np.dot(self.x, x_.T)
        k = (sigma_f ** 2) * np.exp(((-1 / (2 * (sigma_l ** 2)))) * sqdist)
        k = k * (-1/(2*sigma_l**2) * (2*x_ - 2*self.x))

        return k




