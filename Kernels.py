import numpy as np


class IndexKernel:
    def __init__(self, x_kernel, f_kernel):
        self.x_kernel = x_kernel # Data kernel
        self.f_kernel = f_kernel # Function kernel
        self.train_mode = False

    def train(self):
        self.x_kernel.train()

    def eval(self):
        self.x_kernel.eval()

    def set_sqdist(self, x, x_):
        self.x_kernel.set_sqdit(x, x_)

    def set_params(self, theta):
        """
        We take the first k values as the parameters for the dataset kernel
        """
        self.x_kernel.set_params(theta[:self.x_kernel.num_params])
        self.f_kernel.set_params(theta[self.x_kernel.num_params:])

    def __call__(self, x1, x2, i1, i2):
        k = self.x_kernel(x1, x2)
        a = self.f_kernel()
        k_f = np.zeros(shape=(len(i1), len(i2)))

        for i in range(k_f.shape[0]):
            for j in range(k_f.shape[1]):
                k_f[i][j] = a[i1[i]][i2[j]]

        return k * k_f


class ICM:
    def __init__(self, num_tasks, r):
        # The number r can be fully characterized by the length of bounds
        self.num_tasks = num_tasks
        self.a = None
        self.r = r

    def __call__(self):
        b = np.outer(self.a[0], self.a[0].T)
        for i in range(1, len(self.a)):
            b += np.outer(self.a[i], self.a[i].T)

        return b

    def set_params(self, a_):
        self.a = np.reshape(a_, newshape=(self.r, self.num_tasks))


class RBF:
    def __init__(self):
        self.theta = None
        self.sqdist = None
        self.train_mode = False
        self.num_params = 3

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

    def set_sqdist(self, x, x_):
        self.sqdist = np.sum(x ** 2, 1).reshape(-1, 1) \
                               + np.sum(x_ ** 2, 1) - 2 * np.dot(x, x_.T)

    def set_params(self, theta):
        self.theta = theta

    def __call__(self, x, x_):

        if self.train_mode:
            sigma_f, sigma_l, sigma_n = self.theta[0], self.theta[1], self.theta[2]
            k = (sigma_f ** 2) * np.exp(((-1 / (2*(sigma_l ** 2)))) * self.sqdist)
            k += (sigma_n**2)*np.eye(k.shape[0])

        else:
            sqdist = np.sum(x ** 2, 1).reshape(-1, 1) + np.sum(x_ ** 2, 1) - 2 * np.dot(x, x_.T)
            sigma_f, sigma_l, sigma_n = self.theta[0], self.theta[1], self.theta[2]
            k = (sigma_f ** 2) * np.exp(((-1 / (2 * (sigma_l ** 2)))) * sqdist)

        return k


    def derivative(self, theta):
        # Computes the derivative for each theta with dataset x, x_

        sigma_f, sigma_l, sigma_n = theta[0], theta[1], theta[2]
        k_ = np.exp((-1 / (2 * (sigma_l ** 2))) * self.sqdist)
        return 2*sigma_f*k_, (sigma_f ** 2) * k_ * self.sqdist*(sigma_l**-3), 2*sigma_n*np.eye(k_.shape[0])


if __name__ == "__main__":
    from Likelihood import negativeGaussianLogLiklihood

    i = np.random.randint(0, 2, size=(10,))
    icm = ICM(2, [[0, 1]])
    icm.initial_matrix()

    dataset = np.random.uniform(-50, 50, size=(10, 20))
    rbf = RBF()
    rbf.set_sqdist(dataset, dataset)
    rbf.set_params([10, 10, 1])
    rbf.eval()

    x_test = np.random.uniform(-50, 50, size=(2, 20))
    itest = np.random.randint(0, 1, size=(2,))
    k = IndexKernel(rbf, icm)

    k.set_params([1, 1, 1, 1, 1])




