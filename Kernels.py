import numpy as np
import itertools

class ICM:
    def __init__(self, num_tasks, bounds):
        # The number r can be fully characterized by the length of bounds
        self.num_tasks = num_tasks
        self.bounds = bounds
        self.a = None

    def initial_matrix(self):
        self.a = []
        for i in range(len(self.bounds)):
            self.a.append(np.random.uniform(self.bounds[i][0], self.bounds[i][1], size=(self.num_tasks,)))

    def __call__(self):

        b = np.outer(self.a[0], self.a[0].T)
        for i in range(1, len(self.a)):
            b += np.outer(self.a[i], self.a[i].T)

        return b


class IndexKernel:
    def __init__(self, i, i_, num_tasks):
        self.i = i
        self.i_ = i_
        self.num_tasks = num_tasks

        self.index_matrix = np.asarray([[[id1, id2] for id1 in self.i] for id2 in self.i_])


    def __call__(self, lcm):
        b = lcm()
        index_b = np.zeros((self.index_matrix.shape[0], self.index_matrix.shape[1]))

        for i in range(self.index_matrix.shape[0]):
            for j in range(self.index_matrix.shape[1]):
                #print(self.index_matrix[i][j][0], self.index_matrix[i][j][1])
                index_b[i][j] = b[self.index_matrix[i][j][0]][self.index_matrix[i][j][1]]

        return np.asarray(index_b)




class RBF:

    def __init__(self, x, x_):
        self.x = x
        self.x_ = x_
        self.theta = None
        self.sqdist = np.sum(self.x ** 2, 1).reshape(-1, 1) \
                               + np.sum(self.x_ ** 2, 1) - 2 * np.dot(self.x, self.x_.T)

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
        k = (sigma_f ** 2) * np.exp(((-1 / (2*(sigma_l ** 2)))) * self.sqdist)
        k += (sigma_n**2)*np.eye(k.shape[0])

        return k


    def derivative(self, theta):
        # Computes the derivative for each theta with dataset x, x_

        sigma_f, sigma_l, sigma_n = theta[0], theta[1], theta[2]
        k_ = np.exp(((-1 / (2*(sigma_l ** 2)))) * self.sqdist)
        return 2*sigma_f*k_, (sigma_f ** 2) * k_ * self.sqdist*(sigma_l**-3), 2*sigma_n*np.eye(k_.shape[0])


    def input_test_deriv(self, x_):
        sigma_f, sigma_l, sigma_n = self.theta[0], self.theta[1], self.theta[2]
        sqdist = np.sum(self.x ** 2, 1).reshape(-1, 1) + np.sum(x_ ** 2, 1) - 2 * np.dot(self.x, x_.T)
        k = (sigma_f ** 2) * np.exp(((-1 / (2 * (sigma_l ** 2)))) * sqdist)
        k = k * (-1/(2*sigma_l**2) * (2*x_ - 2*self.x))

        return k


if __name__ == "__main__":
    import time
    #i = np.random.randint(0, 2, size=(10,))
    #icm = ICM(2, [[0, 1]])
    #icm.initial_matrix()
    #index_kernel = IndexKernel(i, i, 2)
    #b = index_kernel(icm)

    dataset = np.random.uniform(-50, 50, size=(1000, 20))
    rbf = RBF(dataset, dataset)
    start = time.time()
    rbf([1, 1, 1])
    print(time.time() - start)






