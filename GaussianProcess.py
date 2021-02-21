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



class Sphere:
    """
    Dummy Problem of the sphere benchmark problem
    """

    def __call__(self, x):
        return sum(xi**2 for xi in x)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    function = Sphere()
    bounds = np.asarray([[1e-1, 10], [1e-1, 10], [1e-4, 1]])
    problem = Sphere()

    x = np.random.uniform(-50, 50, size=(3, 1))
    y = np.asarray([problem(xi) for xi in x])

    kernel = Kernels.RBF(x, x)
    likelihood = Likelihood.negativeGaussianLogLiklihood(kernel, y)
    adamopt = GradientOptimizers.AdamOptimizer(likelihood, alpha=1e-2)
    adamopt.set_initial_point(bounds)
    start = time.time()
    theta, score = adamopt.optimize(500)
    kernel.set_theta(theta)

    gp = SingleTaskGP(x, y, kernel)
    xtest = np.linspace(-50, 50, 500)
    xtest = np.asarray([[xi] for xi in xtest])
    mean, var = gp.predict(xtest)

    variance = 2 * np.diag(var) #np.sqrt(np.diag(var))
    upper_confidence = mean + variance
    lower_confidence = mean - variance
    plt.plot(xtest, mean)
    plt.fill_between(xtest.flatten(), lower_confidence, upper_confidence, alpha=0.3)
    plt.plot(x, y, 'o')
    plt.show()

    acq = AcquisitionFunctions.LCB(gp, 1.)

    acq_plot = [acq(xi) for xi in xtest]
    plt.plot(xtest, acq_plot)


    mutant_params = {
        "F": 0.5
    }
    dimension = 1
    design = {
        "bounds": np.asarray([[-50] * dimension, [50] * dimension]),
        "iterations": 10,
        "mutant_function": EvolutionaryOptimizers.rand_1,
        "mutant_params": mutant_params,
        "popsize": 30,
        "cr": 0.9
    }

    de = EvolutionaryOptimizers.DifferentialEvolution(design)
    x, fitness = de.optimize(acq)
    print(time.time() - start)
    plt.plot(x, fitness, 'o')
    plt.show()








