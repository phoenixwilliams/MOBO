from GaussianProcess import HetertopicMultiTaskGP
from Kernels import RBF, ICM, IndexKernel
import numpy as np
from Likelihood import negativeGaussianLogLiklihood
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time



def fitting_and_plotting():
    from GlobalOptimisation.NonSeparable import Functions as ns
    from GlobalOptimisation.Separable import Functions as s

    pb = [-50, 50]
    problem1 = s.Ackley(None, -20, 0.2, 20)
    problem2 = s.Rastrigin(None)
    # Create dataset
    # Set the bounds for the RBF Kernel parameters

    dimension = 5
    train_size = 50
    test_size = 50

    final_diff1 = 0
    final_diff2 = 0

    for i in range(10):
        print(i)

        # Generate Dataset of (x,y) values
        x1 = np.random.uniform(0, 1, size=(train_size, dimension))
        x2 = np.random.uniform(0, 1, size=(train_size, dimension))
        x = np.concatenate([x1, x2])

        x1test = np.random.uniform(0, 1, size=(test_size, dimension))
        x2test = np.random.uniform(0, 1, size=(test_size, dimension))

        y1 = np.array([problem1(xi * (pb[1]-pb[0]) + pb[0]) for xi in x1])
        y2 = np.array([problem2(xi * (pb[1] - pb[0]) + pb[0]) for xi in x2])
        y = np.concatenate([y1, y2])

        y1test = np.asarray([problem1(xi * (pb[1]-pb[0]) + pb[0]) for xi in x1test])
        y2test = np.asarray([problem2(xi * (pb[1]-pb[0]) + pb[0]) for xi in x2test])

        i1 = np.zeros(shape=(len(y1),), dtype=int)
        i2 = np.ones(shape=(len(y2),), dtype=int)
        i = np.concatenate([i1, i2])

        i1test = 0 * np.ones(shape=(len(y1test),), dtype=int)
        i2test = 1 * np.ones(shape=(len(y2test),), dtype=int)

        kernel_x = RBF()
        kernel_f = ICM(2, 1)
        kernel = IndexKernel(kernel_x, kernel_f)

        gp = HetertopicMultiTaskGP(x, y, i, kernel)
        likelihood = negativeGaussianLogLiklihood(gp)
        bounds = [[1e-5, 1e+1], [1e-5, 1e+1], [1e-5, 1e-1], [1e-5, 1e+3], [1e-5, 1e+3]]

        x0 = [np.random.uniform(bounds[0][0], bounds[0][1]),
          np.random.uniform(bounds[1][0], bounds[1][1]),
          np.random.uniform(bounds[2][0], bounds[2][1]),
          np.random.uniform(bounds[3][0], bounds[3][1]),
          np.random.uniform(bounds[4][0], bounds[4][1])]

        params = minimize(likelihood, x0, bounds=bounds, options={"maxfun": 1500, "maxiter": 1500})

        #print(params.fun)
        #print(params.x)
        gp.set_params(params.x)
        gp.initialize()

        gp.kernel.eval()
        mean, var = gp.predict(x1test, i1test)

        avg_diff = 0
        for i in range(len(mean)):
            avg_diff = abs(mean[i] - y1test[i])

        avg_diff = avg_diff / len(mean)
        final_diff1 += avg_diff

        mean, var = gp.predict(x2test, i2test)

        avg_diff = 0
        for i in range(len(mean)):
            avg_diff = abs(mean[i] - y2test[i])

        avg_diff = avg_diff / len(mean)
        final_diff2 += avg_diff

    print(final_diff1/10)
    print(final_diff2/10)

    """
    xtest = np.linspace(0, 1, 500)
    xtest = np.asarray([[xi] for xi in xtest])

    itest = np.zeros(shape=(500,), dtype=int)
    gp.kernel.eval()
    mean, var = gp.predict(xtest, itest)
    ytest = [problem1(xi*(pb[1]-pb[0]) + pb[0]) for xi in xtest]
    plt.plot(xtest, ytest)
    plt.plot(xtest, mean)
    plt.plot(x1, y1, 'o')
    plt.show()

    itest = np.ones(shape=(500,), dtype=int)
    gp.kernel.eval()
    mean, var = gp.predict(xtest, itest)
    ytest = [problem2(xi * (pb[1] - pb[0]) + pb[0]) for xi in xtest]
    plt.plot(xtest, ytest)
    plt.plot(xtest, mean)
    plt.plot(x2, y2, 'o')
    plt.show()
    """



if __name__ == "__main__":
    fitting_and_plotting()
