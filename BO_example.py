from GaussianProcess import SingleTaskGP
from AcquisitionFunctions import nExpectedImprovement
from EvolutionaryOptimizers import DifferentialEvolution, rand_1
from Kernels import RBF
from Likelihood import negativeGaussianLogLiklihood
from GradientOptimizers import AdamOptimizer
import numpy as np

class Sphere:
    """
    Dummy Problem of the sphere benchmark problem
    """

    def __call__(self, x):
        return sum(xi**2 for xi in x)



def show_fitted_gp_and_acquisition():
    import matplotlib.pyplot as plt

    # Set the bounds for the RBF Kernel parameters
    bounds = np.asarray([[1e-1, 10], [1e-1, 10], [1e-4, 1]])
    problem = Sphere()

    # Generate Dataset of (x,y) values
    x = np.random.uniform(-50, 50, size=(3, 1))
    y = np.asarray([problem(xi) for xi in x])

    # Instantiate the Kernel and likelihood used to optimize the kernel parameters
    kernel = RBF(x, x)
    likelihood = negativeGaussianLogLiklihood(kernel, y)
    adamopt = AdamOptimizer(likelihood, alpha=1e-2)
    adamopt.set_initial_point(bounds)
    theta, score = adamopt.optimize(500)
    kernel.set_theta(theta)

    # Initialize the Gaussian Process with the dataset and optimized Kernel
    gp = SingleTaskGP(x, y, kernel)
    xtest = np.linspace(-50, 50, 500)
    xtest = np.asarray([[xi] for xi in xtest])
    mean, var = gp.predict(xtest)

    # Plotting dataset and Gaussian Process Prediction
    variance = 2 * np.diag(var)
    upper_confidence = mean + variance
    lower_confidence = mean - variance
    plt.plot(xtest, mean)
    plt.fill_between(xtest.flatten(), lower_confidence, upper_confidence, alpha=0.3)
    plt.plot(x, y, 'o')
    plt.show()

    # Initialize acquisition function
    acq = nExpectedImprovement(gp, min(y), 0.01)
    acq_plot = [acq(xi) for xi in xtest]
    plt.plot(xtest, acq_plot)


    # Defining Differential Optimization Algorithm design
    mutant_params = {
        "F": 0.5
    }
    dimension = 1
    design = {
        "bounds": np.asarray([[-50] * dimension, [50] * dimension]),
        "iterations": 10,
        "mutant_function": rand_1,
        "mutant_params": mutant_params,
        "popsize": 30,
        "cr": 0.9
    }

    de = DifferentialEvolution(design)

    # Apply Differential Evolution to optimize Acquisition Function
    x, fitness = de.optimize(acq)

    # Plot acquisition function and minimum point returned by the acquisition optimization algorithm
    plt.plot(x, fitness, 'o')
    plt.show()


def example_bayesian_optimization():
    # Set the bounds for the RBF Kernel parameters
    bounds = np.asarray([[1e-1, 10], [1e-1, 10], [1e-4, 1]])
    problem = Sphere()

    dimension = 2

    problem_bounds = [-5, 5]
    diff = problem_bounds[1] - problem_bounds[0]

    # Generate Normalized Dataset of (x,y) values
    x = np.random.uniform(0, 1, size=(20, dimension))

    y = np.asarray([problem(xi*diff + problem_bounds[0]) for xi in x])

    # Defining Differential Optimization Algorithm design
    mutant_params = {
        "F": 0.5
    }

    design = {
        "bounds": np.asarray([[0] * dimension, [1] * dimension]),
        "iterations": 100,
        "mutant_function": rand_1,
        "mutant_params": mutant_params,
        "popsize": 100,
        "cr": 0.9
    }

    de = DifferentialEvolution(design)

    best_values = [min(y)]

    for i in range(50):
        print(i)

        # Instantiate the Kernel and likelihood used to optimize the kernel parameters
        kernel = RBF(x, x)
        likelihood = negativeGaussianLogLiklihood(kernel, y)
        adamopt = AdamOptimizer(likelihood, alpha=1e-2)
        adamopt.set_initial_point(bounds)
        theta, _ = adamopt.optimize(1000, 2)
        kernel.set_theta(theta)

        # Initialize the Gaussian Process with the dataset and optimized Kernel
        gp = SingleTaskGP(x, y, kernel)

        # Initialize acquisition function
        acq = nExpectedImprovement(gp, min(y), 0.01)

        # Apply Differential Evolution to optimize Acquisition Function
        new_point, _ = de.optimize(acq)

        # Ensure that the new_point output type is matching your custom problem
        val = problem(new_point*diff + problem_bounds[0])

        # Update dataset
        x = np.concatenate((x, [new_point]), axis=0)
        y = np.concatenate((y, [val]), axis=0)

        best_values.append(min(y))

    return best_values



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    #show_fitted_gp_and_acquisition()
    process = example_bayesian_optimization()

    plt.plot(process)
    plt.show()
