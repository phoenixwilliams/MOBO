import numpy as np
import Likelihood
import Kernels

class GradientDescentOptimizer:
    def __init__(self, function, lr=1e-4, momentum = 0.0, tol=1e-2):

        self.function = function
        self.lr = lr
        self.momentum = momentum
        self.x = None
        self.vt = 0
        self.tol = tol
        self.bounds = None

    def set_initial_point(self, bounds):
        self.bounds = bounds
        self.x = np.asarray([np.random.uniform(bi[0], bi[1]) for bi in bounds])
        self.vt = np.zeros(self.x.shape)


    def optimize(self, iterations):
        best_x = self.x
        best_score = self.function(self.x)
        for it in range(0, iterations):
            #print(best_score)
            gradient = self.function.derivative(self.x)
            for i in range(self.x.shape[0]):
                self.vt[i] = self.vt[i]*self.momentum + self.lr * gradient[i]
                self.x[i] = np.clip(self.x[i] - self.vt[i], self.bounds[i][0], self.bounds[i][1])

            c_score = self.function(self.x)
            if c_score < best_score:
                best_x = self.x
                best_score = c_score
        return best_x



class AdamOptimizer:
    def __init__(self, function, beta1=0.9, beta2=0.999, eps=1e-8, alpha=1e-2):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.alpha = alpha
        self.m = None
        self.v = None

        self.function = function
        self.x = None
        self.bounds = None


    def set_initial_point(self, bounds):
        self.bounds = bounds
        self.x = np.asarray([np.random.uniform(bi[0], bi[1]) for bi in bounds])
        self.m = [0.0 for _ in range(self.x.shape[0])]
        self.v = [0.0 for _ in range(self.x.shape[0])]


    def optimize(self, iterations, num_restarts):
        best_x = self.x
        best_score = self.function(self.x)
        #print(best_x)
        for r in range(num_restarts):
            for it in range(iterations):
                #print(best_score)
                gradient = self.function.derivative(self.x)
                for i in range(self.x.shape[0]):
                    self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * gradient[i]
                    self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (gradient[i]**2)

                    mhat = self.m[i] / (1.0 - self.beta1**(it+1))
                    vhat = self.v[i] / (1.0 - self.beta2**(it+1))

                    self.x[i] = self.x[i] - self.alpha * (mhat / (np.sqrt(vhat) - self.eps))
                    self.x[i] = np.clip(self.x[i], self.bounds[i][0], self.bounds[i][1])

                c_score = self.function(self.x)
                #print(c_score, self.x, gradient)
                if c_score < best_score:
                    best_x = self.x
                    best_score = c_score

            self.x = np.asarray([np.random.uniform(bi[0], bi[1]) for bi in self.bounds])
            self.m = [0.0 for _ in range(self.x.shape[0])]
            self.v = [0.0 for _ in range(self.x.shape[0])]

            return best_x, best_score






