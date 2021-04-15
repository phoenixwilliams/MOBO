from ctypes import CDLL, c_float, c_int

functions = CDLL("NonSeparable.so")


class AckleyN4:
    def __init__(self, dimension):
        self.dimension = dimension
        self.c_dim = c_int(dimension)
        self.problem = functions.ackleyN4
        self.problem.restype = c_float


    def __call__(self, x):
        floatarray = c_float * self.dimension
        return self.problem(floatarray(*x), self.c_dim)


class AlpineN2:
    def __init__(self, dimension):
        self.dimension = dimension
        self.c_dim = c_int(dimension)
        self.problem = functions.alpineN2
        self.problem.restype = c_float

    def __call__(self, x):
        floatarray = c_float * self.dimension
        return self.problem(floatarray(*x), self.c_dim)


class AlpineN1:
    def __init__(self, dimension):
        self.dimension = dimension
        self.c_dim = c_int(dimension)
        self.problem = functions.alpineN1
        self.problem.restype = c_float

    def __call__(self, x):
        floatarray = c_float * self.dimension
        return self.problem(floatarray(*x), self.c_dim)


class Brown:
    def __init__(self, dimension):
        self.dimension = dimension
        self.c_dim = c_int(dimension)
        self.problem = functions.brown
        self.problem.restype = c_float

    def __call__(self, x):
        floatarray = c_float * self.dimension
        return self.problem(floatarray(*x), self.c_dim)


class Exponential:
    def __init__(self, dimension):
        self.dimension = dimension
        self.c_dim = c_int(dimension)
        self.problem = functions.exponential
        self.problem.restype = c_float

    def __call__(self, x):
        floatarray = c_float * self.dimension
        return self.problem(floatarray(*x), self.c_dim)


class Griewank:
    def __init__(self, dimension):
        self.dimension = dimension
        self.c_dim = c_int(dimension)
        self.problem = functions.griewank
        self.problem.restype = c_float

    def __call__(self, x):
        floatarray = c_float * self.dimension
        return self.problem(floatarray(*x), self.c_dim)


class HappyCat:
    def __init__(self, dimension, alpha):
        self.dimension = dimension
        self.c_dim = c_int(dimension)
        self.problem = functions.happycat
        self.problem.restype = c_float
        self.alpha = c_float(alpha)

    def __call__(self, x):
        floatarray = c_float * self.dimension
        return self.problem(floatarray(*x), self.c_dim, self.alpha)


class Periodic:
    def __init__(self, dimension):
        self.dimension = dimension
        self.c_dim = c_int(dimension)
        self.problem = functions.periodic
        self.problem.restype = c_float

    def __call__(self, x):
        floatarray = c_float * self.dimension
        return self.problem(floatarray(*x), self.c_dim)


class Qing:
    def __init__(self, dimension):
        self.dimension = dimension
        self.c_dim = c_int(dimension)
        self.problem = functions.qing
        self.problem.restype = c_float

    def __call__(self, x):
        floatarray = c_float * self.dimension
        return self.problem(floatarray(*x), self.c_dim)


class Ridge:
    def __init__(self, dimension):
        self.dimension = dimension
        self.c_dim = c_int(dimension)
        self.problem = functions.ridge
        self.problem.restype = c_float

    def __call__(self, x):
        floatarray = c_float * self.dimension
        return self.problem(floatarray(*x), self.c_dim)


class Rosenbrock:
    def __init__(self, dimension, a, b, c):
        self.dimension = dimension
        self.c_dim = c_int(dimension)
        self.problem = functions.rosenbrock
        self.problem.restype = c_float
        self.a = c_float(a)
        self.b = c_float(b)
        self.c = c_float(c)

    def __call__(self, x):
        floatarray = c_float * self.dimension
        return self.problem(floatarray(*x), self.c_dim)


class Salomon:
    def __init__(self, dimension):
        self.dimension = dimension
        self.c_dim = c_int(dimension)
        self.problem = functions.salomon
        self.problem.restype = c_float

    def __call__(self, x):
        floatarray = c_float * self.dimension
        return self.problem(floatarray(*x), self.c_dim)


class Schwefel:
    def __init__(self, dimension):
        self.dimension = dimension
        self.c_dim = c_int(dimension)
        self.problem = functions.schwefel
        self.problem.restype = c_float

    def __call__(self, x):
        floatarray = c_float * self.dimension
        return self.problem(floatarray(*x), self.c_dim)


class Shubert:
    def __init__(self, dimension):
        self.dimension = dimension
        self.c_dim = c_int(dimension)
        self.problem = functions.shubert
        self.problem.restype = c_float

    def __call__(self, x):
        floatarray = c_float * self.dimension
        return self.problem(floatarray(*x), self.c_dim)


class Xin_She_Yang_N2:
    def __init__(self, dimension):
        self.dimension = dimension
        self.c_dim = c_int(dimension)
        self.problem = functions.xin_she_yangN2
        self.problem.restype = c_float

    def __call__(self, x):
        floatarray = c_float * self.dimension
        return self.problem(floatarray(*x), self.c_dim)


class Xin_She_Yang_N3:
    def __init__(self, dimension, m, beta):
        self.dimension = dimension
        self.c_dim = c_int(dimension)
        self.problem = functions.alpineN2
        self.problem.restype = c_float
        self.m = c_float(m)
        self.beta = c_float(beta)

    def __call__(self, x):
        floatarray = c_float * self.dimension
        return self.problem(floatarray(*x), self.c_dim, self.m, self.beta)


class Xin_She_Yang_N4:
    def __init__(self, dimension):
        self.dimension = dimension
        self.c_dim = c_int(dimension)
        self.problem = functions.xin_she_yangN4
        self.problem.restype = c_float

    def __call__(self, x):
        floatarray = c_float * self.dimension
        return self.problem(floatarray(*x), self.c_dim)


if __name__ == "__main__":
    import random
    import time
    d = 1000
    arr = [0 for _ in range(d)]
    eps = [random.random() for _ in range(d)]
    ackley = AlpineN2(d, eps)
    start = time.time()
    ackley(arr)
    print(time.time() - start)