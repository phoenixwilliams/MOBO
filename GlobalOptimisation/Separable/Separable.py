from ctypes import CDLL, c_float, c_int

functions = CDLL("./Separable.so")


class Ackley:
    def __init__(self, dimension, a, b, c):
        self.dimension = dimension
        self.c_dim = c_int(dimension)
        self.a = c_float(a)
        self.b = c_float(b)
        self.c = c_float(c)
        self.problem = functions.ackley
        self.problem.restype = c_float

    def __call__(self, x):
        floatarray = c_float * self.dimension
        return self.problem(floatarray(*x), self.c_dim,
                            self.a, self.b, self.c)


class PowerSum:
    def __init__(self, dimension):
        self.problem = functions.powersum
        self.problem.restype = c_float
        self.dimension = dimension
        self.c_dim = c_int(dimension)

    def __call__(self, x):
        floatarray = c_float * self.dimension
        return self.problem(floatarray(*x), self.c_dim)


class Quartic:
    def __init__(self, dimension, eps):
        self.problem = functions.quartic
        self.problem.restype = c_float
        self.dimension = dimension
        self.c_dim = c_int(dimension)

        self.eps = c_float(eps)

    def __call__(self, x):
        floatarray = c_float * self.dimension
        return self.problem(floatarray(*x), self.c_dim, self.eps)


class Rastrigin:
    def __init__(self, dimension):
        self.problem = functions.rastrigin
        self.problem.restype = c_float
        self.dimension = dimension
        self.c_dim = c_int(dimension)

    def __call__(self, x):
        floatarray = c_float * self.dimension
        return self.problem(floatarray(*x), self.c_dim)


class Schwefel220:
    def __init__(self, dimension):
        self.problem = functions.schwefel220
        self.problem.restype = c_float
        self.dimension = dimension
        self.c_dim = c_int(dimension)

    def __call__(self, x):
        floatarray = c_float * self.dimension
        return self.problem(floatarray(*x), self.c_dim)


class Schwefel221:
    def __init__(self, dimension):
        self.problem = functions.schwefel221
        self.problem.restype = c_float
        self.dimension = dimension
        self.c_dim = c_int(dimension)

    def __call__(self, x):
        floatarray = c_float * self.dimension
        return self.problem(floatarray(*x), self.c_dim)


class Schwefel222:
    def __init__(self, dimension):
        self.problem = functions.schwefel222
        self.problem.restype = c_float
        self.dimension = dimension
        self.c_dim = c_int(dimension)

    def __call__(self, x):
        floatarray = c_float * self.dimension
        return self.problem(floatarray(*x), self.c_dim)


class Shubert3:
    def __init__(self, dimension):
        self.problem = functions.shubert3
        self.problem.restype = c_float
        self.dimension = dimension
        self.c_dim = c_int(dimension)

    def __call__(self, x):
        floatarray = c_float * self.dimension
        return self.problem(floatarray(*x), self.c_dim)


class Shubert4:
    def __init__(self, dimension):
        self.problem = functions.shubert4
        self.problem.restype = c_float
        self.dimension = dimension
        self.c_dim = c_int(dimension)

    def __call__(self, x):
        floatarray = c_float * self.dimension
        return self.problem(floatarray(*x), self.c_dim)


class Sphere:
    def __init__(self, dimension):
        self.problem = functions.sphere
        self.problem.restype = c_float
        self.dimension = dimension
        self.c_dim = c_int(dimension)

    def __call__(self, x):
        floatarray = c_float * self.dimension
        return self.problem(floatarray(*x), self.c_dim)


class Styblinski_tank:
    def __init__(self, dimension):
        self.problem = functions.styblinski_tank
        self.problem.restype = c_float
        self.dimension = dimension
        self.c_dim = c_int(dimension)

    def __call__(self, x):
        floatarray = c_float * self.dimension
        return self.problem(floatarray(*x), self.c_dim)


class SumSquares:
    def __init__(self, dimension):
        self.problem = functions.sumsquares
        self.problem.restype = c_float
        self.dimension = dimension
        self.c_dim = c_int(dimension)

    def __call__(self, x):
        floatarray = c_float * self.dimension
        return self.problem(floatarray(*x), self.c_dim)


class Xin_She_Yang:
    def __init__(self, dimension, eps):
        self.problem = functions.xin_she_yang
        self.problem.restype = c_float
        self.dimension = dimension
        self.c_dim = c_int(dimension)

        floatarray = c_float * dimension
        self.eps = floatarray(*eps)

    def __call__(self, x):
        floatarray = c_float * self.dimension
        return self.problem(floatarray(*x), self.c_dim, self.eps)



