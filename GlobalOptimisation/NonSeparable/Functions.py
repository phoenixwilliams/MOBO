from ctypes import CDLL, c_float, c_int, c_double
import sys

#functions = CDLL(sys.path[0] + '/NonSeparable.so')
functions = CDLL(sys.path[0] + '/GlobalOptimisation/NonSeparable/NonSeparable.so')

functions.ackleyN4.restype = c_float
functions.alpineN2.restype = c_float
functions.alpineN1.restype = c_float
functions.brown.restype = c_float
functions.exponential.restype = c_double
functions.griewank.restype = c_float
functions.happycat.restype = c_float
functions.periodic.restype = c_float
functions.qing.restype = c_float
functions.ridge.restype = c_float
functions.rosenbrock.restype = c_double
functions.salomon.restype = c_float
functions.schwefel.restype = c_float
functions.shubert.restype = c_float
functions.xin_she_yangN2 = c_float
functions.xin_she_yangN4 = c_float


class AckleyN4:
    def __init__(self, params):
        self.parameters = params;

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.ackleyN4((c_float * len(x))(*x), c_int(len(x)))


class AlpineN2:
    def __init__(self, params):
        self.parameters = params;

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.alpineN2((c_float * len(x))(*x), c_int(len(x)))


class AlpineN1:
    def __init__(self, params):
        self.parameters = params;

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.alpineN1((c_float * len(x))(*x), c_int(len(x)))


class Brown:
    def __init__(self, params):
        self.parameters = params;


    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.brown((c_float * len(x))(*x), c_int(len(x)))


class Exponential:
    def __init__(self, params):
        self.parameters = params;

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.exponential((c_float * len(x))(*x), c_int(len(x)))


class Griewank:
    def __init__(self, params):
        self.parameters = params;

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.griewank((c_float * len(x))(*x), c_int(len(x)))


class HappyCat:
    def __init__(self, params, alpha):
        self.parameters = params;
        self.alpha = alpha

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.happycat((c_float * len(x))(*x), c_int(len(x)),
                                  c_float(self.alpha))


class Periodic:
    def __init__(self, params):
        self.parameters = params;

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.periodic((c_float * len(x))(*x), c_int(len(x)))


class Qing:
    def __init__(self, params):
        self.parameters = params;

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.qing((c_float * len(x))(*x), c_int(len(x)))


class Ridge:
    def __init__(self, params, d, alpha):
        self.parameters = params;
        self.d = d
        self.alpha = alpha

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.ridge((c_float * len(x))(*x), c_int(len(x)),
                               c_float(self.d), c_float(self.alpha))


class Rosenbrock:
    def __init__(self, params, a, b):
        self.parameters = params;
        self.a = a
        self.b = b

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.rosenbrock((c_float * len(x))(*x), c_int(len(x)),
                                    c_float(self.a), c_float(self.b))


class Salomon:
    def __init__(self, params):
        self.parameters = params;

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.salomon((c_float * len(x))(*x), c_int(len(x)))


class Schwefel:
    def __init__(self, params):
        self.parameters = params;

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.schwefel((c_float * len(x))(*x), c_int(len(x)))


class Shubert:
    def __init__(self, params):
        self.parameters = params;

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.shubert((c_float * len(x))(*x), c_int(len(x)))


class Xin_She_yangN2:
    def __init__(self, params):
        self.parameters = params;

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.xin_she_yangN2((c_float * len(x))(*x), c_int(len(x)))


class Xin_She_yangN3:
    def __init__(self, params, m, beta):
        self.parameters = params
        self.m = m
        self.beta = beta

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.xin_she_yangN3((c_float * len(x))(*x), c_int(len(x)),
                                        c_float(self.m), c_float(self.beta))


class Xin_She_yangN4:
    def __init__(self, params):
        self.parameters = params;

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.xin_she_yangN4((c_float * len(x))(*x), c_int(len(x)))

