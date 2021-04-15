from ctypes import CDLL, c_float, c_int
import sys

#functions = CDLL(sys.path[0] + '/Separable.so')
functions = CDLL(sys.path[0] + '/GlobalOptimisation/Separable/Separable.so')

#Define the restypes for all C functions

functions.ackley.restype = c_float
functions.powersum.restype = c_float
functions.quartic.restype = c_float
functions.rastrigin.restype = c_float
functions.schwefel220.restype = c_float
functions.schwefel221.restype = c_float
functions.schwefel222.restype = c_float
functions.shubert3.restype = c_float
functions.shubert4.restype = c_float
functions.sphere.restype = c_float
functions.styblinski_tank.restype = c_float
functions.sumsquares.restype = c_float
functions.xin_she_yang.restype = c_float
functions.zakharov.restype = c_float

class Ackley:
    def __init__(self, params, a, b, c):
        self.parameters = params;
        self.a = a
        self.b = b
        self.c = c

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.ackley((c_float * len(x))(*x), c_int(len(x)),
                                c_float(self.a), c_float(self.b),
                                c_float(self.c))



class PowerSum:
    def __init__(self, params):
        self.parameters = params;

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.powersum((c_float * len(x))(*x), c_int(len(x)))


class Quartic:
    def __init__(self, params, eps):
        self.parameters = params;
        self.eps = eps

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.quartic((c_float * len(x))(*x), c_int(len(x)),
                                 c_float * len(self.eps)(*self.eps))


class Rastrigin:
    def __init__(self, params):
        self.parameters = params;

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.rastrigin((c_float * len(x))(*x), c_int(len(x)))


class Schwefel220:
    def __init__(self, params):
        self.parameters = params;

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.schwefel220((c_float * len(x))(*x), c_int(len(x)))


class Schwefel221:
    def __init__(self, params):
        self.parameters = params;

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.schwefel221((c_float * len(x))(*x), c_int(len(x)))


class Schwefel222:
    def __init__(self, params):
        self.parameters = params;

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.schwefel222((c_float * len(x))(*x), c_int(len(x)))


class Schwefel223:
    def __init__(self, params):
        self.parameters = params;

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.schwefek223((c_float * len(x))(*x), c_int(len(x)))


class Shubert3:
    def __init__(self, params):
        self.parameters = params;

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.shubert3((c_float * len(x))(*x), c_int(len(x)))


class Shubert4:
    def __init__(self, params):
        self.parameters = params;

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.shubert4((c_float * len(x))(*x), c_int(len(x)))


class Sphere:
    def __init__(self, params):
        self.parameters = params;

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.sphere((c_float * len(x))(*x), c_int(len(x)))



class Styblinski_tank:
    def __init__(self, params):
        self.parameters = params;

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.styblinski_tank((c_float * len(x))(*x), c_int(len(x)))


class SumSquares:
    def __init__(self, params):
        self.parameters = params;

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.sumsquares((c_float * len(x))(*x), c_int(len(x)))


class Xin_she_yang:
    def __init__(self, params, eps):
        self.parameters = params;
        self.eps = eps

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.xin_she_yang((c_float * len(x))(*x), c_int(len(x)),
                                      (c_float * len(self.eps))(*self.eps))


class Zakharov:
    def __init__(self, params):
        self.parameters = params;

    def params(self):
        return self.parameters

    def __call__(self, x):
        return functions.zakharov((c_float * len(x))(*x), c_int(len(x)))
