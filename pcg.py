from os.path import dirname
from scipy.sparse import find

from juliacall import Main as jl
from juliacall import Pkg as jlPkg
jlPkg.add("SparseArrays")
jl.seval("using SparseArrays")
jlPkg.activate(dirname(__file__) + "/Pcg")
jl.seval("using Pcg")


class Pcg:
    """ Wrapper class that handles Python-Julia interop to Pcg package """

    def __init__(self, A, b, M1, M2, tol, maxiter) -> None:
        """ Initializes Julia packages and converts complex inputs to Julia types """
        self.b = self.__convert_numpy(b)
        self.A, self.M1, self.M2 = [self.__convert_scipy(m) for m in [A, M1, M2]]
        self.tol, self.maxiter = tol, maxiter

    def pcg(self, inject_error=False, error_pos=0, error_iter=0) -> tuple:
        """ Calls Pcg.pcg and wraps response in a Solution object """
        return jl.pcg(self.A, self.b, self.M1, self.M2, self.tol,
                      self.maxiter, inject_error, error_pos, error_iter)

    def __call__(self, *args, **kwargs):
        """ Call shorthand to pcg method """
        return self.pcg(*args, **kwargs)

    def __convert_scipy(self, A):
        """ Converts a scipy sparse matrix to a Julia SparseMatrixCSC """
        I, J, V = map(lambda v: jl.Vector(v), find(A))
        inc = jl.seval("x -> x .+ 1")  # change to 1-indexed
        return jl.sparse(inc(I), inc(J), V)

    def __convert_numpy(self, x):
        """ Converts a numpy array to a Julia Matrix """
        return jl.Matrix(x)
