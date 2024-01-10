from os.path import dirname
from dataclasses import dataclass, asdict

from scipy.sparse import find, csr_matrix
from numpy import ndarray

from juliacall import Main as jl
jl.seval("using Pkg")
jl.seval(f'Pkg.activate("{dirname(__file__) + "/Pcg"}")')
jl.seval("using Pcg")
jl.seval("using SparseArrays")


@dataclass
class IterableDataclass:
    """Superclass to allow iteration on a dataclass"""

    def __iter__(self):
        """allow conversion to list, tuple, etc."""
        for _, v in asdict(self).items():
            yield (v)


@dataclass
class PcgInput(IterableDataclass):
    """
    All non-scalar inputs to Pcg. A and b represent the linear system
    to solve, and M1 and M2 are an LU factorization of the preconditioner.
    Note: types are converted to be Julia-compatable after initialization.
    """
    A: csr_matrix
    b: ndarray
    M1: csr_matrix
    M2: csr_matrix

    def __post_init__(self):
        """Convert types to be compatable with Julia conversions after initialization"""
        self.A, self.M1, self.M2 = [self.__convert_scipy(
            m) for m in [self.A, self.M1, self.M2]]
        self.b = self.__convert_numpy(self.b)

    def __convert_scipy(self, A):
        """Converts a scipy sparse matrix to a Julia SparseMatrixCSC"""
        I, J, V = map(lambda v: jl.Vector(v), find(A))
        inc = jl.seval("x -> x .+ 1")  # change to 1-indexed
        return jl.sparse(inc(I), inc(J), V)

    def __convert_numpy(self, x):
        """Converts a numpy array to a Julia Matrix"""
        return jl.Matrix(x)


@dataclass
class PcgOptions(IterableDataclass):
    """All options for Pcg"""
    tol: float = 1e-6
    maxiter: int = 2**63 - 1  # max Int64
    error_pos: int = 1
    error_iter: int = None
    protections: list[list[bool]] = None


@dataclass
class PcgBatchJob(IterableDataclass):
    """A batch job for Pcg. Runs Pcg over the PcgInput one time for each PcgOptions given"""
    pcg_input: PcgInput
    options_list: list[PcgOptions]


@dataclass
class PcgResult(IterableDataclass):
    """Result information given from Pcg"""
    solve_iterations: int
    final_relres: float
    did_converge: bool
    realtime_s: float
    final_x: ndarray


def pcg(inp: PcgInput, opts: PcgOptions) -> PcgResult:
    """Returns a PcgResult from calling Pcg on one PcgInput and PcgOptions"""
    return PcgResult(*jl.pcg(*inp, tuple(opts)))


def batched_pcg(job: PcgBatchJob) -> list[PcgResult]:
    """Returns one PcgResult for each job in the PcgBatchJob"""
    results = jl.batched_pcg(
        *job.pcg_input, [tuple(opts) for opts in job.options_list])
    return [PcgResult(*res) for res in results]
