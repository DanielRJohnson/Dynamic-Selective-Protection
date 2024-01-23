from os.path import dirname
from argparse import ArgumentParser
from dataclasses import dataclass

from numpy import ones, array
from numpy.random import randint
from scipy.sparse.linalg import norm
from pandas import DataFrame

from pcg import pcg, batched_pcg, PcgOptions, PcgBatchJob, PcgInput
from io_utils import load_matrices_from_dir, write_results_csv


@dataclass
class ExperimentOptions:
    """All options for running an experiment"""
    matrix_set: list[str]  # ex: ["cbuckle", "bcsstk18"]
    n_runs: int  # ex: 100
    tol: float  # ex: 0.00001
    error_iterations: list[int]  # ex: [1, 50, 100]


def get_options_from_args() -> ExperimentOptions:
    """Parses a ExperimentOptions from command line arguments"""
    parser = ArgumentParser(
        description="Runs a PCG solver over a set of matrices with errors.")
    parser.add_argument("--subset", nargs="+",
                        help="Subset of matrices to solve, names in /matrices (ex: cbuckle).")
    parser.add_argument("--n_runs", type=int, default=1,
                        help="Number of solves per matrix.")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="Tolerance of the residual to be considered done.")
    parser.add_argument("--error_iterations", type=int, nargs="+", default=[],
                        help="Manually specify multiple runs to happen at each fixed iteration (ex: 1 50 100)."
                             "If not specified, an error will be injected randomly for each run.")
    args = parser.parse_args()
    return ExperimentOptions(args.subset, args.n_runs, args.tol, args.error_iterations)


def run_experiment(opts: ExperimentOptions) -> None:
    """Carries out a Pcg experiment as described in the given ExperimentOptions"""
    mats = load_matrices_from_dir(
        dirname(__file__) + "/matrices/raw", opts.matrix_set)
    preconditioners = load_matrices_from_dir(
        dirname(__file__) + "/matrices/preconditioners", opts.matrix_set)

    for mat_name, A, L in zip(mats.keys(), mats.values(), preconditioners.values()):
        b = A @ ones((A.shape[0], 1))  # fixed right hand side
        pcg_input = PcgInput(A, b, L, L.T)
        errorfree_opts = PcgOptions(tol=opts.tol)
        errorfree_iters = pcg(pcg_input, errorfree_opts).solve_iterations
        maxiter = errorfree_iters * 100

        pcg_runs = []
        for _ in range(opts.n_runs):
            pos = randint(1, A.shape[0] + 1)  # one-based indexing for Julia
            error_indices = opts.error_iterations if len(opts.error_iterations) \
                else [randint(0, errorfree_iters)]  # default is one random iteration
            for i in error_indices:
                pcg_runs.append(PcgOptions(
                    tol=opts.tol, maxiter=maxiter, error_pos=pos, error_iter=i))

        inp = PcgInput(A, b, L, L.T)
        job = PcgBatchJob(inp, pcg_runs)
        results = batched_pcg(job)
        write_results_csv(mat_name, A, errorfree_iters, pcg_runs, results)


def main():
    opts = get_options_from_args()
    run_experiment(opts)


if __name__ == "__main__":
    main()
