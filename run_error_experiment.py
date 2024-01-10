from os.path import dirname
from argparse import ArgumentParser
from dataclasses import dataclass

from numpy import ones, reshape
from numpy.random import randint

from pcg import pcg, batched_pcg, PcgOptions, PcgBatchJob, PcgInput
from io_utils import load_matrices_from_dir, write_results_csv, load_models


@dataclass
class ExperimentOptions:
    """All options for running an experiment"""
    matrix_set: list[str]  # ex: ["cbuckle", "bcsstk18"]
    n_runs: int  # ex: 100
    tol: float  # ex: 0.00001
    error_iterations: list[int]  # ex: [1, 50, 100]
    model_paths: list[str]  # ex: ["./models/cbuckle/best_XGBRegressor.pkl"]
    ps: list[float]  # ex: [0.05]


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
    parser.add_argument("--model_paths", type=str, nargs="*", default=[],
                        help="Model paths to build protection schemes from (should match length with 'subset').")
    parser.add_argument("--ps", type=float, nargs="*", default=[],
                        help="Probabilities to build protection scheme from (should match length with 'subset')")
    args = parser.parse_args()
    return ExperimentOptions(args.subset, args.n_runs, args.tol,
                             args.error_iterations, args.model_paths, args.ps)


def run_experiment(opts: ExperimentOptions) -> None:
    """Carries out a Pcg experiment as described in the given ExperimentOptions"""
    mats = load_matrices_from_dir(
        dirname(__file__) + "/matrices/raw", opts.matrix_set)
    preconditioners = load_matrices_from_dir(
        dirname(__file__) + "/matrices/preconditioners", opts.matrix_set)
    models = load_models(opts.model_paths) if \
        len(opts.model_paths) else [None] * len(mats)
    ps = opts.ps if len(opts.ps) else [None] * len(mats)

    for mat_name, A, L, model, p in zip(mats.keys(), mats.values(), preconditioners.values(), models, ps):
        b = A @ ones((A.shape[0], 1))  # fixed right hand side
        pcg_input = PcgInput(A, b, L, L.T)
        errorfree_opts = PcgOptions(tol=opts.tol)
        errorfree_iters = pcg(pcg_input, errorfree_opts).solve_iterations
        maxiter = errorfree_iters * 100

        iter_pos_pairs = reshape([[i, pos] for pos in range(len(b))
                                  for i in range(errorfree_iters)], (-1, 2))
        protections = (model.predict(iter_pos_pairs).reshape(errorfree_iters, len(b)) > (1 + (1 / p))
                       if model is not None and p is not None else None)

        pcg_runs = []
        for _ in range(opts.n_runs):
            pos = randint(1, A.shape[0] + 1)  # one-based indexing for Julia
            error_indices = opts.error_iterations if len(opts.error_iterations) \
                else [randint(1, errorfree_iters)]  # default is one random iteration
            for i in error_indices:
                pcg_runs.append(PcgOptions(
                    tol=opts.tol, maxiter=maxiter, error_pos=pos, error_iter=i))
                if protections is not None:
                    pcg_runs.append(PcgOptions(tol=opts.tol, maxiter=maxiter, error_pos=pos,
                                               error_iter=i, protections=protections))

        inp = PcgInput(A, b, L, L.T)
        job = PcgBatchJob(inp, pcg_runs)
        results = batched_pcg(job)
        write_results_csv(mat_name, A, errorfree_iters, pcg_runs, results)


def main():
    opts = get_options_from_args()
    run_experiment(opts)


if __name__ == "__main__":
    main()
