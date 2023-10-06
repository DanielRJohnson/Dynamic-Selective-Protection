from os.path import dirname
import numpy as np
import pandas as pd
from scipy.sparse.linalg import norm
import argparse
from tqdm import tqdm

from pcg import Pcg
from io_utils import load_matrices_from_dir


def run_experiment(matrix_set: list[str], n_runs, maxiter, tol, error_percentages) -> list[pd.DataFrame]:
    """ Runs PCG over a given matrix_set n_runs times possibly with errors. """
    mats = load_matrices_from_dir(dirname(__file__) + "/matrices/raw", matrix_set)
    preconditioners = load_matrices_from_dir(dirname(__file__) + "/matrices/preconditioners", matrix_set)
    inject_error = len(error_percentages) != 0

    sols = []
    for mat_name, A, L in zip(mats.keys(), mats.values(), preconditioners.values()):
        sol = pd.DataFrame(columns=["matrix_name", "error_perc", "error_pos", "pos_2norm", "errorfree_iterations",
                                    "iterations", "final_relres", "converged", "time_s"])
        error_positions = np.random.randint(1, A.shape[0], (n_runs,))
        for error_perc in error_percentages if inject_error else [None]:
            b = A @ np.ones((A.shape[0], 1))
            pcg = Pcg(A, b, L, L.T, tol, maxiter)

            errorfree_iters = pcg(inject_error=False)[0]
            error_iter = int(errorfree_iters * (error_perc / 100)) if inject_error else -1

            for i in tqdm(range(n_runs), desc=f"Solving {mat_name} ({error_perc}% Io)"):
                res = pcg(inject_error, error_positions[i], error_iter)
                row = [mat_name, error_perc, error_positions[i],
                       norm(A[error_positions[i]]), errorfree_iters, *res[:-1]]
                sol = pd.concat([sol, pd.DataFrame([row], columns=sol.columns)], ignore_index=True)

        sols.append(sol)

    return sols


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs a PCG solver over a set of matrices possibly with errors.")
    parser.add_argument("--subset", nargs="+", help="Subset of matrices to solve")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of solves per matrix")
    parser.add_argument("--maxiter", type=int, default=10000, help="Maximum number of solver iterations")
    parser.add_argument("--tol", type=float, default=1e-6, help="Tolerance of the residual to be considered done")
    parser.add_argument("--error_percentages", type=float, nargs="+", default=[],
                        help="Percentages of error-free iterations to place errors at")
    args = parser.parse_args()

    sols = run_experiment(args.subset, args.n_runs, args.maxiter, args.tol, args.error_percentages)

    for name, sol in zip(args.subset, sols):
        fn = "analyses/data/" + "_".join([name, str(args.n_runs)]) + ".csv"
        sol.to_csv(fn, index=False)
