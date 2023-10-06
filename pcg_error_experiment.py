import numpy as np
import pandas as pd
from scipy.sparse.linalg import norm
import argparse
from tqdm import tqdm

from pcg import Pcg
from io_utils import load_matrices_from_dir


def run_experiment(matrix_set: list[str], n_runs, maxiter, tol, error_perc, error_pos) -> list[pd.DataFrame]:
    """ Runs PCG over a given matrix_set n_runs times possibly with errors. """
    mats = load_matrices_from_dir("./matrices/raw", matrix_set)
    preconditioners = load_matrices_from_dir("./matrices/preconditioners", matrix_set)
    inject_error = error_perc is not None

    sols = []
    for mat_name, A, L in zip(mats.keys(), mats.values(), preconditioners.values()):
        sol = pd.DataFrame(columns=["matrix_name", "error_perc", "error_pos", "pos_2norm", "errorfree_iterations",
                                    "iterations", "final_relres", "converged", "time_s"])

        b = A @ np.ones((A.shape[0], 1))
        pcg = Pcg(A, b, L, L.T, tol, maxiter)

        errorfree_iters = pcg(inject_error=False)[0]
        error_iter = int(errorfree_iters * (error_perc / 100))

        for i in tqdm(range(n_runs), desc=f"Solving {mat_name[15:]}"):
            ep = error_pos if error_pos is not None else np.random.randint(1, A.shape[0] + 1)
            res = pcg(inject_error, ep, error_iter)
            sol.loc[i] = [mat_name, error_perc, ep, norm(A[ep]), errorfree_iters, *res[:-1]]

        sols.append(sol)

    return sols


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs a PCG solver over a set of matrices possibly with errors.")
    parser.add_argument("--subset", nargs="+", help="Subset of matrices to solve")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of solves per matrix")
    parser.add_argument("--maxiter", type=int, default=10000, help="Maximum number of solver iterations")
    parser.add_argument("--tol", type=float, default=1e-6, help="Tolerance of the residual to be considered done")
    parser.add_argument("--error_perc", type=float, default=None,
                        help="Percentage of error-free iterations to place error at")
    parser.add_argument("--error_pos", type=int, default=None, help="Position in p vector to place error at")
    args = parser.parse_args()

    sols = run_experiment(args.subset, args.n_runs, args.maxiter, args.tol, args.error_perc, args.error_pos)

    for name, sol in zip(args.subset, sols):
        fn = "analyses/data/" + "_".join([name, str(args.n_runs), str(args.error_perc), str(args.error_pos)]) + ".csv"
        sol.to_csv(fn, index=False)
