from collections import defaultdict
import argparse
import numpy as np
from tqdm import tqdm

from io_utils import load_matrices_from_dir
from pcg import pcg, Solution


def run_experiment(matrix_set: list[str], n_runs=1, maxiter=10000, tol=1e-6,
                   error_iter=None, error_pos=None) -> dict[str, list[Solution]]:
    """ Runs PCG over given matrices with given params describing the experiment process """
    raw_fps = [s + ".mat" for s in matrix_set]
    precond_fps = [s + "_precond.mat" for s in matrix_set]

    mats = load_matrices_from_dir("./matrices/raw", lambda m: m["Problem"]["A"][0][0], raw_fps)
    preconditioners = load_matrices_from_dir("./matrices/preconditioners", lambda m: m["L"], precond_fps)

    inject_error = error_iter is not None
    sols = defaultdict(list)
    for mat_name, A, L in zip(mats.keys(), mats.values(), preconditioners.values()):
        LT = L.T.tocsc()
        for _ in tqdm(range(n_runs), desc=f"Solving {mat_name[15:]}"):
            ep = error_pos if error_pos is not None else np.random.randint(A.shape[0])
            b = A @ np.ones((A.shape[0], 1))
            sol = pcg(A, b, tol, maxiter, L, LT, inject_error, ep, error_iter)
            sols[mat_name].append(sol)

    return sols


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs a PCG solver over a set of matrices possibly with errors.")
    parser.add_argument("--subset", nargs="+", help="Subset of matrices to solve")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of solves per matrix")
    parser.add_argument("--maxiter", type=int, default=10000, help="Maximum number of solver iterations")
    parser.add_argument("--tol", type=float, default=1e-6, help="Tolerance of the residual to be considered done")
    parser.add_argument("--error_iter", type=int, default=None, help="Solver iteration to inject error")
    parser.add_argument("--error_pos", type=int, default=None, help="Position in p vector to inject error")
    args = parser.parse_args()

    sols = run_experiment(args.subset, args.n_runs, args.maxiter, args.tol, args.error_iter, args.error_pos)
    print("matrix_name, final_iteration, final_relres, converged")
    for name, sol_list in sols.items():
        for sol in sol_list:
            print(f"{name}, {sol.final_iteration}, {sol.final_relres}, {sol.converged}")
