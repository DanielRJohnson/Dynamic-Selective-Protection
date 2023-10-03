import numpy as np
from numpy import ndarray
from scipy.linalg import norm
from scipy.sparse.linalg import spsolve_triangular
from dataclasses import dataclass


@dataclass
class Solution:
    """ Output Solution object holding a solved x, final relres, and convergence_iteration """
    x: ndarray
    final_relres: float
    final_iteration: int
    converged: bool


def pcg(A: ndarray, b: ndarray, tol: float, maxit: int, M1: ndarray, M2: ndarray,
        inject_error=False, error_pos=0, error_iter=0) -> Solution:
    """
    Solves a system of linear equations of the form Ax=b where the input matrix
    A is symmetric, positive definite, and possibly very large and sparse.
    https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method

    :param A: Input matrix A
    :param b: Output vector b
    :param tol: tolerance of solution to break early if |r|/|b| < tol
    :param maxit: maximum number of iterations for the solver to make
    :param M1: M1 factor of the preconditioner matrix M s.t. M1M2 = M
    :param M2: M2 factor of the preconditioner matrix M s.t. M2M2 = M
    :param inject_error: whether or not to inject an error
    :param error_pos: The position of p to inject the error
    :param error_iter: The iteration in which to inject the error
    :return: Solution
    """
    N = A.shape[0]
    x = np.zeros((N, 1))  # Nx1
    r = b - A @ x  # Nx1

    z = spsolve_triangular(M2, spsolve_triangular(M1, r), lower=False)  # Nx1  (z = M^(-1)r => z = (M2\(M1\r))
    p = z  # Nx1

    norm_r, norm_b = norm(r), norm(b)
    relres = norm_r / norm_b
    i = 0

    for i in range(maxit):
        if relres < tol:
            break

        if inject_error and i == error_iter:
            p[error_pos] = p[error_pos] + p.max()

        q = A @ p  # Nx1
        v = z.T @ r  # 1x1
        alpha = v / (p.T @ q)  # 1x1
        x += alpha * p  # Nx1
        r -= alpha * q  # Nx1
        z = spsolve_triangular(M2, spsolve_triangular(M1, r), lower=False)  # Nx1
        beta = (z.T @ r) / v  # 1x1
        p = z + beta * p  # Nx1

        norm_r = norm(r)
        relres = norm_r / norm_b

    return Solution(x, relres, i, relres <= tol)
