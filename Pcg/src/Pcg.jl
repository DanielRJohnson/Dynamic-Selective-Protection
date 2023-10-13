module Pcg

using SparseArrays
using LinearAlgebra

"""
Solves a system of linear equations of the form Ax=b where the input matrix
A is symmetric, positive definite, and possibly very large and sparse.
M1M2 is a decomposed preconditioner matrix M.
Returns (solution x, final relres, final iteration, whether convergence was reached, time(s))
https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method

# Options
`tol::Float64` tolerance of solution to break early if |r|/|b| < tol
`maxit::Int` maximum number of iterations for the solver to make
`inject_error::Bool` whether or not to inject an error
`error_pos::Int` The position of p to inject the error
`error_iter::Int` The iteration in which to inject the error
"""
function pcg(A::SparseMatrixCSC, b::Matrix{Float64}, M1::SparseMatrixCSC, M2::SparseMatrixCSC,
    tol::Float64, maxit::Int, inject_error=false, error_pos=1,
    error_iter=0)::Tuple{Int64, Float64, Bool, Float64, Matrix{Float64}}

    time = @elapsed begin
        N = A.m
        x = zeros(N, 1)  # Nx1
        r = b - A * x  # Nx1
        z = M2 \ (M1 \ r)  # Nx1
        p = z  #Nx1

        norm_b = norm(b)
        relres = norm(r) / norm_b

        i = 0
        while i < maxit && relres > tol
            if inject_error && (i == error_iter)
                p[error_pos] += maximum(p)
            end

            q = A * p # Nx1
            v = transpose(z) * r # 1x1
            alpha = v / (transpose(p) * q) # 1x1
            x += alpha .* p # Nx1
            r -= alpha .* q #Nx1
            z = M2 \ (M1 \ r) # Nx1
            beta = (transpose(z) * r) / v # 1x1
            p = z + beta .* p # Nx1

            relres = norm(r) / norm_b
            i += 1
        end # while
    end # elapsed

    return (i - 1, relres, relres <= tol, time, x)
end #function

export pcg

end # module