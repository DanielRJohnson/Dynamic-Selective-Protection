module Pcg

using SparseArrays
using LinearAlgebra
using ProgressMeter
using PythonCall

"""
A PcgResult is of the following form:
(solve_iterations, final_relres, did_converge, realtime_s, final_x)
"""
const PcgResult = Tuple{Int64,Float64,Bool,Float64,Matrix{Float64}}

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
`protections::Matrix{Bool}` Protection scheme describing which iter-pos pairs to protect
"""
function _pcg(A::SparseMatrixCSC, b::Matrix{Float64}, M1::SparseMatrixCSC, M2::SparseMatrixCSC,
    tol::Float64, maxit::Int, error_pos=1, error_iter=nothing)::PcgResult

    # convert M1, M2 to UmfpackLU for efficient, non-allocating solving
    M1_umf = SparseArrays.UMFPACK.UmfpackLU(M1)
    M2_umf = SparseArrays.UMFPACK.UmfpackLU(M2)

    time = @elapsed begin
        N = A.m
        x = zeros(N, 1)  # Nx1
        r = b - A * x  # Nx1
        z = M2_umf \ (M1_umf \ r)  # Nx1
        p = copy(z)  # Nx1

        q = zeros(N, 1)  # Nx1, prealloc
        M1_solve_r = zeros(N, 1)  # Nx1, prealloc

        norm_b = norm(b)
        relres = norm(r) / norm_b

        i = 0
        while i < maxit && relres > tol
            if !isnothing(error_iter) && (i == error_iter)
                p[error_pos] += maximum(p)
            end

            mul!(q, A, p)  # q = A * p, nonallocating
            v = transpose(z) * r  # 1x1
            alpha = v / (transpose(p) * q)  # 1x1

            x .+= alpha .* p # Nx1
            r .-= alpha .* q #Nx1

            ldiv!(M1_solve_r, M1_umf, r)  # z = M^-1*r = M2 \ (M1 \ r)
            ldiv!(z, M2_umf, M1_solve_r)  # nonallocating

            beta = (transpose(z) * r) / v # 1x1
            p .= z .+ beta .* p

            relres = norm(r) / norm_b
            i += 1
        end
    end

    return (i - 1, relres, relres <= tol, time, x)
end


"""
Runs one Pcg job over (A, b, M1, M2)
See _pcg for options that can be specified.
"""
function pcg(A, b, M1, M2, opts)
    return _pcg(A, b, M1, M2, opts...)
end


"""
Runs multiple Pcg jobs over (A, b, M1, M2), one for each batch_job.
See _pcg for options that can be specified in a batch_job.
"""
function batched_pcg(A, b, M1, M2, batch_job)
    return @time @showprogress map((job) -> (_pcg(A, b, M1, M2, job...)), batch_job)
end


export pcg, batched_pcg

end
