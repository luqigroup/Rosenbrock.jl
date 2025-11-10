export RosenbrockDistribution, logpdf, gradlogpdf

import Random: rand
import Base: rand

"""
    RosenbrockDistribution{T<:AbstractFloat}(μ, a)

2-dimensional Rosenbrock distribution, based on
https://arxiv.org/abs/1903.09556

Parameters:
- μ: mean parameter
- a: scaling parameter
"""
struct RosenbrockDistribution{T<:AbstractFloat}
    μ::T
    a::T
end

# Convenience constructor
RosenbrockDistribution(μ::Real, a::Real) = RosenbrockDistribution(promote(float(μ), float(a))...)

"""
    rand(RB::RosenbrockDistribution, n_samples::Int)

Sample from the 2D Rosenbrock distribution
Returns a 2×n_samples matrix
"""
function rand(RB::RosenbrockDistribution{T}, n_samples::Int) where T
    x1 = randn(T, n_samples) / sqrt(2 * RB.a) .+ RB.μ
    x2 = randn(T, n_samples) / sqrt(T(2)) .+ x1.^2
    return vcat(x1', x2')  # 2×n_samples matrix
end

"""
    logpdf(RB::RosenbrockDistribution, X::AbstractMatrix)

Compute log-pdf for samples X (2×n_samples matrix)
"""
function logpdf(RB::RosenbrockDistribution{T}, X::AbstractMatrix{T}) where T
    @assert size(X, 1) == 2 "X must have 2 rows (dimensions)"

    x1, x2 = X[1, :], X[2, :]
    log_pdf = -RB.a * (x1 .- RB.μ).^2 .- (x2 .- x1.^2).^2

    # Normalization constant
    log_norm = log(sqrt(RB.a / π))
    return log_pdf .+ log_norm
end

"""
    gradlogpdf(RB::RosenbrockDistribution, X::AbstractMatrix)

Compute gradient of log-pdf with respect to X (2×n_samples matrix)
Returns a 2×n_samples matrix
"""
function gradlogpdf(RB::RosenbrockDistribution{T}, X::AbstractMatrix{T}) where T
    @assert size(X, 1) == 2 "X must have 2 rows (dimensions)"

    x1, x2 = X[1, :], X[2, :]

    # ∂log p/∂x1
    grad_x1 = -2 * RB.a * (x1 .- RB.μ) .+ 4 * x1 .* (x2 .- x1.^2)

    # ∂log p/∂x2
    grad_x2 = -2 * (x2 .- x1.^2)

    return vcat(grad_x1', grad_x2')  # 2×n_samples matrix
end