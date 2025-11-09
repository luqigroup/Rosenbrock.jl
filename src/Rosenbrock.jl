module Rosenbrock

# Rosenbrock distribution — based on https://arxiv.org/abs/1903.09556
# Author: Ali Siahkoohi, alisk@rice.edu
# Date: Feb 2023
# Copyright: Rice University

export RosenbrockDistribution, logpdf, gradlogpdf

import Random: rand
import Base: rand

"""
    RosenbrockDistribution(μ, a)

2-dimensional Rosenbrock distribution, based on
https://arxiv.org/abs/1903.09556
"""
struct RosenbrockDistribution
    μ::Float32
    a::Float32
end


"""
    rand(RB::RosenbrockDistribution, n_samples::Int64)

2D Rosenbrock distribution sampler
"""
function rand(RB::RosenbrockDistribution, n_samples::Int64)
    X = zeros(Float32, 1, 1, 2, n_samples)
    X[1, 1, 1, :] = randn(Float32, n_samples) / sqrt(2.0f0 * RB.a) .+ RB.μ
    X[1, 1, 2, :] = randn(Float32, n_samples) / sqrt(2.0f0)
    X[1, 1, 2, :] += X[1, 1, 1, :] .^ 2.0f0
    return X
end


"""
    logpdf(RB::RosenbrockDistribution, X::AbstractArray{Float32,4})

2D Rosenbrock distribution log-pdf
"""
function logpdf(RB::RosenbrockDistribution, X::AbstractArray{Float32,4})
    log_pdf = -RB.a * (X[1, 1, 1, :] .- RB.μ) .^ 2.0f0
    log_pdf -= (X[1, 1, 2, :] - X[1, 1, 1, :] .^ 2.0f0) .^ 2.0f0
    return log_pdf .+ log(sqrt(RB.a) / 1.0f0π)
end


"""
    gradlogpdf(RB::RosenbrockDistribution, X::AbstractArray{Float32,4})

2D Rosenbrock distribution log-pdf gradient
"""
function gradlogpdf(RB::RosenbrockDistribution, X::AbstractArray{Float32,4})
    nlog_pdf_grad = cat(
        2.0f0 * RB.a * (X[:, :, 1:1, :] .- RB.μ) -
        4.0f0 * X[:, :, 1:1, :] .* (X[:, :, 2:2, :] - X[:, :, 1:1, :] .^ 2.0f0),
        2.0f0 * (X[:, :, 2:2, :] - X[:, :, 1:1, :] .^ 2.0f0);
        dims = 3,
    )
    return -nlog_pdf_grad
end

end # module
