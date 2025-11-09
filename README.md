# Rosenbrock.jl

A Julia implementation of the 2D Rosenbrock distribution for testing sampling algorithms.

Based on [Hoffman & Ma (2019)](https://arxiv.org/abs/1903.09556).

## Installation
```julia
] add /path/to/Rosenbrock
```

## Usage
```julia
using Rosenbrock

# Create distribution
rb = RosenbrockDistribution(0.0f0, 1.0f0)

# Generate samples
samples = rand(rb, 1000)

# Compute log-pdf and gradient
lp = logpdf(rb, samples)
grad = gradlogpdf(rb, samples)
```

## API

- `RosenbrockDistribution(μ, a)` - Create distribution with location `μ` and scale `a`
- `rand(rb, n)` - Generate `n` samples
- `logpdf(rb, X)` - Compute log probability density
- `gradlogpdf(rb, X)` - Compute gradient of log probability

## Author

Ali Siahkoohi (alisk@ucf.edu), University of Central Florida, 2025

## License

MIT