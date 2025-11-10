# Rosenbrock.jl

A Julia implementation of the 2D Rosenbrock distribution for testing sampling algorithms.

Based on [Hoffman & Ma (2019)](https://arxiv.org/abs/1903.09556).

## Installation

Make sure that you first have ``matplotlib`` installed in your Python environment, as it is required for some plotting functionalities.

```bash
pip install matplotlib
```

Then in Julia:
```julia
ENV["PYTHON"] = "/usr/bin/python3"  # Adjust path
using Pkg
Pkg.build("PyCall")
# Restart Julia
```

```julia
] add /path/to/Rosenbrock
```

## Usage

```julia
using Rosenbrock
using Distributions

# Create distribution
rb = RosenbrockDistribution(0.0f0, 1.0f0)

# Generate samples
samples = rand(rb, 1000)  # Returns 2×1000 matrix

# Compute log-pdf and gradient
lp = logpdf(rb, samples)  # Extends Distributions.logpdf
grad = gradlogpdf(rb, samples)

println("Sample shape: ", size(samples))  # (2, 1000)
println("Log-pdf shape: ", size(lp))      # (1000,)
println("Gradient shape: ", size(grad))   # (2, 1000)
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