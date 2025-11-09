module Rosenbrock

# Rosenbrock distribution — based on https://arxiv.org/abs/1903.09556
# Author: Ali Siahkoohi, alisk@ucf.edu
# Date: Nov 2025

using DrWatson
using Random

# Utilities
include("./utils/savefig.jl")

# Core functionalities
include("./distribution/rosenbrock.jl")

end