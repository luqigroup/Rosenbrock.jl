using DrWatson
@quickactivate "Rosenbrock"

using Rosenbrock
using Plots

# Set up parameters using DrWatson's @dict macro
params = @dict μ=0.0f0 a=1.0f0 n_samples=1000

# Create Rosenbrock distribution
rb = RosenbrockDistribution(params[:μ], params[:a])

# Generate samples
samples = rand(rb, params[:n_samples])

# Compute log-pdf and gradient
lp = logpdf(rb, samples)
grad = gradlogpdf(rb, samples)

# Save results using DrWatson
data_dict = @dict samples lp grad params
safesave(datadir("sims", savename(params, "jld2")), data_dict)

# Plot samples
p = scatter(
    samples[1, 1, 1, :],
    samples[1, 1, 2, :],
    xlabel="x₁",
    ylabel="x₂",
    title="Rosenbrock Distribution Samples",
    legend=false,
    alpha=0.5
)

# Save plot
safesave(plotsdir("rosenbrock_samples.png"), p)

println("Results saved to: ", datadir("sims"))
println("Plot saved to: ", plotsdir())