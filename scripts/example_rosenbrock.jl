using DrWatson
@quickactivate "Rosenbrock"

using Rosenbrock
using PyPlot

# Set up parameters using DrWatson's @dict macro
params = @dict μ=0.0f0 a=1.0f0 n_samples=5000

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

# Extract x and y coordinates
x = samples[1, 1, 1, :]
y = samples[1, 1, 2, :]

# Create figure
fig = figure(figsize=(12, 5))

# Scatter plot
subplot(1, 2, 1)
scatter(x, y, alpha=0.3, s=10, c="blue")
xlabel("x₁")
ylabel("x₂")
title("Rosenbrock Distribution Samples")
grid(true, alpha=0.3)

# 2D histogram
subplot(1, 2, 2)
plt.hist2d(x, y, bins=50, cmap="viridis")
colorbar(label="Density")
xlabel("x₁")
ylabel("x₂")
title("Sample Density")

tight_layout()

# Save plot using DrWatson
wsave(plotsdir("rosenbrock_samples.png"), fig)
close()

println("Results saved to: ", datadir("sims"))
println("Plot saved to: ", plotsdir("rosenbrock_samples.png"))
println("Generated $(params[:n_samples]) samples")
println("Log-pdf range: ", extrema(lp))
