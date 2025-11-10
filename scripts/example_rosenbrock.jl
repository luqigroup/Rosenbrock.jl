using DrWatson
@quickactivate :Rosenbrock

using Distributions
using Rosenbrock
using PyPlot

# Set up parameters using DrWatson's @dict macro
params = @dict μ=0.0f0 a=1.0f0 n_samples=Int(5f4)

# Create Rosenbrock distribution
rb = RosenbrockDistribution(params[:μ], params[:a])

# Generate samples
samples = rand(rb, params[:n_samples])

# Compute log-pdf and gradient
lp = logpdf(rb, samples)
grad = gradlogpdf(rb, samples)

# Save results using DrWatson
data_dict = @strdict samples lp grad params
safesave(datadir("sims", savename(params, "jld2")), data_dict)

# Extract x and y coordinates
x = samples[1, :]  # First row = x1 coordinates
y = samples[2, :]  # Second row = x2 coordinates

# Create figure
fig = figure(figsize=(11, 5))

# Scatter plot
subplot(1, 2, 1)
scatter(x, y, alpha=0.3, s=0.5, c="blue")
xlabel("x1")
ylabel("x2")
title("Rosenbrock Distribution Samples")
grid(true, alpha=0.3)

# 2D histogram
subplot(1, 2, 2)
plt.hist2d(x, y, bins=75, cmap="viridis", density=true)
colorbar(label="Density")
xlabel("x1")
ylabel("x2")
title("Sample Density")

# Save plot using DrWatson
wsave(plotsdir("rosenbrock_samples.png"), fig)
close()

println("Results saved to: ", datadir("sims"))
println("Plot saved to: ", plotsdir("rosenbrock_samples.png"))
println("Generated $(params[:n_samples]) samples")
println("Log-pdf range: ", extrema(lp))
println("Sample shape: ", size(samples))  # Will print (2, 50000)
println("Gradient shape: ", size(grad))    # Will print (2, 50000)