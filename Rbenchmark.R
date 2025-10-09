# R script to benchmark scalar vs. vectorized random number generation.
# To run this script from your terminal: Rscript benchmark.R

# Install the necessary packages if you don't have them
# You only need to do this once.
if (!require("microbenchmark", quietly = TRUE)) {
  install.packages("microbenchmark")
}
if (!require("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2")
}

# Load the libraries
library(microbenchmark)
library(ggplot2)

# --- Benchmark Parameters ---
n <- 1000000   # The number of samples to generate
min_val <- 0.0   # The minimum value of the uniform distribution
max_val <- 1.0   # The maximum value of the uniform distribution
num_trials <- 100 # How many times to repeat each test

cat(paste("Benchmarking with n =", n, "samples over", num_trials, "trials.\n\n"))

# --- Run the Benchmark ---
# microbenchmark will run each expression 'num_trials' times and collect stats.
benchmark_results <- microbenchmark(
  "Scalar (for loop)" = {
    # This is the slow, un-R-like way, similar to your scalar Zig code.
    # Pre-allocate memory for the results (slightly better than growing it)
    results <- numeric(n)
    for (i in 1:n) {
      results[i] <- runif(1, min_val, max_val)
    }
    # We return 'results' to make sure the work isn't optimized away
    results
  },
  "Vectorized (R way)" = {
    # This is the fast, idiomatic R way.
    # The runif function is vectorized; you just tell it how many numbers you want.
    runif(n, min_val, max_val)
  },
  times = num_trials
)

# --- Print the Results ---
# This prints a nicely formatted summary table with stats in nanoseconds,
# microseconds, or milliseconds depending on the speed.
print(benchmark_results)

# --- Create a Plot of the Results ---
# A boxplot is a great way to visualize the distribution of timings.
plot <- autoplot(benchmark_results) +
  ggtitle(paste("Benchmark Results for n =", n)) +
  theme_minimal()

# Save the plot to a file
ggsave("benchmark_plot.png", plot, width = 8, height = 6)

cat("\nBenchmark complete. Plot saved to benchmark_plot.png\n")

