# -------------------------------
# Lévy Distribution Simulation
# Parameters: μ = 1, c = 4
# -------------------------------

# Set parameters
mu <- 1
c <- 4
n <- 10000  # Target sample size

# Set random seed for reproducibility
set.seed(123)

# Define Lévy distribution theoretical density function
f_levy <- function(x, mu, c) {
  ifelse(x > mu,
         sqrt(c / (2 * pi)) * exp(-c / (2 * (x - mu))) / ((x - mu)^(3/2)),
         0)
}

# Define Lévy distribution theoretical CDF
F_levy <- function(x, mu, c) {
  ifelse(x <= mu, 0, 
         2 * pnorm(sqrt(c / (x - mu)), lower.tail = FALSE))
}

cat("========================================\n")
cat("LÉVY DISTRIBUTION SIMULATION\n")
cat("Parameters: mu =", mu, ", c =", c, "\n")
cat("Sample size: n =", n, "\n")
cat("========================================\n\n")

# ========================================
# Method 1: Direct Sampling via Normal Distribution
# ========================================
cat("METHOD 1: Direct Sampling via Normal Distribution\n")
cat("--------------------------------------------------\n")
cat("Transformation: X = mu + c/Z^2, where Z ~ N(0,1)\n")
cat("This is an exact method with no rejection.\n\n")

Z <- rnorm(n, mean = 0, sd = 1)
X_direct <- mu + c / (Z^2)

cat("Successfully generated", n, "samples\n")
cat("Sample mean:", round(mean(X_direct), 4), "\n")
cat("Sample median:", round(median(X_direct), 4), "\n")
cat("Sample variance:", round(var(X_direct), 4), "\n\n")

# ========================================
# Method 2: Rejection Sampling with Pareto Proposal
# ========================================
cat("METHOD 2: Rejection Sampling with Pareto Proposal\n")
cat("--------------------------------------------------\n")

# Pareto distribution: g(x) = a*x_m^a / x^(a+1), x >= x_m
# Choose parameters: x_m = mu = 1, a = 0.45 (to match Lévy tail behavior)
x_m <- mu
a <- 0.45

cat("Proposal: Pareto distribution\n")
cat("Parameters: x_m =", x_m, ", a =", a, "\n")
cat("Density: g(x) = a*x_m^a / x^(a+1), x >= x_m\n\n")

# Pareto density function
g_pareto <- function(x, x_m, a) {
  ifelse(x >= x_m, a * x_m^a / x^(a+1), 0)
}

# Pareto sampling function (inverse transform)
rpareto <- function(n, x_m, a) {
  u <- runif(n)
  x_m / (u^(1/a))
}

# Find optimal constant M such that f(x) <= M*g(x) for all x
x_test <- seq(mu + 0.01, 100, length.out = 10000)
ratio_pareto <- f_levy(x_test, mu, c) / g_pareto(x_test, x_m, a)
M_pareto <- max(ratio_pareto, na.rm = TRUE)

cat("Constant M =", round(M_pareto, 4), "\n")
cat("(M ensures f(x) <= M*g(x) for all x)\n\n")

# Rejection sampling
X_pareto <- numeric(n)
n_accepted_pareto <- 0
n_total_pareto <- 0

while(n_accepted_pareto < n) {
  # Sample from proposal distribution
  y <- rpareto(1, x_m, a)
  n_total_pareto <- n_total_pareto + 1
  
  # Calculate acceptance probability
  u <- runif(1)
  acceptance_prob <- f_levy(y, mu, c) / (M_pareto * g_pareto(y, x_m, a))
  
  # Accept or reject
  if(u <= acceptance_prob) {
    n_accepted_pareto <- n_accepted_pareto + 1
    X_pareto[n_accepted_pareto] <- y
  }
}

acceptance_rate_pareto <- n / n_total_pareto
cat("Acceptance rate:", round(acceptance_rate_pareto * 100, 2), "%\n")
cat("Total proposals:", n_total_pareto, "\n")
cat("Sample mean:", round(mean(X_pareto), 4), "\n")
cat("Sample median:", round(median(X_pareto), 4), "\n")
cat("Sample variance:", round(var(X_pareto), 4), "\n\n")

# ========================================
# Method 3: Rejection Sampling with Uniform Proposal
# ========================================
cat("METHOD 3: Rejection Sampling with Uniform Proposal\n")
cat("--------------------------------------------------\n")

# Uniform distribution: g(x) = 1/(b-a), a <= x <= b
# Choose parameters: a = mu = 1, b = 10 (cover main region of Lévy)
a_unif <- mu
b_unif <- 10

cat("Proposal: Uniform distribution\n")
cat("Parameters: a =", a_unif, ", b =", b_unif, "\n")
cat("Density: g(x) = 1/(b-a) for x in [a,b]\n\n")

# Uniform density function
g_unif <- function(x, a, b) {
  ifelse(x >= a & x <= b, 1 / (b - a), 0)
}

# Find optimal constant M
x_test <- seq(a_unif + 0.01, b_unif, length.out = 10000)
ratio_unif <- f_levy(x_test, mu, c) / g_unif(x_test, a_unif, b_unif)
M_unif <- max(ratio_unif, na.rm = TRUE) * 1.1  # 1.1 as safety margin

cat("Constant M =", round(M_unif, 4), "\n")
cat("(M ensures f(x) <= M*g(x) for all x)\n\n")

# Rejection sampling
X_unif <- numeric(n)
n_accepted_unif <- 0
n_total_unif <- 0

while(n_accepted_unif < n) {
  # Sample from proposal distribution
  y <- runif(1, min = a_unif, max = b_unif)
  n_total_unif <- n_total_unif + 1
  
  # Calculate acceptance probability
  u <- runif(1)
  acceptance_prob <- f_levy(y, mu, c) / (M_unif * g_unif(y, a_unif, b_unif))
  
  # Accept or reject
  if(u <= acceptance_prob) {
    n_accepted_unif <- n_accepted_unif + 1
    X_unif[n_accepted_unif] <- y
  }
}

acceptance_rate_unif <- n / n_total_unif
cat("Acceptance rate:", round(acceptance_rate_unif * 100, 2), "%\n")
cat("Total proposals:", n_total_unif, "\n")
cat("Sample mean:", round(mean(X_unif), 4), "\n")
cat("Sample median:", round(median(X_unif), 4), "\n")
cat("Sample variance:", round(var(X_unif), 4), "\n\n")

# ========================================
# Method 4: Rejection Sampling with Inverse Gamma Proposal
# ========================================
cat("METHOD 4: Rejection Sampling with Inverse Gamma Proposal\n")
cat("----------------------------------------------------------------------\n")
cat("Goal: Achieve high acceptance rate (target >80%)\n\n")

# Strategy: Use Inverse Gamma distribution which has similar shape to Lévy
# Inverse Gamma: g(y) = (beta^alpha / Gamma(alpha)) * y^(-alpha-1) * exp(-beta/y)
# After shift: x = y + mu, we match the Lévy distribution shape

# Parameters chosen to match Lévy distribution closely
alpha_ig <- 0.47  # Shape parameter
beta_ig <- 1.55   # Scale parameter

cat("Proposal: Shifted Inverse Gamma Distribution\n")
cat("If Y ~ InverseGamma(alpha, beta), then X = Y + mu\n")
cat("Parameters:\n")
cat("  - Shape alpha =", alpha_ig, "\n")
cat("  - Scale beta =", beta_ig, "\n\n")

# Define shifted Inverse Gamma density
g_invgamma <- function(x, mu, alpha, beta) {
  y <- x - mu
  ifelse(y > 0, 
         (beta^alpha / gamma(alpha)) * y^(-alpha-1) * exp(-beta/y),
         0)
}

# Shifted Inverse Gamma sampling function
rinvgamma_shifted <- function(n, mu, alpha, beta) {
  # Sample from Inverse Gamma and shift
  y <- 1 / rgamma(n, shape = alpha, rate = beta)
  x <- y + mu
  return(x)
}

# Find optimal constant M
x_test <- seq(mu + 0.001, 100, length.out = 10000)
ratio_invgamma <- f_levy(x_test, mu, c) / g_invgamma(x_test, mu, alpha_ig, beta_ig)
M_invgamma <- max(ratio_invgamma[is.finite(ratio_invgamma)], na.rm = TRUE) * 1.02

cat("Constant M =", round(M_invgamma, 4), "\n")
cat("(M ensures f(x) <= M*g(x) for all x)\n\n")

# Rejection sampling
X_mixture <- numeric(n)
n_accepted_mixture <- 0
n_total_mixture <- 0

while(n_accepted_mixture < n) {
  # Sample from proposal distribution
  y <- rinvgamma_shifted(1, mu, alpha_ig, beta_ig)
  n_total_mixture <- n_total_mixture + 1
  
  # Calculate acceptance probability
  u <- runif(1)
  g_y <- g_invgamma(y, mu, alpha_ig, beta_ig)
  if(g_y > 0 && is.finite(g_y)) {
    acceptance_prob <- f_levy(y, mu, c) / (M_invgamma * g_y)
    
    # Accept or reject
    if(is.finite(acceptance_prob) && u <= acceptance_prob) {
      n_accepted_mixture <- n_accepted_mixture + 1
      X_mixture[n_accepted_mixture] <- y
    }
  }
}

acceptance_rate_mixture <- n / n_total_mixture
cat("Acceptance rate:", round(acceptance_rate_mixture * 100, 2), "%\n")
cat("Total proposals:", n_total_mixture, "\n")
cat("Sample mean:", round(mean(X_mixture), 4), "\n")
cat("Sample median:", round(median(X_mixture), 4), "\n")
cat("Sample variance:", round(var(X_mixture), 4), "\n\n")


# ========================================
# Summary Comparison
# ========================================
cat("========================================\n")
cat("SUMMARY COMPARISON\n")
cat("========================================\n\n")

methods <- c("Direct Sampling", "Pareto Proposal", "Uniform Proposal", "Mixed Proposal")
means <- c(mean(X_direct), mean(X_pareto), mean(X_unif), mean(X_mixture))
medians <- c(median(X_direct), median(X_pareto), median(X_unif), median(X_mixture))
sds <- c(sd(X_direct), sd(X_pareto), sd(X_unif), sd(X_mixture))
accept_rates <- c(100, acceptance_rate_pareto * 100, acceptance_rate_unif * 100, acceptance_rate_mixture * 100)

comparison <- data.frame(
  Method = methods,
  Mean = round(means, 4),
  Median = round(medians, 4),
  SD = round(sds, 4),
  Accept_Rate = round(accept_rates, 2)
)
print(comparison)

# # ========================================
# # Visualization
# # ========================================
# par(mfrow = c(2, 3), mar = c(4, 4, 3, 2))

# x_vals <- seq(1, 50, length.out = 1000)

# # Plot 1: Direct sampling - Histogram
# hist(X_direct, breaks = 100, freq = FALSE, 
#      xlim = c(1, 50), ylim = c(0, 0.5),
#      main = "Method 1: Direct Sampling",
#      xlab = "x", ylab = "Density",
#      col = "lightyellow", border = "white")
# lines(x_vals, f_levy(x_vals, mu, c), col = "red", lwd = 2)
# legend("topright", 
#        legend = c("Sample", "Target PDF"),
#        col = c("lightyellow", "red"), 
#        lwd = c(10, 2), cex = 0.8)

# # Plot 2: Pareto proposal - Histogram
# hist(X_pareto, breaks = 100, freq = FALSE, 
#      xlim = c(1, 50), ylim = c(0, 0.5),
#      main = "Method 2: Pareto Proposal",
#      xlab = "x", ylab = "Density",
#      col = "lightblue", border = "white")
# lines(x_vals, f_levy(x_vals, mu, c), col = "red", lwd = 2)
# lines(x_vals, g_pareto(x_vals, x_m, a), col = "green", lwd = 2, lty = 2)
# legend("topright", 
#        legend = c("Sample", "Target PDF", "Proposal PDF"),
#        col = c("lightblue", "red", "green"), 
#        lwd = c(10, 2, 2), lty = c(1, 1, 2), cex = 0.7)

# # Plot 3: Uniform proposal - Histogram
# hist(X_unif, breaks = 100, freq = FALSE, 
#      xlim = c(1, 50), ylim = c(0, 0.5),
#      main = "Method 3: Uniform Proposal",
#      xlab = "x", ylab = "Density",
#      col = "lightgreen", border = "white")
# lines(x_vals, f_levy(x_vals, mu, c), col = "red", lwd = 2)
# x_unif_range <- seq(a_unif, b_unif, length.out = 1000)
# lines(x_unif_range, g_unif(x_unif_range, a_unif, b_unif), col = "purple", lwd = 2, lty = 2)
# legend("topright", 
#        legend = c("Sample", "Target PDF", "Proposal PDF"),
#        col = c("lightgreen", "red", "purple"), 
#        lwd = c(10, 2, 2), lty = c(1, 1, 2), cex = 0.7)

# # Plot 4-6: CDF Comparison
# x_cdf <- seq(1, 50, length.out = 500)
# theoretical_cdf <- F_levy(x_cdf, mu, c)

# for(i in 1:3) {
#   X_current <- if(i == 1) X_direct else if(i == 2) X_pareto else X_unif
#   empirical_cdf <- sapply(x_cdf, function(x) mean(X_current <= x))
  
#   plot(x_cdf, empirical_cdf, type = "l", col = "blue", lwd = 2,
#        main = paste("CDF:", methods[i]),
#        xlab = "x", ylab = "CDF", ylim = c(0, 1))
#   lines(x_cdf, theoretical_cdf, col = "red", lwd = 2, lty = 2)
#   legend("bottomright", 
#          legend = c("Empirical", "Theoretical"),
#          col = c("blue", "red"), lwd = 2, lty = c(1, 2), cex = 0.8)
# }

# par(mfrow = c(1, 1))

# cat("\n========================================\n")
# cat("SIMULATION COMPLETED\n")
# cat("========================================\n")