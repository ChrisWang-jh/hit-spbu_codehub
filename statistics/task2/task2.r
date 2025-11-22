# ============================================================
# 整合版：混合分布采样（二项/均匀） + 拒绝采样（任务2&3）
# 依赖：ggplot2, gridExtra
# 输出：mixture_plots.pdf, rejection_plots.pdf
# ============================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(gridExtra)
})

set.seed(123)
N <- 10000

cat("============================================================\n")
cat("整合脚本启动：混合采样 + 拒绝采样\n")
cat("============================================================\n\n")

# ----------------------------
# 通用：Laplace 辅助函数
# ----------------------------
qlaplace <- function(u, alpha, beta) ifelse(u < 0.5, beta + log(2*u)/alpha, beta - log(2*(1-u))/alpha)
dlaplace <- function(x, alpha, beta) (alpha/2) * exp(-alpha * abs(x - beta))
plaplace <- function(x, alpha, beta) ifelse(x < beta, 0.5 * exp(alpha * (x - beta)), 1 - 0.5 * exp(-alpha * (x - beta)))

# =====================================================================
# 问题1：正态 + Laplace 的混合（二项法 vs. 均匀法）
# =====================================================================
cat("------------------------------------------------------------\n")
cat("问题1：正态 + Laplace 混合\n")
cat("------------------------------------------------------------\n")

mean_norm <- 1
var_norm  <- 9
sd_norm   <- sqrt(var_norm)
alpha     <- 3
beta      <- 2
p         <- 0.4

# 理论密度/CDF
dmixture1 <- function(x) p * dnorm(x, mean_norm, sd_norm) + (1 - p) * dlaplace(x, alpha, beta)
pmixture1 <- function(x) p * pnorm(x, mean_norm, sd_norm) + (1 - p) * plaplace(x, alpha, beta)

# 方法1：二项法（基础）
simulate_binomial_method_basic <- function(n, p) {
  samples <- numeric(n)
  n1 <- rbinom(1, n, p)
  n2 <- n - n1
  for (i in 1:n1) samples[i] <- rnorm(1, mean_norm, sd_norm)
  for (i in 1:n2) {
    u <- runif(1)
    samples[n1 + i] <- if (u < 0.5) beta + log(2*u)/alpha else beta - log(2*(1-u))/alpha
  }
  # 打乱
  for (i in n:2) {
    j <- sample.int(i, 1)
    tmp <- samples[i]; samples[i] <- samples[j]; samples[j] <- tmp
  }
  samples
}

# 方法2：均匀法（基础）
simulate_uniform_method_basic <- function(n, p) {
  samples <- numeric(n)
  for (i in 1:n) {
    u <- runif(1)
    if (u < p) {
      samples[i] <- rnorm(1, mean_norm, sd_norm)
    } else {
      u2 <- runif(1)
      samples[i] <- if (u2 < 0.5) beta + log(2*u2)/alpha else beta - log(2*(1-u2))/alpha
    }
  }
  samples
}

cat("开始采样 (问题1)...\n")
time1_binom <- system.time(samples1_binom <- simulate_binomial_method_basic(N, p))
time1_unif  <- system.time(samples1_unif  <- simulate_uniform_method_basic(N, p))
cat("完成。\n\n")

# 理论矩
theoretical_mean1 <- p * mean_norm + (1 - p) * beta
var_laplace <- 2 / (alpha^2)
theoretical_var1  <- p * var_norm + (1 - p) * var_laplace + p*(1-p)*(mean_norm - beta)^2

# 经验矩
empirical_mean1_binom <- mean(samples1_binom); empirical_var1_binom <- var(samples1_binom)
empirical_mean1_unif  <- mean(samples1_unif);  empirical_var1_unif  <- var(samples1_unif)

cat("理论期望:", theoretical_mean1, "\n")
cat("理论方差:", theoretical_var1,  "\n\n")
cat("方法1-二项法：E=", empirical_mean1_binom, " D=", empirical_var1_binom, " | 时间=", time1_binom[3], "s\n")
cat("方法2-均匀法：E=", empirical_mean1_unif,  " D=", empirical_var1_unif,  " | 时间=", time1_unif[3],  "s\n\n")

# 绘图（问题1）
x_seq1 <- seq(-8, 10, length.out = 1000)
pdf1_data <- data.frame(x = x_seq1, Theoretical = dmixture1(x_seq1))

p1_pdf_binom <- ggplot() +
  geom_histogram(data = data.frame(x = samples1_binom), aes(x, after_stat(density)), bins = 60, fill = "lightblue", alpha = 0.6) +
  geom_line(data = pdf1_data, aes(x, Theoretical), color = "red", linewidth = 0.2) +
  labs(title = "1 - PDF (Ber)", x = "x", y = "pdf") + theme_minimal()

p1_pdf_unif <- ggplot() +
  geom_histogram(data = data.frame(x = samples1_unif), aes(x, after_stat(density)), bins = 60, fill = "lightgreen", alpha = 0.6) +
  geom_line(data = pdf1_data, aes(x, Theoretical), color = "red", linewidth = 0.2) +
  labs(title = "1 - PDF (Unif)", x = "x", y = "pdf") + theme_minimal()

cdf1_binom <- ecdf(samples1_binom)
cdf1_unif  <- ecdf(samples1_unif)

p1_cdf <- ggplot() +
  geom_line(data = data.frame(x = x_seq1, y = pmixture1(x_seq1)), aes(x, y, color = "Theoretical"), linewidth = 1.2) +
  geom_line(data = data.frame(x = x_seq1, y = cdf1_binom(x_seq1)), aes(x, y, color = "Ber"), linewidth = 0.8) +
  geom_line(data = data.frame(x = x_seq1, y = cdf1_unif(x_seq1)),  aes(x, y, color = "Unif"), linewidth = 0.8, linetype = "dashed") +
  labs(title = "1 - CDF", x = "x", y = "cdf", color = "") + theme_minimal()

# =====================================================================
# 问题2：分段混合密度（[-1,1] 与 [2,∞)），二项法/均匀法 + 拒绝采样(改进版)
# =====================================================================
cat("------------------------------------------------------------\n")
cat("问题2：分段混合密度（混合采样 + 拒绝采样改进）\n")
cat("------------------------------------------------------------\n")

a <- 3/8
dfun2 <- function(x) ifelse(x >= -1 & x <= 1, a*(1 - x^2), ifelse(x >= 2, 0.5*exp(-(x-2)), 0))

# 权重
p1_2 <- integrate(function(x) a * (1 - x^2), -1, 1)$value
p2_2 <- integrate(function(x) 0.5 * exp(-(x-2)), 2, Inf)$value

# 逆CDF（第一段）
inv_cdf2_part1 <- function(u) {
  f <- function(x) a * ((x - x^3/3) - (-1 + 1/3)) - u * p1_2
  uniroot(f, c(-1, 1))$root
}

# 二项法
simulate_prob2_binomial_basic <- function(n) {
  samples <- numeric(n)
  n1 <- rbinom(1, n, p1_2); n2 <- n - n1
  for (i in 1:n1) samples[i] <- inv_cdf2_part1(runif(1))
  for (i in 1:n2) samples[n1 + i] <- 2 + rexp(1, rate = 1)
  for (i in n:2) { j <- sample.int(i,1); tmp <- samples[i]; samples[i] <- samples[j]; samples[j] <- tmp }
  samples
}

# 均匀法
simulate_prob2_uniform_basic <- function(n) {
  samples <- numeric(n)
  for (i in 1:n) {
    if (runif(1) < p1_2) samples[i] <- inv_cdf2_part1(runif(1)) else samples[i] <- 2 + rexp(1, 1)
  }
  samples
}

# 拒绝采样（改进版，使用 U(-1,10) 作提议；M=1）
f2 <- dfun2
rejection_sampling_task2_naive_v2 <- function(n = 10000) {
  samples <- numeric(n); rejections <- 0; accepted <- 0
  M <- 1.25
  a_range <- -1; b_range <- 5; g_density <- 1/(b_range - a_range)
  while (accepted < n) {
    x <- runif(1, a_range, b_range); u <- runif(1)
    if (u <= f2(x)/(M*g_density)) { accepted <- accepted + 1; samples[accepted] <- x } else { rejections <- rejections + 1 }
  }
  acceptance_rate <- n/(n + rejections)
  cat("task2 - acceptance rate:", round(acceptance_rate, 4), "\n")
  cat("task2 - total proposals:", n + rejections, "\n")
  samples
}

# 采样
cat("开始采样 (问题2)...\n")
time2_binom <- system.time(samples2_binom <- simulate_prob2_binomial_basic(N))
time2_unif  <- system.time(samples2_unif  <- simulate_prob2_uniform_basic(N))
time2_rej   <- system.time(samples2_rej   <- rejection_sampling_task2_naive_v2(N))
cat("完成。\n\n")

# 理论矩
e1_2    <- integrate(function(x) x   * a*(1 - x^2), -1, 1)$value + integrate(function(x) x   * 0.5*exp(-(x-2)), 2, Inf)$value
e1sq_2  <- integrate(function(x) x^2 * a*(1 - x^2), -1, 1)$value + integrate(function(x) x^2 * 0.5*exp(-(x-2)), 2, Inf)$value
theoretical_mean2 <- e1_2
theoretical_var2  <- e1sq_2 - theoretical_mean2^2

# 经验矩
empirical_mean2_binom <- mean(samples2_binom); empirical_var2_binom <- var(samples2_binom)
empirical_mean2_unif  <- mean(samples2_unif);  empirical_var2_unif  <- var(samples2_unif)
empirical_mean2_rej   <- mean(samples2_rej);   empirical_var2_rej   <- var(samples2_rej)

cat("理论期望:", theoretical_mean2, "\n")
cat("理论方差:", theoretical_var2,  "\n\n")
cat("二项法：E=", empirical_mean2_binom, " D=", empirical_var2_binom, " | 时间=", time2_binom[3], "s\n")
cat("均匀法：E=", empirical_mean2_unif,  " D=", empirical_var2_unif,  " | 时间=", time2_unif[3],  "s\n")
cat("拒绝采样(改)：E=", empirical_mean2_rej,   " D=", empirical_var2_rej,   " | 时间=", time2_rej[3],   "s\n\n")

# 理论 CDF
pfun2 <- function(x) {
  ifelse(x < -1, 0,
         ifelse(x <= 1, a * ((x - x^3/3) - (-1 + 1/3)),
                ifelse(x < 2, p1_2,
                       p1_2 + 0.5 * (1 - exp(-(x-2))))))
}

# 绘图（问题2）
x_seq2 <- seq(-1.5, 8, length.out = 1000)
p2_pdf_binom <- ggplot() +
  geom_histogram(data = data.frame(x = samples2_binom), aes(x, after_stat(density)), bins = 60, fill = "lightblue", alpha = 0.6) +
  geom_line(data = data.frame(x = x_seq2, y = sapply(x_seq2, dfun2)), aes(x, y), color = "red", linewidth = 0.2) +
  labs(title = "2 - PDF (Ber)", x = "x", y = "pdf") + theme_minimal()

p2_pdf_unif <- ggplot() +
  geom_histogram(data = data.frame(x = samples2_unif), aes(x, after_stat(density)), bins = 60, fill = "lightgreen", alpha = 0.6) +
  geom_line(data = data.frame(x = x_seq2, y = sapply(x_seq2, dfun2)), aes(x, y), color = "red", linewidth = 0.2) +
  labs(title = "2 - PDF (Unif)", x = "x", y = "pdf") + theme_minimal()

p2_pdf_rej <- ggplot() +
  geom_histogram(data = data.frame(x = samples2_rej), aes(x, after_stat(density)), bins = 60, fill = "khaki", alpha = 0.6) +
  geom_line(data = data.frame(x = x_seq2, y = sapply(x_seq2, dfun2)), aes(x, y), color = "red", linewidth = 0.2) +
  labs(title = "2 - PDF (Reject) ", x = "x", y = "pdf") + theme_minimal()

cdf2_binom <- ecdf(samples2_binom)
cdf2_unif  <- ecdf(samples2_unif)
cdf2_rej   <- ecdf(samples2_rej)

p2_cdf <- ggplot() +
  geom_line(data = data.frame(x = x_seq2, y = sapply(x_seq2, pfun2)), aes(x, y, color = "Theoretical"), linewidth = 1.1) +
  geom_line(data = data.frame(x = x_seq2, y = cdf2_binom(x_seq2)), aes(x, y, color = "Ber"), linewidth = 0.7) +
  geom_line(data = data.frame(x = x_seq2, y = cdf2_unif(x_seq2)),  aes(x, y, color = "Unif"), linewidth = 0.7, linetype = "dashed") +
  geom_line(data = data.frame(x = x_seq2, y = cdf2_rej(x_seq2)),   aes(x, y, color = "Reject"), linewidth = 0.7, linetype = "dotdash") +
  labs(title = "2 - CDF (Theoretical vs Ber/Unif/Reject)", x = "x", y = "cdf", color = "") + theme_minimal()

# =====================================================================
# 问题3：三段混合密度（-1~1，1~2，2~∞），混合采样 + 拒绝采样（朴素）
# =====================================================================
cat("------------------------------------------------------------\n")
cat("问题3：三段混合密度（混合采样 + 拒绝采样朴素）\n")
cat("------------------------------------------------------------\n")

# 权重
p1_3 <- integrate(function(x) (1/8) * (x + 1), -1, 1)$value
p2_3 <- integrate(function(x) 1/4 + 0*x, 1, 2)$value
p3_3 <- integrate(function(x) 0.5 * exp(-(x-2)), 2, Inf)$value

dfun3 <- function(x) ifelse(x >= -1 & x < 1, (1/8)*(x + 1),
                            ifelse(x >= 1 & x < 2, 1/4,
                                   ifelse(x >= 2, 0.5*exp(-(x-2)), 0)))

inv_cdf3_part1 <- function(u) {
  u_scaled <- u * p1_3
  (-2 + sqrt(4 + 4*(16*u_scaled - 1))) / 2
}

simulate_prob3_binomial_basic <- function(n) {
  samples <- numeric(n)
  probs <- c(p1_3, p2_3, p3_3)
  u <- runif(n)
  n1 <- sum(u < probs[1])
  n2 <- sum(u >= probs[1] & u < (probs[1] + probs[2]))
  n3 <- n - n1 - n2
  idx <- 1
  for (i in 1:n1) { samples[idx] <- inv_cdf3_part1(runif(1)); idx <- idx + 1 }
  for (i in 1:n2) { samples[idx] <- runif(1, 1, 2);             idx <- idx + 1 }
  for (i in 1:n3) { samples[idx] <- 2 + rexp(1, 1);            idx <- idx + 1 }
  for (i in n:2) { j <- sample.int(i,1); tmp <- samples[i]; samples[i] <- samples[j]; samples[j] <- tmp }
  samples
}

simulate_prob3_uniform_basic <- function(n) {
  samples <- numeric(n)
  for (i in 1:n) {
    u <- runif(1)
    if (u < p1_3) {
      samples[i] <- inv_cdf3_part1(runif(1))
    } else if (u < p1_3 + p2_3) {
      samples[i] <- runif(1, 1, 2)
    } else {
      samples[i] <- 2 + rexp(1, 1)
    }
  }
  samples
}

# 拒绝采样（朴素，U(-1,10)，M=0.5）
f3 <- dfun3
rejection_sampling_task3_naive <- function(n = 10000) {
  samples <- numeric(n); rejections <- 0; accepted <- 0
  M <- 0.5
  a_range <- -1; b_range <- 10; g_density <- 1/(b_range - a_range)
  while (accepted < n) {
    x <- runif(1, a_range, b_range); u <- runif(1)
    if (u <= f3(x)/(M*g_density)) { accepted <- accepted + 1; samples[accepted] <- x } else { rejections <- rejections + 1 }
  }
  acceptance_rate <- n/(n + rejections)
  cat("task3 - acceptance rate:", round(acceptance_rate, 4), "\n")
  cat("task3 - total proposals:", n + rejections, "\n")
  samples
}

# 采样
cat("开始采样 (问题3)...\n")
time3_binom <- system.time(samples3_binom <- simulate_prob3_binomial_basic(N))
time3_unif  <- system.time(samples3_unif  <- simulate_prob3_uniform_basic(N))
time3_rej   <- system.time(samples3_rej   <- rejection_sampling_task3_naive(N))
cat("完成。\n\n")

# 理论矩
e1_3    <- integrate(function(x) x   * (1/8)*(x + 1), -1, 1)$value +
           integrate(function(x) x   * (1/4),          1, 2)$value +
           integrate(function(x) x   * 0.5*exp(-(x-2)), 2, Inf)$value
e1sq_3  <- integrate(function(x) x^2 * (1/8)*(x + 1), -1, 1)$value +
           integrate(function(x) x^2 * (1/4),          1, 2)$value +
           integrate(function(x) x^2 * 0.5*exp(-(x-2)), 2, Inf)$value
theoretical_mean3 <- e1_3
theoretical_var3  <- e1sq_3 - theoretical_mean3^2

# 经验矩
empirical_mean3_binom <- mean(samples3_binom); empirical_var3_binom <- var(samples3_binom)
empirical_mean3_unif  <- mean(samples3_unif);  empirical_var3_unif  <- var(samples3_unif)
empirical_mean3_rej   <- mean(samples3_rej);   empirical_var3_rej   <- var(samples3_rej)

cat("理论期望:", theoretical_mean3, "\n")
cat("理论方差:", theoretical_var3,  "\n\n")
cat("二项法：E=", empirical_mean3_binom, " D=", empirical_var3_binom, " | 时间=", time3_binom[3], "s\n")
cat("均匀法：E=", empirical_mean3_unif,  " D=", empirical_var3_unif,  " | 时间=", time3_unif[3],  "s\n")
cat("拒绝采样(朴)：E=", empirical_mean3_rej,   " D=", empirical_var3_rej,   " | 时间=", time3_rej[3],   "s\n\n")

# 理论 CDF
pfun3 <- function(x) {
  ifelse(x < -1, 0,
         ifelse(x < 1, (1/8) * ((x^2/2 + x) - (1/2 - 1)),
                ifelse(x < 2, p1_3 + (x - 1) * (1/4),
                       p1_3 + p2_3 + 0.5 * (1 - exp(-(x-2))))))
}

# 绘图（问题3）
x_seq3 <- seq(-1.5, 8, length.out = 1000)
p3_pdf_binom <- ggplot() +
  geom_histogram(data = data.frame(x = samples3_binom), aes(x, after_stat(density)), bins = 60, fill = "lightblue", alpha = 0.6) +
  geom_line(data = data.frame(x = x_seq3, y = sapply(x_seq3, dfun3)), aes(x, y), color = "red", linewidth = 0.2) +
  labs(title = "3 - PDF (Ber)", x = "x", y = "pdf") + theme_minimal()

p3_pdf_unif <- ggplot() +
  geom_histogram(data = data.frame(x = samples3_unif), aes(x, after_stat(density)), bins = 60, fill = "lightgreen", alpha = 0.6) +
  geom_line(data = data.frame(x = x_seq3, y = sapply(x_seq3, dfun3)), aes(x, y), color = "red", linewidth = 0.2) +
  labs(title = "3 - PDF (Unif)", x = "x", y = "pdf") + theme_minimal()

p3_pdf_rej <- ggplot() +
  geom_histogram(data = data.frame(x = samples3_rej), aes(x, after_stat(density)), bins = 60, fill = "plum", alpha = 0.6) +
  geom_line(data = data.frame(x = x_seq3, y = sapply(x_seq3, dfun3)), aes(x, y), color = "red", linewidth = 0.2) +
  labs(title = "3 - PDF (Reject)", x = "x", y = "pdf") + theme_minimal()

cdf3_binom <- ecdf(samples3_binom)
cdf3_unif  <- ecdf(samples3_unif)
cdf3_rej   <- ecdf(samples3_rej)

p3_cdf <- ggplot() +
  geom_line(data = data.frame(x = x_seq3, y = sapply(x_seq3, pfun3)), aes(x, y, color = "Theoretical"), linewidth = 1.1) +
  geom_line(data = data.frame(x = x_seq3, y = cdf3_binom(x_seq3)), aes(x, y, color = "Ber"), linewidth = 0.7) +
  geom_line(data = data.frame(x = x_seq3, y = cdf3_unif(x_seq3)),  aes(x, y, color = "Unif"), linewidth = 0.7, linetype = "dashed") +
  geom_line(data = data.frame(x = x_seq3, y = cdf3_rej(x_seq3)),   aes(x, y, color = "Reject"), linewidth = 0.7, linetype = "dotdash") +
  labs(title = "3 - CDF (Theoretical vs Ber/Unif/Reject)", x = "x", y = "cdf", color = "") + theme_minimal()

# =====================================================================
# 图像输出
# =====================================================================
cat("输出图像文件...\n")

# Mixture（问题1-3，二项/均匀）
pdf("mixture_plots.pdf", width = 10, height = 12)
grid.arrange(p1_pdf_binom, p1_pdf_unif, p1_cdf, ncol = 2, nrow = 2)
grid.arrange(p2_pdf_binom, p2_pdf_unif, p2_cdf, ncol = 2, nrow = 2)
grid.arrange(p3_pdf_binom, p3_pdf_unif, p3_cdf, ncol = 2, nrow = 2)
dev.off()

# Rejection（问题2-3，拒绝采样对比）
pdf("rejection_plots.pdf", width = 10, height = 8)
grid.arrange(p2_pdf_rej, p2_cdf, p3_pdf_rej, p3_cdf, ncol = 2, nrow = 2)
dev.off()

cat("✅ 已保存：mixture_plots.pdf, rejection_plots.pdf\n\n")

# =====================================================================
# 详细结果对比表（含拒绝采样）
# =====================================================================
cat("============================================================\n")
cat("理论矩 vs 经验矩 - 详细比较\n")
cat("============================================================\n\n")

comparison_table <- data.frame(
  Problem = rep(c("Task1", "Task2", "Task3"), each = 4),
  Method  = rep(c("Theoretical", "Binomial", "Uniform", "Reject"), 3),
  Expectation = c(
    theoretical_mean1, empirical_mean1_binom, empirical_mean1_unif, NA,
    theoretical_mean2, empirical_mean2_binom, empirical_mean2_unif, empirical_mean2_rej,
    theoretical_mean3, empirical_mean3_binom, empirical_mean3_unif, empirical_mean3_rej
  ),
  Variance = c(
    theoretical_var1, empirical_var1_binom, empirical_var1_unif, NA,
    theoretical_var2, empirical_var2_binom, empirical_var2_unif, empirical_var2_rej,
    theoretical_var3, empirical_var3_binom, empirical_var3_unif, empirical_var3_rej
  ),
  Time_sec = c(
    NA, time1_binom[3], time1_unif[3], NA,
    NA, time2_binom[3], time2_unif[3], time2_rej[3],
    NA, time3_binom[3], time3_unif[3], time3_rej[3]
  )
)

print(comparison_table, row.names = FALSE)

cat("\n时间对比（采样部分）：\n")
cat(sprintf("  task1: bin = %.4fs, unif = %.4fs\n",
            time1_binom[3], time1_unif[3]))
cat(sprintf("  task2: bin = %.4fs, unif = %.4fs, rej = %.4fs\n",
            time2_binom[3], time2_unif[3], time2_rej[3]))
cat(sprintf("  task3: bin = %.4fs, unif = %.4fs, rej = %.4fs\n",
            time3_binom[3], time3_unif[3], time3_rej[3]))
