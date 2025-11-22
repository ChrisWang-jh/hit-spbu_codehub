library(ggplot2)
library(gridExtra)
set.seed(123)

N <- 10000

# ============================================================================
# 问题 1: 正态分布与Laplace分布的混合
# ============================================================================

cat("========================================================================\n")
cat("问题 1: 正态分布 + Laplace分布混合\n")
cat("======================================================================\n\n")

# Constants
mean_norm <- 1
var_norm <- 9
sd_norm <- sqrt(var_norm)
alpha <- 3
beta <- 2
p <- 0.4

# Laplace分布的分位数函数
qlaplace <- function(u, alpha, beta) {
  ifelse(u < 0.5,
         beta + log(2*u) / alpha,
         beta - log(2*(1-u)) / alpha)
}

# Laplace分布的密度函数
dlaplace <- function(x, alpha, beta) {
  (alpha/2) * exp(-alpha * abs(x - beta))
}

# Laplace分布的CDF
plaplace <- function(x, alpha, beta) {
  ifelse(x < beta,
         0.5 * exp(alpha * (x - beta)),
         1 - 0.5 * exp(-alpha * (x - beta)))
}

# ========== 方法1: 二项分布法 (Binomial Method) - 基础版本 ==========
simulate_binomial_method_basic <- function(n, p) {
  samples <- numeric(n)
  
  # 用二项分布决定有多少个样本来自第一个分布
  n1 <- rbinom(1, n, p)
  n2 <- n - n1
  
  # 逐个生成第一个分布的样本
  for(i in 1:n1) {
    samples[i] <- rnorm(1, mean = mean_norm, sd = sd_norm)
  }
  
  # 逐个生成第二个分布的样本
  for(i in 1:n2) {
    u <- runif(1)
    if(u < 0.5) {
      samples[n1 + i] <- beta + log(2*u) / alpha
    } else {
      samples[n1 + i] <- beta - log(2*(1-u)) / alpha
    }
  }
  
  # 逐个打乱
  for(i in n:2) {
    j <- sample.int(i, 1)
    temp <- samples[i]
    samples[i] <- samples[j]
    samples[j] <- temp
  }
  
  return(samples)
}

# ========== 方法2: 均匀分布法 (Uniform Method) - 基础版本 ==========
simulate_uniform_method_basic <- function(n, p) {
  samples <- numeric(n)
  
  # 对每个样本用均匀分布决定来自哪个分布
  for(i in 1:n) {
    u <- runif(1)
    if(u < p) {
      # 来自正态分布
      samples[i] <- rnorm(1, mean = mean_norm, sd = sd_norm)
    } else {
      # 来自Laplace分布
      u2 <- runif(1)
      if(u2 < 0.5) {
        samples[i] <- beta + log(2*u2) / alpha
      } else {
        samples[i] <- beta - log(2*(1-u2)) / alpha
      }
    }
  }
  
  return(samples)
}

# 生成样本并计时 - 只计时采样过程
cat("开始采样 (问题1)...\n")
time1_binom <- system.time({
  samples1_binom <- simulate_binomial_method_basic(N, p)
})
cat("二项分布法完成\n")

time1_unif <- system.time({
  samples1_unif <- simulate_uniform_method_basic(N, p)
})
cat("均匀分布法完成\n\n")

# 理论矩计算
theoretical_mean1 <- p * mean_norm + (1-p) * beta
var_laplace <- 2 / (alpha^2)
theoretical_var1 <- p * var_norm + (1-p) * var_laplace + 
                    p * (1-p) * (mean_norm - beta)^2

# 经验矩
empirical_mean1_binom <- mean(samples1_binom)
empirical_var1_binom <- var(samples1_binom)
empirical_mean1_unif <- mean(samples1_unif)
empirical_var1_unif <- var(samples1_unif)

cat("理论期望:", theoretical_mean1, "\n")
cat("理论方差:", theoretical_var1, "\n\n")

cat("方法1 - 二项分布法 (Binomial Method):\n")
cat("  经验期望:", empirical_mean1_binom, "  误差:", abs(empirical_mean1_binom - theoretical_mean1), "\n")
cat("  经验方差:", empirical_var1_binom, "  误差:", abs(empirical_var1_binom - theoretical_var1), "\n")
cat("  计算时间:", time1_binom[3], "秒\n")
cat("  算法复杂度: O(n)\n\n")

cat("方法2 - 均匀分布法 (Uniform Method):\n")
cat("  经验期望:", empirical_mean1_unif, "  误差:", abs(empirical_mean1_unif - theoretical_mean1), "\n")
cat("  经验方差:", empirical_var1_unif, "  误差:", abs(empirical_var1_unif - theoretical_var1), "\n")
cat("  计算时间:", time1_unif[3], "秒\n")
cat("  算法复杂度: O(n)\n\n")

# 混合分布的理论密度和CDF
dmixture1 <- function(x) {
  p * dnorm(x, mean = mean_norm, sd = sd_norm) + 
    (1-p) * dlaplace(x, alpha, beta)
}

pmixture1 <- function(x) {
  p * pnorm(x, mean = mean_norm, sd = sd_norm) + 
    (1-p) * plaplace(x, alpha, beta)
}

# 绘图
x_seq1 <- seq(-8, 10, length.out = 1000)

pdf1_data <- data.frame(x = x_seq1, Theoretical = dmixture1(x_seq1))

p1_pdf_binom <- ggplot() +
  geom_histogram(data = data.frame(x = samples1_binom), 
                 aes(x = x, y = after_stat(density)), 
                 bins = 60, fill = "lightblue", alpha = 0.6) +
  geom_line(data = pdf1_data, aes(x = x, y = Theoretical), 
            color = "red", linewidth = 0.2) +
  labs(title = "1 - PDF (Ber)", x = "x", y = "pdf") +
  theme_minimal()

p1_pdf_unif <- ggplot() +
  geom_histogram(data = data.frame(x = samples1_unif), 
                 aes(x = x, y = after_stat(density)), 
                 bins = 60, fill = "lightgreen", alpha = 0.6) +
  geom_line(data = pdf1_data, aes(x = x, y = Theoretical), 
            color = "red", linewidth = 0.2) +
  labs(title = "1 - PDF (Unif)", x = "x", y = "pdf") +
  theme_minimal()

# CDF
cdf1_binom <- ecdf(samples1_binom)
cdf1_unif <- ecdf(samples1_unif)

p1_cdf <- ggplot() +
  geom_line(data = data.frame(x = x_seq1, y = pmixture1(x_seq1)),
            aes(x = x, y = y, color = "Theoretical"), linewidth = 1.2) +
  geom_line(data = data.frame(x = x_seq1, y = cdf1_binom(x_seq1)),
            aes(x = x, y = y, color = "Ber"), linewidth = 0.8) +
  geom_line(data = data.frame(x = x_seq1, y = cdf1_unif(x_seq1)),
            aes(x = x, y = y, color = "Unif"), linewidth = 0.8, linetype = "dashed") +
  labs(title = "1 - CDF", x = "x", y = "cdf", color = "") +
  theme_minimal()

# ============================================================================
# 问题 2: 分段混合密度
# ============================================================================

cat("\n======================================================================\n")
cat("问题 2: 分段混合密度函数\n")
cat("======================================================================\n\n")

# 归一化常数 a = 3/8
a <- 3/8

# weight
p1_2 <- integrate(function(x) a * (1 - x^2), -1, 1)$value
p2_2 <- integrate(function(x) 0.5 * exp(-(x-2)), 2, Inf)$value

cat("归一化常数 a =", a, "\n")
cat("第一部分权重 [-1, 1]:", p1_2, "\n")
cat("第二部分权重 [2, ∞):", p2_2, "\n\n")

# pdf
dfun2 <- function(x) {
  ifelse(x >= -1 & x <= 1, a * (1 - x^2),
         ifelse(x >= 2, 0.5 * exp(-(x-2)), 0))
}

# 第一部分的逆CDF（数值求解）
inv_cdf2_part1 <- function(u) {
  f <- function(x) {
    a * ((x - x^3/3) - (-1 + 1/3)) - u * p1_2
  }
  uniroot(f, c(-1, 1))$root
}

# ========== 方法1: 二项分布法 - 基础版本 ==========
simulate_prob2_binomial_basic <- function(n) {
  samples <- numeric(n)
  
  # 用二项分布决定样本数
  n1 <- rbinom(1, n, p1_2)
  n2 <- n - n1
  
  # 逐个生成第一部分样本
  for(i in 1:n1) {
    samples[i] <- inv_cdf2_part1(runif(1))
  }
  
  # 逐个生成第二部分样本
  for(i in 1:n2) {
    samples[n1 + i] <- 2 + rexp(1, rate = 1)
  }
  
  # 逐个打乱
  for(i in n:2) {
    j <- sample.int(i, 1)
    temp <- samples[i]
    samples[i] <- samples[j]
    samples[j] <- temp
  }
  
  return(samples)
}

# ========== 方法2: 均匀分布法 - 基础版本 ==========
simulate_prob2_uniform_basic <- function(n) {
  samples <- numeric(n)
  
  for(i in 1:n) {
    u <- runif(1)
    if(u < p1_2) {
      samples[i] <- inv_cdf2_part1(runif(1))
    } else {
      samples[i] <- 2 + rexp(1, rate = 1)
    }
  }
  
  return(samples)
}

# 生成样本
cat("开始采样 (问题2)...\n")
time2_binom <- system.time({
  samples2_binom <- simulate_prob2_binomial_basic(N)
})
cat("二项分布法完成\n")

time2_unif <- system.time({
  samples2_unif <- simulate_prob2_uniform_basic(N)
})
cat("均匀分布法完成\n\n")

# 理论矩
e1_2 <- integrate(function(x) x * a * (1 - x^2), -1, 1)$value
e2_2 <- integrate(function(x) x * 0.5 * exp(-(x-2)), 2, Inf)$value
theoretical_mean2 <- e1_2 + e2_2

e1_sq_2 <- integrate(function(x) x^2 * a * (1 - x^2), -1, 1)$value
e2_sq_2 <- integrate(function(x) x^2 * 0.5 * exp(-(x-2)), 2, Inf)$value
theoretical_var2 <- e1_sq_2 + e2_sq_2 - theoretical_mean2^2

# 经验矩
empirical_mean2_binom <- mean(samples2_binom)
empirical_var2_binom <- var(samples2_binom)
empirical_mean2_unif <- mean(samples2_unif)
empirical_var2_unif <- var(samples2_unif)

cat("理论期望:", theoretical_mean2, "\n")
cat("理论方差:", theoretical_var2, "\n\n")

cat("方法1 - 二项分布法:\n")
cat("  经验期望:", empirical_mean2_binom, "  误差:", abs(empirical_mean2_binom - theoretical_mean2), "\n")
cat("  经验方差:", empirical_var2_binom, "  误差:", abs(empirical_var2_binom - theoretical_var2), "\n")
cat("  计算时间:", time2_binom[3], "秒\n\n")

cat("方法2 - 均匀分布法:\n")
cat("  经验期望:", empirical_mean2_unif, "  误差:", abs(empirical_mean2_unif - theoretical_mean2), "\n")
cat("  经验方差:", empirical_var2_unif, "  误差:", abs(empirical_var2_unif - theoretical_mean2), "\n")
cat("  计算时间:", time2_unif[3], "秒\n\n")

# 绘图
x_seq2 <- seq(-1.5, 8, length.out = 1000)

p2_pdf_binom <- ggplot() +
  geom_histogram(data = data.frame(x = samples2_binom), 
                 aes(x = x, y = after_stat(density)), 
                 bins = 60, fill = "lightblue", alpha = 0.6) +
  geom_line(data = data.frame(x = x_seq2, y = sapply(x_seq2, dfun2)), 
            aes(x = x, y = y), color = "red", linewidth = 0.2) +
  labs(title = "2 - PDF (Ber)", x = "x", y = "pdf") +
  theme_minimal()

p2_pdf_unif <- ggplot() +
  geom_histogram(data = data.frame(x = samples2_unif), 
                 aes(x = x, y = after_stat(density)), 
                 bins = 60, fill = "lightgreen", alpha = 0.6) +
  geom_line(data = data.frame(x = x_seq2, y = sapply(x_seq2, dfun2)), 
            aes(x = x, y = y), color = "red", linewidth = 0.2) +
  labs(title = "2 - PDF (Unif)", x = "x", y = "pdf") +
  theme_minimal()

# CDF
pfun2 <- function(x) {
  ifelse(x < -1, 0,
         ifelse(x <= 1, a * ((x - x^3/3) - (-1 + 1/3)),
                ifelse(x < 2, p1_2,
                       p1_2 + 0.5 * (1 - exp(-(x-2))))))
}

cdf2_binom <- ecdf(samples2_binom)
cdf2_unif <- ecdf(samples2_unif)

p2_cdf <- ggplot() +
  geom_line(data = data.frame(x = x_seq2, y = sapply(x_seq2, pfun2)),
            aes(x = x, y = y, color = "Theoretical"), linewidth = 1.2) +
  geom_line(data = data.frame(x = x_seq2, y = cdf2_binom(x_seq2)),
            aes(x = x, y = y, color = "Ber"), linewidth = 0.8) +
  geom_line(data = data.frame(x = x_seq2, y = cdf2_unif(x_seq2)),
            aes(x = x, y = y, color = "Unif"), linewidth = 0.8, linetype = "dashed") +
  labs(title = "2 - CDF", x = "x", y = "cdf", color = "") +
  theme_minimal()

# ============================================================================
# 问题 3: 三段混合密度
# ============================================================================

cat("\n======================================================================\n")
cat("问题 3: 三段混合密度函数\n")
cat("======================================================================\n\n")

# 计算各部分的概率权重
p1_3 <- integrate(function(x) (1/8) * (x + 1), -1, 1)$value
p2_3 <- integrate(function(x) 1/4 + 0*x, 1, 2)$value
p3_3 <- integrate(function(x) 0.5 * exp(-(x-2)), 2, Inf)$value

cat("第一部分权重 [-1, 1):", p1_3, "\n")
cat("第二部分权重 [1, 2):", p2_3, "\n")
cat("第三部分权重 [2, ∞):", p3_3, "\n")
cat("总和:", p1_3 + p2_3 + p3_3, "\n\n")

# 密度函数
dfun3 <- function(x) {
  ifelse(x >= -1 & x < 1, (1/8) * (x + 1),
         ifelse(x >= 1 & x < 2, 1/4,
                ifelse(x >= 2, 0.5 * exp(-(x-2)), 0)))
}

# 第一部分的逆CDF
inv_cdf3_part1 <- function(u) {
  u_scaled <- u * p1_3
  (-2 + sqrt(4 + 4*(16*u_scaled - 1))) / 2
}

# ========== 方法1: 二项分布法 - 基础版本 ==========
simulate_prob3_binomial_basic <- function(n) {
  samples <- numeric(n)
  
  # 用多项分布决定各部分的样本数
  probs <- c(p1_3, p2_3, p3_3)
  u <- runif(n)
  n1 <- sum(u < probs[1])
  n2 <- sum(u >= probs[1] & u < (probs[1] + probs[2]))
  n3 <- n - n1 - n2
  
  # 逐个生成第一部分样本
  idx <- 1
  for(i in 1:n1) {
    samples[idx] <- inv_cdf3_part1(runif(1))
    idx <- idx + 1
  }
  
  # 逐个生成第二部分样本
  for(i in 1:n2) {
    samples[idx] <- runif(1, 1, 2)
    idx <- idx + 1
  }
  
  # 逐个生成第三部分样本
  for(i in 1:n3) {
    samples[idx] <- 2 + rexp(1, rate = 1)
    idx <- idx + 1
  }
  
  # 逐个打乱
  for(i in n:2) {
    j <- sample.int(i, 1)
    temp <- samples[i]
    samples[i] <- samples[j]
    samples[j] <- temp
  }
  
  return(samples)
}

# ========== 方法2: 均匀分布法 - 基础版本 ==========
simulate_prob3_uniform_basic <- function(n) {
  samples <- numeric(n)
  
  for(i in 1:n) {
    u <- runif(1)
    if(u < p1_3) {
      samples[i] <- inv_cdf3_part1(runif(1))
    } else if(u < p1_3 + p2_3) {
      samples[i] <- runif(1, 1, 2)
    } else {
      samples[i] <- 2 + rexp(1, rate = 1)
    }
  }
  
  return(samples)
}

# 生成样本
cat("开始采样 (问题3)...\n")
time3_binom <- system.time({
  samples3_binom <- simulate_prob3_binomial_basic(N)
})
cat("二项分布法完成\n")

time3_unif <- system.time({
  samples3_unif <- simulate_prob3_uniform_basic(N)
})
cat("均匀分布法完成\n\n")

# 理论矩
e1_3 <- integrate(function(x) x * (1/8) * (x + 1), -1, 1)$value
e2_3 <- integrate(function(x) x * (1/4), 1, 2)$value
e3_3 <- integrate(function(x) x * 0.5 * exp(-(x-2)), 2, Inf)$value
theoretical_mean3 <- e1_3 + e2_3 + e3_3

e1_sq_3 <- integrate(function(x) x^2 * (1/8) * (x + 1), -1, 1)$value
e2_sq_3 <- integrate(function(x) x^2 * (1/4), 1, 2)$value
e3_sq_3 <- integrate(function(x) x^2 * 0.5 * exp(-(x-2)), 2, Inf)$value
theoretical_var3 <- e1_sq_3 + e2_sq_3 + e3_sq_3 - theoretical_mean3^2

# 经验矩
empirical_mean3_binom <- mean(samples3_binom)
empirical_var3_binom <- var(samples3_binom)
empirical_mean3_unif <- mean(samples3_unif)
empirical_var3_unif <- var(samples3_unif)

cat("理论期望:", theoretical_mean3, "\n")
cat("理论方差:", theoretical_var3, "\n\n")

cat("方法1 - 二项分布法:\n")
cat("  经验期望:", empirical_mean3_binom, "  误差:", abs(empirical_mean3_binom - theoretical_mean3), "\n")
cat("  经验方差:", empirical_var3_binom, "  误差:", abs(empirical_var3_binom - theoretical_var3), "\n")
cat("  计算时间:", time3_binom[3], "秒\n\n")

cat("方法2 - 均匀分布法:\n")
cat("  经验期望:", empirical_mean3_unif, "  误差:", abs(empirical_mean3_unif - theoretical_mean3), "\n")
cat("  经验方差:", empirical_var3_unif, "  误差:", abs(empirical_var3_unif - theoretical_var3), "\n")
cat("  计算时间:", time3_unif[3], "秒\n\n")

# 绘图
x_seq3 <- seq(-1.5, 8, length.out = 1000)

p3_pdf_binom <- ggplot() +
  geom_histogram(data = data.frame(x = samples3_binom), 
                 aes(x = x, y = after_stat(density)), 
                 bins = 60, fill = "lightblue", alpha = 0.6) +
  geom_line(data = data.frame(x = x_seq3, y = sapply(x_seq3, dfun3)), 
            aes(x = x, y = y), color = "red", linewidth = 0.2) +
  labs(title = "3 - PDF (Ber)", x = "x", y = "pdf") +
  theme_minimal()

p3_pdf_unif <- ggplot() +
  geom_histogram(data = data.frame(x = samples3_unif), 
                 aes(x = x, y = after_stat(density)), 
                 bins = 60, fill = "lightgreen", alpha = 0.6) +
  geom_line(data = data.frame(x = x_seq3, y = sapply(x_seq3, dfun3)), 
            aes(x = x, y = y), color = "red", linewidth = 0.2) +
  labs(title = "3 - PDF (Unif)", x = "x", y = "pdf") +
  theme_minimal()

# CDF
pfun3 <- function(x) {
  ifelse(x < -1, 0,
         ifelse(x < 1, (1/8) * ((x^2/2 + x) - (1/2 - 1)),
                ifelse(x < 2, p1_3 + (x - 1) * (1/4),
                       p1_3 + p2_3 + 0.5 * (1 - exp(-(x-2))))))
}

cdf3_binom <- ecdf(samples3_binom)
cdf3_unif <- ecdf(samples3_unif)

p3_cdf <- ggplot() +
  geom_line(data = data.frame(x = x_seq3, y = sapply(x_seq3, pfun3)),
            aes(x = x, y = y, color = "Theoretical"), linewidth = 1.2) +
  geom_line(data = data.frame(x = x_seq3, y = cdf3_binom(x_seq3)),
            aes(x = x, y = y, color = "Ber"), linewidth = 0.8) +
  geom_line(data = data.frame(x = x_seq3, y = cdf3_unif(x_seq3)),
            aes(x = x, y = y, color = "Unif"), linewidth = 0.8, linetype = "dashed") +
  labs(title = "3 - CDF", x = "x", y = "cdf", color = "") +
  theme_minimal()

# ============================================================================
# 汇总展示
# ============================================================================

cat("\n======================================================================\n")
cat("生成图形...\n")
cat("======================================================================\n\n")

# 问题1的图形
grid.arrange(p1_pdf_binom, p1_pdf_unif, p1_cdf, ncol = 2, nrow = 2)

# 问题2的图形
grid.arrange(p2_pdf_binom, p2_pdf_unif, p2_cdf, ncol = 2, nrow = 2)

# 问题3的图形
grid.arrange(p3_pdf_binom, p3_pdf_unif, p3_cdf, ncol = 2, nrow = 2)

# ============================================================================
# 算法复杂度分析总结
# ============================================================================

cat("时间对比 (采样部分):\n")
cat(sprintf("  task1: bin = %.4f秒, unif = %.4f秒, 比值 = %.2f\n", 
            time1_binom[3], time1_unif[3], time1_unif[3]/time1_binom[3]))
cat(sprintf("  task2: bin = %.4f秒, unif = %.4f秒, 比值 = %.2f\n", 
            time2_binom[3], time2_unif[3], time2_unif[3]/time2_binom[3]))
cat(sprintf("  task3: bin = %.4f秒, unif = %.4f秒, 比值 = %.2f\n\n", 
            time3_binom[3], time3_unif[3], time3_unif[3]/time3_binom[3]))


# ============================================================================
# 详细结果对比表
# ============================================================================

cat("======================================================================\n")
cat("理论矩 vs 经验矩 - 详细比较\n")
cat("======================================================================\n\n")

comparison_table <- data.frame(
  Problem = rep(c("Task1", "Task2", "Task3"), each = 3),
  Method = rep(c("Theoretical", "Binomial", "Uniform"), 3),
  Expectation = c(
    theoretical_mean1, empirical_mean1_binom, empirical_mean1_unif,
    theoretical_mean2, empirical_mean2_binom, empirical_mean2_unif,
    theoretical_mean3, empirical_mean3_binom, empirical_mean3_unif
  ),
  Variance = c(
    theoretical_var1, empirical_var1_binom, empirical_var1_unif,
    theoretical_var2, empirical_var2_binom, empirical_var2_unif,
    theoretical_var3, empirical_var3_binom, empirical_var3_unif
  )
)

print(comparison_table)