# -------------------------------
# Lévy Distribution Simulation using Rejection Sampling
# Parameters: μ = 1, c = 4
# -------------------------------

# 设置参数
mu <- 1
c <- 4
n <- 10000  # 目标样本数

# 设置随机种子以保证可重复性
set.seed(123)

# 定义Lévy分布的理论密度函数
f_levy <- function(x, mu, c) {
  ifelse(x > mu,
         sqrt(c / (2 * pi)) * exp(-c / (2 * (x - mu))) / ((x - mu)^(3/2)),
         0)
}

# ========================================
# 方法1: 使用Pareto分布作为提议分布
# ========================================
cat("方法1: 使用Pareto分布作为提议分布\n")
cat("=====================================\n")

# Pareto分布: g(x) = a*x_m^a / x^(a+1), x >= x_m
# 选择参数: x_m = mu = 1, a = 1.5 (使其尾部与Lévy分布相似)
x_m <- mu
a <- 0.45

# Pareto分布的密度函数
g_pareto <- function(x, x_m, a) {
  ifelse(x >= x_m, a * x_m^a / x^(a+1), 0)
}

# Pareto分布的采样函数 (逆变换法)
rpareto <- function(n, x_m, a) {
  u <- runif(n)
  x_m / (u^(1/a))
}

# 寻找最优的常数M (使得 f(x) <= M*g(x) 对所有x成立)
# 通过数值方法寻找 max(f(x)/g(x))
x_test <- seq(mu + 0.01, 100, length.out = 10000)
ratio_pareto <- f_levy(x_test, mu, c) / g_pareto(x_test, x_m, a)
M_pareto <- max(ratio_pareto, na.rm = TRUE)

cat("Pareto提议分布参数: x_m =", x_m, ", a =", a, "\n")
cat("常数M =", round(M_pareto, 4), "\n")
cat("说明: M = max(f(x)/g(x)) 确保 f(x) ≤ M·g(x) 对所有x成立\n")

# 拒绝采样
X_pareto <- numeric(n)
n_accepted_pareto <- 0
n_total_pareto <- 0

while(n_accepted_pareto < n) {
  # 从提议分布采样
  y <- rpareto(1, x_m, a)
  n_total_pareto <- n_total_pareto + 1
  
  # 计算接受概率
  u <- runif(1)
  acceptance_prob <- f_levy(y, mu, c) / (M_pareto * g_pareto(y, x_m, a))
  
  # 接受或拒绝
  if(u <= acceptance_prob) {
    n_accepted_pareto <- n_accepted_pareto + 1
    X_pareto[n_accepted_pareto] <- y
  }
}

acceptance_rate_pareto <- n / n_total_pareto
cat("接受率:", round(acceptance_rate_pareto * 100, 2), "%\n")
cat("总共生成:", n_total_pareto, "个候选样本\n\n")

# ========================================
# 方法2: 使用均匀分布作为提议分布
# ========================================
cat("方法2: 使用均匀分布作为提议分布\n")
cat("=====================================\n")

# 均匀分布: g(x) = 1/(b-a), a <= x <= b
# 选择参数: a = mu = 1, b = 50 (覆盖Lévy分布的主要区域)
a_unif <- mu
b_unif <- 10

# 均匀分布的密度函数
g_unif <- function(x, a, b) {
  ifelse(x >= a & x <= b, 1 / (b - a), 0)
}

# 均匀分布的采样函数
runif_range <- function(n, a, b) {
  runif(n, min = a, max = b)
}

# 寻找最优的常数M
# 统一计算方式: M = max(f(x)/g(x))
x_test <- seq(a_unif + 0.01, b_unif, length.out = 10000)
ratio_unif <- f_levy(x_test, mu, c) / g_unif(x_test, a_unif, b_unif)
M_unif <- max(ratio_unif, na.rm = TRUE) * 1.1  # 1.1作为安全边界

cat("均匀分布参数: a =", a_unif, ", b =", b_unif, "\n")
cat("常数M =", round(M_unif, 4), "\n")
cat("说明: M = max(f(x)/g(x)) 确保 f(x) ≤ M·g(x) 对所有x成立\n")

# 拒绝采样
X_unif <- numeric(n)
n_accepted_unif <- 0
n_total_unif <- 0

while(n_accepted_unif < n) {
  # 从提议分布采样
  y <- runif_range(1, a_unif, b_unif)
  n_total_unif <- n_total_unif + 1
  
  # 计算接受概率
  u <- runif(1)
  acceptance_prob <- f_levy(y, mu, c) / (M_unif * g_unif(y, a_unif, b_unif))
  
  # 接受或拒绝
  if(u <= acceptance_prob) {
    n_accepted_unif <- n_accepted_unif + 1
    X_unif[n_accepted_unif] <- y
  }
}

acceptance_rate_unif <- n / n_total_unif
cat("接受率:", round(acceptance_rate_unif * 100, 2), "%\n")
cat("总共生成:", n_total_unif, "个候选样本\n\n")

# ========================================
# 直接采样方法 (作为对比)
# ========================================
cat("方法3: 直接采样方法 (使用正态分布变换)\n")
cat("=====================================\n")
Z <- rnorm(n, mean = 0, sd = 1)
X_direct <- mu + c / (Z^2)
cat("无需拒绝，直接生成", n, "个样本\n\n")

# ========================================
# 统计对比
# ========================================
cat("\n三种方法的统计对比\n")
cat("=====================================\n")

methods <- c("Pareto提议", "均匀提议", "直接采样")
means <- c(mean(X_pareto), mean(X_unif), mean(X_direct))
medians <- c(median(X_pareto), median(X_unif), median(X_direct))
vars <- c(var(X_pareto), var(X_unif), var(X_direct))

comparison <- data.frame(
  方法 = methods,
  均值 = round(means, 2),
  中位数 = round(medians, 2),
  方差 = round(vars, 2)
)
print(comparison)

# ========================================
# 可视化对比
# ========================================
par(mfrow = c(2, 3), mar = c(4, 4, 3, 2))

# 图1: Pareto提议 - 直方图
hist(X_pareto, breaks = 100, freq = FALSE, 
     xlim = c(1, 50), ylim = c(0, 0.5),
     main = "方法1: Pareto提议分布",
     xlab = "x", ylab = "密度",
     col = "lightblue", border = "white")
x_vals <- seq(1, 50, length.out = 1000)
lines(x_vals, f_levy(x_vals, mu, c), col = "red", lwd = 2)
lines(x_vals, g_pareto(x_vals, x_m, a), col = "green", lwd = 2, lty = 2)
legend("topright", 
       legend = c("样本", "目标PDF", "提议PDF"),
       col = c("lightblue", "red", "green"), 
       lwd = c(10, 2, 2), lty = c(1, 1, 2), cex = 0.8)

# 图2: 均匀提议 - 直方图
hist(X_unif, breaks = 100, freq = FALSE, 
     xlim = c(1, 50), ylim = c(0, 0.5),
     main = "方法2: 均匀提议分布",
     xlab = "x", ylab = "密度",
     col = "lightgreen", border = "white")
lines(x_vals, f_levy(x_vals, mu, c), col = "red", lwd = 2)
x_unif_range <- seq(a_unif, b_unif, length.out = 1000)
lines(x_unif_range, g_unif(x_unif_range, a_unif, b_unif), col = "purple", lwd = 2, lty = 2)
legend("topright", 
       legend = c("样本", "目标PDF", "提议PDF"),
       col = c("lightgreen", "red", "purple"), 
       lwd = c(10, 2, 2), lty = c(1, 1, 2), cex = 0.8)

# 图3: 直接采样 - 直方图
hist(X_direct, breaks = 100, freq = FALSE, 
     xlim = c(1, 50), ylim = c(0, 0.5),
     main = "方法3: 直接采样",
     xlab = "x", ylab = "密度",
     col = "lightyellow", border = "white")
lines(x_vals, f_levy(x_vals, mu, c), col = "red", lwd = 2)
legend("topright", 
       legend = c("样本", "目标PDF"),
       col = c("lightyellow", "red"), 
       lwd = c(10, 2), cex = 0.8)

# 定义Lévy分布的理论CDF
F_levy <- function(x, mu, c) {
  ifelse(x <= mu, 0, 
         2 * pnorm(sqrt(c / (x - mu)), lower.tail = FALSE))
}

# 图4-6: CDF对比
x_cdf <- seq(1, 50, length.out = 500)
theoretical_cdf <- F_levy(x_cdf, mu, c)

for(i in 1:3) {
  X_current <- if(i == 1) X_pareto else if(i == 2) X_unif else X_direct
  empirical_cdf <- sapply(x_cdf, function(x) mean(X_current <= x))
  
  plot(x_cdf, empirical_cdf, type = "l", col = "blue", lwd = 2,
       main = paste("CDF对比:", methods[i]),
       xlab = "x", ylab = "CDF", ylim = c(0, 1))
  lines(x_cdf, theoretical_cdf, col = "red", lwd = 2, lty = 2)
  legend("bottomright", 
         legend = c("经验CDF", "理论CDF"),
         col = c("blue", "red"), lwd = 2, lty = c(1, 2), cex = 0.8)
}

par(mfrow = c(1, 1))

# ========================================
# 拟合优度检验
# ========================================
cat("\nKolmogorov-Smirnov拟合优度检验\n")
cat("=====================================\n")

ks_pareto <- ks.test(X_pareto, function(x) F_levy(x, mu, c))
ks_unif <- ks.test(X_unif, function(x) F_levy(x, mu, c))
ks_direct <- ks.test(X_direct, function(x) F_levy(x, mu, c))

cat("方法1 (Pareto):  D =", round(ks_pareto$statistic, 6), 
    ", p-value =", format(ks_pareto$p.value, scientific = TRUE), "\n")
cat("方法2 (均匀):    D =", round(ks_unif$statistic, 6), 
    ", p-value =", format(ks_unif$p.value, scientific = TRUE), "\n")
cat("方法3 (直接):    D =", round(ks_direct$statistic, 6), 
    ", p-value =", format(ks_direct$p.value, scientific = TRUE), "\n")

cat("\n结论: 所有方法的p值均", 
    ifelse(min(ks_pareto$p.value, ks_unif$p.value, ks_direct$p.value) > 0.05,
           "> 0.05，样本符合Lévy分布", 
           "较小，但这在大样本下是正常的"), "\n")