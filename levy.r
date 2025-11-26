# ==============================================================================
# 0. 环境初始化与定义
# ==============================================================================
rm(list=ls())       # 清空工作空间
set.seed(2023)      # 设置随机种子
par(mfrow = c(2, 2)) # 设置绘图布局: 2行2列

# 定义 Lévy 分布参数
mu <- 1
c_param <- 4
N <- 10000          # 模拟样本量

# 定义存储接受率的列表，用于最后打印
efficiency_results <- list()

# ------------------------------------------------------------------------------
# 目标分布：Lévy Probability Density Function (PDF)
# ------------------------------------------------------------------------------
dlevy <- function(x) {
  y <- numeric(length(x))
  idx <- x > mu
  if(any(idx)) {
    val <- x[idx]
    # 公式: sqrt(c/2pi) * exp(-c/(2(x-mu))) / (x-mu)^(3/2)
    term1 <- sqrt(c_param / (2 * pi))
    term2 <- exp(-c_param / (2 * (val - mu)))
    term3 <- (val - mu)^(1.5)
    y[idx] <- term1 * term2 / term3
  }
  return(y)
}

# ------------------------------------------------------------------------------
# 通用拒绝采样运行函数
# ------------------------------------------------------------------------------
run_rejection <- function(n, M, r_prop, d_prop, name) {
  samples <- numeric(n)
  count <- 0
  attempts <- 0
  
  while(count < n) {
    attempts <- attempts + 1
    
    # 1. 从建议分布生成候选值 Y
    Y <- r_prop(1)
    
    # 2. 生成均匀随机数 U
    U <- runif(1)
    
    # 3. 计算接受比率
    g_val <- d_prop(Y)
    
    # 边界保护：如果 g(Y) <= 0 (超出支撑集)，直接拒绝
    if(g_val <= 0) next 
    
    ratio <- dlevy(Y) / (M * g_val)
    
    if(U <= ratio) {
      count <- count + 1
      samples[count] <- Y
    }
  }
  
  eff <- n / attempts
  cat(sprintf("[%s] 完成。M = %.4f\n", name, M))
  return(list(samples = samples, efficiency = eff))
}

# ==============================================================================
# 方法 1: Sampling via the Normal Distribution (正态变换法)
# ==============================================================================
cat(">>> 方法 1: 正态变换法 (Exact Method)...\n")
Z <- rnorm(N)
X_normal <- mu + c_param / Z^2

# 绘图 Method 1
x_grid <- seq(1.01, 20, length.out = 1000)
y_true <- dlevy(x_grid)
hist(X_normal[X_normal <= 20], breaks = 50, prob = TRUE, col = "lightblue", border = "white",
     main = "1. Normal Transform", xlab = "x")
lines(x_grid, y_true, col = "red", lwd = 2)


# ==============================================================================
# 方法 2a: Pareto Proposal (最佳参数 alpha=0.5)
# ==============================================================================
cat("\n>>> 方法 2a: Pareto Proposal (alpha=0.5)...\n")

pareto_xm <- 1
pareto_alpha <- 0.5 # 匹配 Lévy 尾部 (x^-1.5)

dpareto <- function(x) {
  ifelse(x >= pareto_xm, pareto_alpha * pareto_xm^pareto_alpha / x^(pareto_alpha + 1), 0)
}
rpareto <- function(n) {
  u <- runif(n)
  pareto_xm / (1 - u)^(1 / pareto_alpha)
}

# 计算 M
ratio_fn_p <- function(x) { dlevy(x) / dpareto(x) }
opt_p <- optimize(ratio_fn_p, interval = c(1.001, 100), maximum = TRUE)
M_pareto <- opt_p$objective * 1.05

# 执行采样
res_pareto <- run_rejection(N, M_pareto, rpareto, dpareto, "Pareto")
X_pareto <- res_pareto$samples
efficiency_results$Pareto <- res_pareto$efficiency

# 绘图 Method 2a
hist(X_pareto[X_pareto <= 20], breaks = 50, prob = TRUE, col = "lightgreen", border = "white",
     main = "2a. Pareto (alpha=0.5)", xlab = "x")
lines(x_grid, y_true, col = "red", lwd = 2)
# 图例中仅显示 M，不显示接受率
legend("topright", paste("M =", round(M_pareto, 2)), bty="n", cex=0.8)


# ==============================================================================
# 方法 2b: Uniform Proposal (区间缩小至 [1, 10])
# ==============================================================================
cat("\n>>> 方法 2b: Uniform Proposal [1, 10]...\n")

unif_min <- 1
unif_max <- 10  # 区间 [1, 10]

dunif_c <- function(x) dunif(x, min = unif_min, max = unif_max)
runif_c <- function(n) runif(n, min = unif_min, max = unif_max)

# 计算 M (仅在 1 到 10 之间)
ratio_fn_u <- function(x) { dlevy(x) / dunif_c(x) }
opt_u <- optimize(ratio_fn_u, interval = c(1.001, 10), maximum = TRUE)
M_unif <- opt_u$objective * 1.05

# 执行采样
res_unif <- run_rejection(N, M_unif, runif_c, dunif_c, "Uniform")
X_unif <- res_unif$samples
efficiency_results$Uniform <- res_unif$efficiency

# 绘图 Method 2b
hist(X_unif[X_unif <= 20], breaks = 50, prob = TRUE, col = "orange", border = "white",
     main = "2b. Uniform [1, 10]", xlab = "x")
lines(x_grid, y_true, col = "red", lwd = 2)
# 图例中仅显示 M，不显示接受率
legend("topright", paste("M =", round(M_unif, 2)), bty="n", cex=0.8)


# ==============================================================================
# 方法 3: Own Sight - Shifted Inverse Gamma (平移逆伽马)
# ==============================================================================
cat("\n>>> 方法 3: Own Sight - Shifted Inverse Gamma (alpha=0.4)...\n")

ig_shape <- 0.4
ig_scale <- 1

# 平移逆伽马 PDF: Y ~ InvGamma, X = mu + Y
dinvgamma_shifted <- function(x) {
  y <- x - mu
  ifelse(y > 0, 
         (ig_scale^ig_shape / gamma(ig_shape)) * y^(-ig_shape - 1) * exp(-ig_scale/y),
         0)
}
# 平移逆伽马 RNG
rinvgamma_shifted <- function(n) {
  g <- rgamma(n, shape = ig_shape, rate = ig_scale) 
  return(mu + 1/g) 
}

# 计算 M
ratio_fn_ig <- function(x) { dlevy(x) / dinvgamma_shifted(x) }
opt_ig <- optimize(ratio_fn_ig, interval = c(1.001, 500), maximum = TRUE)
M_ig <- opt_ig$objective * 1.05

# 执行采样
res_ig <- run_rejection(N, M_ig, rinvgamma_shifted, dinvgamma_shifted, "Inverse Gamma")
X_own <- res_ig$samples
efficiency_results$InvGamma <- res_ig$efficiency

# 绘图 Method 3
hist(X_own[X_own <= 20], breaks = 50, prob = TRUE, col = "violet", border = "white",
     main = "3. Own Sight (InvGamma)", xlab = "x")
lines(x_grid, y_true, col = "red", lwd = 2)
# 图例中仅显示 M，不显示接受率
legend("topright", paste("M =", round(M_ig, 2)), bty="n", cex=0.8)

# 恢复默认绘图设置
par(mfrow = c(1, 1))

# ==============================================================================
# 最终结果打印：接受率对比
# ==============================================================================
cat("\n=======================================================\n")
cat("            拒绝采样算法效率对比 (Acceptance Rates)      \n")
cat("=======================================================\n")
cat(sprintf("1. Pareto Proposal (alpha=0.5):   %.2f%%\n", efficiency_results$Pareto * 100))
cat(sprintf("2. Uniform Proposal [1, 10]:      %.2f%%\n", efficiency_results$Uniform * 100))
cat(sprintf("3. InvGamma Proposal (alpha=0.4): %.2f%%\n", efficiency_results$InvGamma * 100))
cat("=======================================================\n")
cat("注：Uniform 分布仅模拟了区间 [1, 10] 内的数据。\n")