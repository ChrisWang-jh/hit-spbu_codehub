# Linear Regression Task
# Joint density: f(x,y) = c(x + y^2) on [0,1]^2 with x + y < 1

# ============================================================================
# Part 1: 理论计算
# ============================================================================

# 归一化常数 c = 4 (已通过积分验证)
c <- 4

# 理论条件期望 E[Y|X=x]
theoretical_cond_exp <- function(x) {
  return(3 * (1 - x) * (1 + x^2) / (4 * (1 + x + x^2)))
}

# 数值积分计算理论回归线系数
calculate_theoretical_regression <- function(n_grid = 1000) {
  # 初始化
  EX <- 0
  EY <- 0
  EX2 <- 0
  EXY <- 0
  total_prob <- 0
  
  # 网格积分
  dx <- 1 / n_grid
  dy <- 1 / n_grid
  
  for (i in 0:(n_grid - 1)) {
    x <- i / n_grid
    for (j in 0:(n_grid - 1)) {
      y <- j / n_grid
      
      if (x + y < 1) {
        prob <- c * (x + y^2) * dx * dy
        EX <- EX + x * prob
        EY <- EY + y * prob
        EX2 <- EX2 + x^2 * prob
        EXY <- EXY + x * y * prob
        total_prob <- total_prob + prob
      }
    }
  }
  
  # 归一化
  EX <- EX / total_prob
  EY <- EY / total_prob
  EX2 <- EX2 / total_prob
  EXY <- EXY / total_prob
  
  # 计算回归系数: beta = Cov(X,Y) / Var(X)
  beta <- (EXY - EX * EY) / (EX2 - EX^2)
  alpha <- EY - beta * EX
  
  return(list(alpha = alpha, beta = beta, EX = EX, EY = EY, EX2 = EX2, EXY = EXY))
}

# 计算理论回归线 E[Y|X=x] = 3(1-x)(1+x²) / [4(1+x+x²)]
theoretical_reg <- calculate_theoretical_regression()
cat("Theoretical Regression Line:\n")
cat(sprintf("Y = %.4f + %.4f * X\n", theoretical_reg$alpha, theoretical_reg$beta))
cat(sprintf("E[X] = %.4f, E[Y] = %.4f\n\n", theoretical_reg$EX, theoretical_reg$EY))

# ============================================================================
# Part 2: 从分布中抽样 (Rejection Sampling)
# ============================================================================

# 拒绝抽样方法生成样本
generate_sample <- function(n, c = 4) {
  samples <- matrix(nrow = 0, ncol = 2)
  max_density <- c * 1  # maximum value of c(x + y^2) on the region
  
  attempts <- 0
  max_attempts <- n * 1000
  
  while (nrow(samples) < n && attempts < max_attempts) {
    # 在 [0,1]^2 中均匀抽样
    x <- runif(1)
    y <- runif(1)
    
    # 检查是否在定义域内
    if (x + y < 1) {
      # 计算密度
      density <- c * (x + y^2)
      u <- runif(1) * max_density
      
      # 接受-拒绝步骤
      if (u <= density) {
        samples <- rbind(samples, c(x, y))
      }
    }
    attempts <- attempts + 1
  }
  
  colnames(samples) <- c("x", "y")
  return(as.data.frame(samples))
}

# ============================================================================
# Part 3: 模拟不同样本量并计算OLS估计
# ============================================================================

# 样本量
sample_sizes <- c(50, 100, 200, 500, 1000)

# 存储结果
ols_results <- list()

# 打开PDF设备保存图片
pdf("linear_regression.pdf", width = 12, height = 8)

# 设置绘图参数
par(mfrow = c(2, 3), mar = c(4, 4, 2, 1))

for (n in sample_sizes) {
  cat(sprintf("\n=== Sample size: %d ===\n", n))
  
  # 生成样本
  data <- generate_sample(n)
  
  # OLS回归
  ols_model <- lm(y ~ x, data = data)
  ols_alpha <- coef(ols_model)[1]
  ols_beta <- coef(ols_model)[2]
  
  # 存储结果
  ols_results[[as.character(n)]] <- list(
    alpha = ols_alpha,
    beta = ols_beta,
    n = n
  )
  
  cat(sprintf("OLS Regression: Y = %.4f + %.4f * X\n", ols_alpha, ols_beta))
  
  # 绘图
  plot(data$x, data$y, 
       pch = 16, col = rgb(0, 0, 1, 0.3), 
       xlim = c(0, 1), ylim = c(0, 0.6),
       xlab = "X", ylab = "Y",
       main = sprintf("n = %d", n))
  
  # 添加理论条件期望曲线 (橙色)
  x_seq <- seq(0, 0.95, length.out = 100)
  y_cond <- sapply(x_seq, theoretical_cond_exp)
  lines(x_seq, y_cond, col = "orange", lwd = 2)
  
  # 添加理论回归线 (绿色虚线)
  y_theo_reg <- theoretical_reg$alpha + theoretical_reg$beta * x_seq
  lines(x_seq, y_theo_reg, col = "green", lwd = 2, lty = 2)
  
  # 添加OLS回归线 (红色虚线)
  abline(ols_model, col = "red", lwd = 2, lty = 3)
  
  # 添加图例
  if (n == 50) {
    legend("topright", 
           legend = c("Data", "E[Y|X] (Theoretical)", 
                      "Theo. Regression", "OLS Regression"),
           col = c(rgb(0, 0, 1, 0.3), "orange", "green", "red"),
           lty = c(NA, 1, 2, 3),
           pch = c(16, NA, NA, NA),
           lwd = c(NA, 2, 2, 2),
           cex = 0.7)
  }
}

# 关闭PDF设备
dev.off()
cat("\nPlots saved to linear_regression.pdf\n")

# ============================================================================
# Part 4: 比较结果
# ============================================================================

cat("\n\n=== Summary of Results ===\n")
cat(sprintf("Theoretical Regression: Y = %.4f + %.4f * X\n\n", 
            theoretical_reg$alpha, theoretical_reg$beta))

cat("OLS Estimates:\n")
for (n in sample_sizes) {
  result <- ols_results[[as.character(n)]]
  cat(sprintf("n = %4d: Y = %.4f + %.4f * X\n", 
              result$n, result$alpha, result$beta))
}
