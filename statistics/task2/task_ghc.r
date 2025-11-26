a <- 3 / 8

# 定义 f2(x)（目标分布）
f2 <- function(x) {
  ifelse(x >= -1 & x <= 1, a * (1 - x^2),
         ifelse(x >= 2, 0.5 * exp(-(x - 2)), 0))
}

# 任务2：提议区间[-1,7]（长度=8），M=4.0
rejection_sampling_task2_naive_v2 <- function(n = 10000) {
  samples <- numeric(n)
  rejections <- 0
  accepted <- 0
  
  # 提议分布：[-1,7]（长度=8，与Task3彻底不同）
  a_range <- -1
  b_range <- 7  # 关键：与Task3的10差异明显
  g_density <- 1 / (b_range - a_range)
  
  # 高精度找f2全局最大值（x=2处，0.5）
  opt1 <- optimize(f2, c(-1,1), maximum=TRUE, tol=1e-10)
  opt2 <- optimize(f2, c(2,7), maximum=TRUE, tol=1e-10)
  max_f2 <- max(opt1$objective, opt2$objective)
  M <- max_f2 * (b_range - a_range)  # 0.5×8=4.0
  
  # 采样循环
  while(accepted < n) {
    x <- runif(1, a_range, b_range)
    u <- runif(1)
    if(u <= f2(x) / (M * g_density)) {
      accepted <- accepted + 1
      samples[accepted] <- x
    } else {
      rejections <- rejections + 1
    }
  }
  
  # 输出结果（包含M值）
  acceptance_rate <- n / (n + rejections)
  cat("task2 - acceptance rate:", round(acceptance_rate, 4), "\n")
  cat("task2 - total proposals:", n + rejections, "\n")
  cat("task2 - 自动计算的 M =", round(M, 4), "\n")  # 理论值4.0
  
  # 返回采样结果和M值（供总结用）
  return(list(samples = samples, M = M))
}

# 定义 f3(x)（目标分布）
f3 <- function(x) {
  ifelse(x >= -1 & x < 1, (1/8) * (x + 1),
         ifelse(x >= 1 & x < 2, 1/4,
                ifelse(x >= 2, 0.5 * exp(-(x - 2)), 0)))
}

# 任务3：提议区间[-1,10]（长度=11），M=5.5
rejection_sampling_task3_naive <- function(n = 10000) {
  samples <- numeric(n)
  rejections <- 0
  accepted <- 0
  
  # 提议分布：[-1,10]（长度=11，与Task2不同）
  a_range <- -1
  b_range <- 10  # 关键：与Task3的7差异明显
  g_density <- 1 / (b_range - a_range)
  
  # 高精度找f3全局最大值（x=2处，0.5）
  opt1 <- optimize(f3, c(-1,1), maximum=TRUE, tol=1e-10)
  opt2 <- optimize(f3, c(1,2), maximum=TRUE, tol=1e-10)
  opt3 <- optimize(f3, c(2,10), maximum=TRUE, tol=1e-10)
  max_f3 <- max(opt1$objective, opt2$objective, opt3$objective)
  M <- max_f3 * (b_range - a_range)  # 0.5×11=5.5
  
  # 采样循环
  while(accepted < n) {
    x <- runif(1, a_range, b_range)
    u <- runif(1)
    if(u <= f3(x) / (M * g_density)) {
      accepted <- accepted + 1
      samples[accepted] <- x
    } else {
      rejections <- rejections + 1
    }
  }
  
  # 输出结果（包含M值）
  acceptance_rate <- n / (n + rejections)
  cat("task3 - acceptance rate:", round(acceptance_rate, 4), "\n")
  cat("task3 - total proposals:", n + rejections, "\n")
  cat("task3 - 自动计算的 M =", round(M, 4), "\n")  # 理论值5.5
  
  # 返回采样结果和M值（供总结用）
  return(list(samples = samples, M = M))
}

# ============================================================================
# 执行模拟（确保M不同）
# ============================================================================

set.seed(123)
N <- 10000

# 任务2执行（返回samples和M）
cat("\n========== 任务2：拒绝采样（提议区间[-1,7]）==========\n")
time2_reject_v2 <- system.time({
  res2 <- rejection_sampling_task2_naive_v2(N)
})
samples2_v2 <- res2$samples
M2 <- res2$M
cat("task2 - computation time:", round(time2_reject_v2[3], 4), "seconds\n")

# 任务3执行（返回samples和M）
cat("\n========== 任务3：拒绝采样（提议区间[-1,10]）==========\n")
time3_reject <- system.time({
  res3 <- rejection_sampling_task3_naive(N)
})
samples3 <- res3$samples
M3 <- res3$M
cat("task3 - computation time:", round(time3_reject[3], 4), "seconds\n")

# ============================================================================
# 统计量总结（修正变量作用域）
# ============================================================================

cat("\n========== summary ==========\n")
cat("task2:\n")
cat("  empirical E:", round(mean(samples2_v2), 4), "\n")
cat("  empirical D:", round(var(samples2_v2), 4), "\n")
cat("  M:", round(M2, 4), "| time:", round(time2_reject_v2[3], 4), "s\n")

cat("\ntask3:\n")
cat("  empirical E:", round(mean(samples3), 4), "\n")
cat("  empirical D:", round(var(samples3), 4), "\n")
cat("  M:", round(M3, 4), "| time:", round(time3_reject[3], 4), "s\n")

# ============================================================================
# 保存图像（匹配各自提议区间）
# ============================================================================

pdf("R_reject_naive_final.pdf", width = 10, height = 8)
par(mfrow = c(2, 2))

# 任务2 - 密度（xlim=[-1,7]）
hist(samples2_v2, breaks = 50, freq = FALSE, main = "task2: M=4.0 (interval [-1,7])",
     xlab = "x", ylab = "pdf", col = "lightblue", xlim = c(-1,7))
x_seq <- seq(-1,7, length.out=1000)
lines(x_seq, sapply(x_seq, f2), col="red", lwd=2)
legend("topright", c("empirical", "theoretical"), 
       col=c("lightblue", "red"), lwd=c(10,2), cex=0.8)

# 任务2 - CDF
plot(ecdf(samples2_v2), main = "task2: cdf (M=4.0)",
     xlab = "x", ylab = "F(x)", col = "blue", lwd = 2)

# 任务3 - 密度（xlim=[-1,10]）
hist(samples3, breaks = 50, freq = FALSE, main = "task3: M=5.5 (interval [-1,10])",
     xlab = "x", ylab = "pdf", col = "lightgreen", xlim = c(-1,10))
x_seq <- seq(-1,10, length.out=1000)
lines(x_seq, sapply(x_seq, f3), col="red", lwd=2)
legend("topright", c("empirical", "theoretical"), 
       col=c("lightgreen", "red"), lwd=c(10,2), cex=0.8)

# 任务3 - CDF
plot(ecdf(samples3), main = "task3: cdf (M=5.5)",
     xlab = "x", ylab = "F(x)", col = "darkgreen", lwd = 2)

dev.off()
cat("\n✅ 图像已保存为文件: R_reject_naive_final.pdf\n")
