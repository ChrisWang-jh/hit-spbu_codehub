# ============================================================================
# 任务2：拒绝采样实现 - 朴素方法
# ============================================================================

a <- 3 / 8

# 定义 f2(x)（目标分布）
f2 <- function(x) {
  ifelse(x >= -1 & x <= 1, a * (1 - x^2),
         ifelse(x >= 2, 0.5 * exp(-(x - 2)), 0))
}

# 朴素拒绝采样：使用均匀分布作为提议分布
# 为了真正体现"朴素"特性，使用更大的提议范围和更保守的包络常数
rejection_sampling_task2_naive <- function(n = 10000) {
  samples <- numeric(n)
  rejections <- 0
  accepted <- 0
  
  # 朴素方法：使用一个很宽的均匀分布作为提议
  # 范围设为 [-1, 15]，远大于实际需要的范围
  a_range <- -1
  b_range <- 15
  g_density <- 1 / (b_range - a_range)  # = 1/16
  
  # 找到 f2(x) / g(x) 的最大值作为包络常数
  # f2(x) 在 x=0 处达到最大值 3/8
  # 在 x=2 处，指数部分起始值为 0.5
  # 所以 M 应该是 max(f2(x)) / g(x) = 0.5 / (1/16) = 8
  # 这样包络函数 M*g(x) = 8 * (1/16) = 0.5，刚好覆盖 f2(x)
  M <- 8
  
  while(accepted < n) {
    # 从提议分布中采样
    x <- runif(1, a_range, b_range)
    
    # 计算接受概率
    u <- runif(1)
    
    # 接受条件: u <= f(x) / (M * g(x))
    acceptance_prob <- f2(x) / (M * g_density)
    
    if(u <= acceptance_prob) {
      accepted <- accepted + 1
      samples[accepted] <- x
    } else {
      rejections <- rejections + 1
    }
  }
  
  # 输出接受率和总提议次数
  acceptance_rate <- n / (n + rejections)
  cat("task2 - acceptance rate:", round(acceptance_rate, 4), "\n")
  cat("task2 - total proposals:", n + rejections, "\n")
  
  return(samples)
}

# ============================================================================
# 任务3：拒绝采样实现 - 朴素方法
# ============================================================================

f3 <- function(x) {
  ifelse(x >= -1 & x < 1, (1/8) * (x + 1),
         ifelse(x >= 1 & x < 2, 1/4,
                ifelse(x >= 2, 0.5 * exp(-(x - 2)), 0)))
}

# 最朴素的拒绝采样：使用单一的均匀分布作为提议分布
# 为了真正体现"朴素"特性，使用更大的提议范围
rejection_sampling_task3_naive <- function(n = 10000) {
  samples <- numeric(n)
  rejections <- 0
  accepted <- 0
  
  # 朴素方法：使用一个很宽的均匀分布作为提议
  # 范围设为 [-1, 15]
  a_range <- -1
  b_range <- 15
  g_density <- 1 / (b_range - a_range)  # = 1/16
  
  # 找到f3(x)的最大值
  # 在[-1,1)上: max = (1/8) * 2 = 1/4 (在x=1处)
  # 在[1,2)上: f(x) = 1/4
  # 在[2,∞)上: max = 0.5 (在x=2处)
  # 所以全局最大值是 0.5
  # M = max(f3(x)) / g(x) = 0.5 / (1/16) = 8
  M <- 8
  
  while(accepted < n) {
    # 从提议分布采样
    x <- runif(1, a_range, b_range)
    
    # 计算接受概率
    u <- runif(1)
    
    # 接受条件: u <= f(x) / (M * g(x))
    acceptance_prob <- f3(x) / (M * g_density)
    
    if(u <= acceptance_prob) {
      accepted <- accepted + 1
      samples[accepted] <- x
    } else {
      rejections <- rejections + 1
    }
  }
  
  acceptance_rate <- n / (n + rejections)
  cat("task3 - acceptance rate:", round(acceptance_rate, 4), "\n")
  cat("task3 - total proposals:", n + rejections, "\n")
  
  return(samples)
}

# ============================================================================
# 执行模拟与可视化（带时间测量）
# ============================================================================

set.seed(123)
N <- 10000

# 任务2 - 测量运算时间
cat("\n========== 任务2：拒绝采样（朴素方法）==========\n")
time2_reject <- system.time({
  samples2 <- rejection_sampling_task2_naive(N)
})
cat("task2 - computation time:", round(time2_reject[3], 4), "seconds\n")

# 任务3 - 测量运算时间
cat("\n========== 任务3：拒绝采样（朴素方法）==========\n")
time3_reject <- system.time({
  samples3 <- rejection_sampling_task3_naive(N)
})
cat("task3 - computation time:", round(time3_reject[3], 4), "seconds\n")

# ============================================================================
# 统计量总结
# ============================================================================

cat("\n========== summary ==========\n")
cat("task2:\n")
cat("  empirical E:", round(mean(samples2), 4), "\n")
cat("  empirical D:", round(var(samples2), 4), "\n")
cat("  time:", round(time2_reject[3], 4), "s\n")

cat("\ntask3:\n")
cat("  empirical E:", round(mean(samples3), 4), "\n")
cat("  empirical D:", round(var(samples3), 4), "\n")
cat("  time:", round(time3_reject[3], 4), "s\n")

# ============================================================================
# 保存图像为 R_reject_naive.pdf
# ============================================================================

pdf("R_reject_naive.pdf", width = 10, height = 8)

par(mfrow = c(2, 2))

# 任务2 - 密度
hist(samples2, breaks = 50, freq = FALSE, main = "task2: rejection sampling (naive)",
     xlab = "x", ylab = "pdf", col = "lightblue", xlim = c(-1, 6))
x_seq <- seq(-1, 6, length.out = 1000)
lines(x_seq, sapply(x_seq, f2), col = "red", lwd = 2)
legend("topright", c("empirical", "theoretical"), 
       col = c("lightblue", "red"), lwd = c(10, 2), cex = 0.8)

# 任务2 - CDF
plot(ecdf(samples2), main = "task2: cdf (naive)",
     xlab = "x", ylab = "F(x)", col = "blue", lwd = 2)

# 任务3 - 密度
hist(samples3, breaks = 50, freq = FALSE, main = "task3: rejection sampling (naive)",
     xlab = "x", ylab = "pdf", col = "lightgreen", xlim = c(-1, 6))
x_seq <- seq(-1, 6, length.out = 1000)
lines(x_seq, sapply(x_seq, f3), col = "red", lwd = 2)
legend("topright", c("empirical", "theoretical"), 
       col = c("lightgreen", "red"), lwd = c(10, 2), cex = 0.8)

# 任务3 - CDF
plot(ecdf(samples3), main = "task3: cdf (naive)",
     xlab = "x", ylab = "F(x)", col = "darkgreen", lwd = 2)

dev.off()
cat("\n✅ 图像已保存为文件: R_reject_naive.pdf\n")