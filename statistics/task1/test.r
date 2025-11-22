set.seed(42)

## ---------------- 参数 ----------------
alpha <- 4
xm    <- 2
n     <- 1000      # 样本量，可改
do_clt <- FALSE    # 是否做CLT可视化（大量重复模拟，会较慢）

## --------- Pareto 分布的函数族（自定义） ---------
dpareto <- function(x, alpha, xm) ifelse(x >= xm, alpha * xm^alpha / x^(alpha + 1), 0)
ppareto <- function(x, alpha, xm) ifelse(x <  xm, 0, 1 - (xm / x)^alpha)
qpareto <- function(p, alpha, xm) xm * (1 - p)^(-1/alpha)
rpareto <- function(n, alpha, xm) qpareto(runif(n), alpha, xm)

## ---------------- 抽样 ----------------
x <- rpareto(n, alpha, xm)

## ---------------- 理论量 ----------------
mu_th  <- alpha * xm / (alpha - 1)                            # E[X]  = 8/3
var_th <- alpha * xm^2 / ((alpha - 1)^2 * (alpha - 2))        # VarX  = 8/9

## ---------------- 经验量 ----------------
mean_emp <- mean(x)
var_emp  <- var(x)

cat("=== Theoretical vs Empirical ===\n")
cat(sprintf("E[X]:   theory = %.6f, empirical = %.6f,  rel.err = %.2f%%\n",
            mu_th, mean_emp, 100 * abs(mean_emp - mu_th) / mu_th))
cat(sprintf("Var[X]: theory = %.6f, empirical = %.6f,  rel.err = %.2f%%\n\n",
            var_th, var_emp, 100 * abs(var_emp - var_th) / var_th))

## ---------------- KS 检验（对 F(x)） ----------------
ks <- ks.test(x, function(y) ppareto(y, alpha = alpha, xm = xm))
cat("=== Kolmogorov–Smirnov test ===\n")
cat(sprintf("D = %.4f, p-value = %.4g\n\n", ks$statistic, ks$p.value))

## ---------------- 作图：密度 & CDF ----------------
oldpar <- par(no.readonly = TRUE)
on.exit(par(oldpar), add = TRUE)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

# (1) 直方图 + 理论密度
x_max <- quantile(x, 0.99)      # 为了看清主体，截到 99% 分位
hist(x, breaks = "FD", freq = FALSE, col = "lightblue",
     border = "white",
     main = sprintf("Histogram & f(x)", alpha, xm),
     xlab = "x", xlim = c(xm, x_max))
curve(dpareto(x, alpha, xm), from = xm, to = x_max,
      add = TRUE, col = "red", lwd = 2)
legend("topright", bty = "n",
       legend = c("Empirical", "Theoretical density f(x)"),
       col = c("lightblue", "red"), lty = c(1,1), lwd = c(10,2))

# (2) 经验CDF + 理论CDF
plot(ecdf(x), main = expression(paste("ECDF ", hat(F)[n], " vs TCDF F(x)")),
     col = "blue", lwd = 2, xlab = "x", ylab = "CDF",
     xlim = c(xm, x_max))
curve(ppareto(x, alpha, xm), from = xm, to = x_max,
      add = TRUE, col = "red", lwd = 2, lty = 2)
legend("bottomright", bty = "n",
       legend = c(expression(hat(F)[n](x)), "F(x)"),
       col = c("blue", "red"), lty = c(1,2), lwd = 2)

# ## ----------------（可选）CLT 可视化：样本均值近似正态 ----------------
# if (do_clt) {
#   set.seed(123)
#   M <- 100000
#   means <- replicate(M, mean(rpareto(n, alpha, xm)))
#   sd_mean <- sqrt(var_th / n)

#   dev.new()  # 开新图窗（有些环境不支持则忽略）
#   hist(means, breaks = 100, freq = FALSE,
#        main = sprintf("CLT check: sample mean (n=%d, M=%d)", n, M),
#        xlab = "sample mean", col = "lightgray", border = "white")
#   curve(dnorm(x, mean = mu_th, sd = sd_mean), add = TRUE, col = "red", lwd = 2)
#   legend("topright", bty = "n",
#          legend = c("Empirical means", "Normal approx"),
#          col = c("lightgray", "red"), lty = c(1,1), lwd = c(10,2))
# }
