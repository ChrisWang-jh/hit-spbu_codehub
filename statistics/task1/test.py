# pareto_demo.py
# 帕累托分布：理论 vs 经验（a=4, xm=2）
# 运行：python pareto_demo.py

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# -------- 参数区：可按需修改 --------
ALPHA = 4.0       # 形状参数 a
XM = 2.0          # 尺度参数 x_m
N = 10000         # 样本量
SEED = 42         # 随机种子
SAVE_FIGS = True  # 是否保存图像
# -----------------------------------

@dataclass
class Moments:
    mean_theory: float
    var_theory: float
    mean_emp: float
    var_emp: float
    ks_stat: float

def pareto_pdf(x, a=ALPHA, xm=XM):
    """帕累托分布的 PDF；x 为标量或数组。"""
    x = np.asarray(x)
    out = np.zeros_like(x, dtype=float)
    mask = x >= xm
    out[mask] = a * xm**a / (x[mask]**(a + 1))
    return out

def pareto_cdf(x, a=ALPHA, xm=XM):
    """帕累托分布的 CDF；x 为标量或数组。"""
    x = np.asarray(x)
    out = np.zeros_like(x, dtype=float)
    mask = x >= xm
    out[~mask] = 0.0
    out[mask] = 1.0 - (xm / x[mask])**a
    return out

def simulate_pareto(n=N, a=ALPHA, xm=XM, seed=SEED):
    """使用 numpy 生成帕累托样本：X = xm * (1 + Pareto(a))"""
    rng = np.random.default_rng(seed)
    return xm * (1.0 + rng.pareto(a, size=n))

def compute_moments(samples, a=ALPHA, xm=XM):
    """计算理论与经验矩，以及 KS 统计量。"""
    # 理论矩（存在性：均值 a>1，方差 a>2）
    mean_th = a * xm / (a - 1) if a > 1 else np.nan
    var_th = (a * xm**2) / ((a - 1)**2 * (a - 2)) if a > 2 else np.nan

    # 经验量
    mean_emp = float(np.mean(samples))
    var_emp = float(np.var(samples, ddof=1))

    # KS 统计量（经验 CDF 与理论 CDF 的 supremum 差）
    xs = np.sort(samples)
    ecdf = np.arange(1, len(samples) + 1) / len(samples)
    ks_stat = float(np.max(np.abs(ecdf - pareto_cdf(xs, a=a, xm=xm))))

    return Moments(mean_th, var_th, mean_emp, var_emp, ks_stat)

def plot_cdf(samples, a=ALPHA, xm=XM, save=SAVE_FIGS):
    xs = np.sort(samples)
    ecdf = np.arange(1, len(samples) + 1) / len(samples)

    grid = np.linspace(xm, np.percentile(samples, 99.9), 800)
    cdf_th = pareto_cdf(grid, a=a, xm=xm)

    plt.figure()
    plt.step(xs, ecdf, where="post", label="Empirical CDF")
    plt.plot(grid, cdf_th, label="Theoretical CDF")
    plt.title(f"Pareto(a={a}, xm={xm}): Empirical vs Theoretical CDF")
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig("pareto_cdf_compare.png", dpi=160)
    plt.show()

def plot_pdf(samples, a=ALPHA, xm=XM, save=SAVE_FIGS):
    plt.figure()
    # 经验密度（直方图）
    count, bins, _ = plt.hist(samples, bins=60, density=True, alpha=0.5, label="Empirical density")
    centers = 0.5 * (bins[1:] + bins[:-1])
    plt.plot(centers, pareto_pdf(centers, a=a, xm=xm), label="Theoretical PDF")
    plt.title(f"Pareto(a={a}, xm={xm}): Empirical density vs Theoretical PDF")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig("pareto_pdf_compare.png", dpi=160)
    plt.show()

def main():
    samples = simulate_pareto()
    m = compute_moments(samples)

    # 打印对比表
    print("=== Pareto 分布（理论 vs 经验） ===")
    print(f"a = {ALPHA}, xm = {XM}, n = {N}")
    print(f"理论均值 E[X]      : {m.mean_theory:.6f}")
    print(f"经验均值           : {m.mean_emp:.6f}")
    print(f"理论方差 Var[X]    : {m.var_theory:.6f}")
    print(f"经验方差           : {m.var_emp:.6f}")
    print(f"KS 统计量           : {m.ks_stat:.6f}")
    if SAVE_FIGS:
        print("已保存图像：pareto_cdf_compare.png, pareto_pdf_compare.png")

    # 绘图
    plot_cdf(samples)
    plot_pdf(samples)

if __name__ == "__main__":
    main()
