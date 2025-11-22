import numpy as np
import matplotlib.pyplot as plt

# 定义函数
def f(x1, x2):
    return 6*x1 + 4*x2 - 13 - x1**2 - x2**2

# 网格范围
x1 = np.linspace(0, 3.5, 200)
x2 = np.linspace(0, 3.5, 200)
X1, X2 = np.meshgrid(x1, x2)

# 函数值
Z = f(X1, X2)

# 可行域掩码（x1 + x2 <= 3）
feasible = (X1 + X2 <= 3)

# 绘制等高线
plt.figure(figsize=(7,6))
contours = plt.contour(X1, X2, Z, levels=20, cmap='viridis')
plt.clabel(contours, inline=True, fontsize=8)

# 可行域用半透明色标出
plt.imshow(feasible, extent=(0,3.5,0,3.5), origin='lower', 
           cmap='Greys', alpha=0.2)

# 绘制边界线 x1 + x2 = 3
x = np.linspace(0,3,100)
plt.plot(x, 3-x, 'r--', label=r'$x_1 + x_2 = 3$')

# 坐标轴与标题
plt.xlim(0, 3.5)
plt.ylim(0, 3.5)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'$f(x_1, x_2) = 6x_1 + 4x_2 - 13 - x_1^2 - x_2^2$')
plt.legend()
plt.colorbar(label='f(x1, x2)')

# 保存为PDF
plt.savefig("hw5_task23_visualization.pdf", bbox_inches='tight')
plt.close()
