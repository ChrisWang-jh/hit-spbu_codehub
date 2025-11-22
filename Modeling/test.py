import numpy as np
from scipy.linalg import solve, cholesky
import warnings

def active_set_qp(G, c, A, b, x0, W0, max_iter=100):
    """
    凸二次规划的有效集方法
    
    参数:
    G: 二次项矩阵 (n x n)
    c: 线性项向量 (n,)
    A: 约束矩阵 (m x n)
    b: 约束右端项 (m,)
    x0: 初始点 (n,)
    W0: 初始工作集 (约束索引列表)
    max_iter: 最大迭代次数
    
    返回:
    x: 最优解
    history: 迭代历史
    """
    n = len(x0)  # 变量维度
    m = len(b)   # 约束数量
    
    x = x0.copy()
    W = W0.copy()
    history = []
    
    for k in range(max_iter):
        # 确保工作集中的约束在当前点活跃
        for i in W:
            if abs(A[i].dot(x) - b[i]) > 1e-10:
                # 如果约束不活跃，从工作集中移除
                W.remove(i)
        
        print(f"\n迭代 {k}: x = [{x[0]:.6f}, {x[1]:.6f}], 工作集 W = {W}")
        
        # 构建等式约束矩阵
        if len(W) > 0:
            A_eq = A[W]
            b_eq = b[W]
        else:
            A_eq = np.zeros((0, n))
            b_eq = np.zeros(0)
        
        # 步骤4: 求解等式约束子问题
        try:
            if len(W) > 0:
                # 构建KKT系统
                KKT = np.block([[G, A_eq.T], 
                               [A_eq, np.zeros((len(W), len(W)))]])
                rhs = np.concatenate([-G.dot(x) - c, np.zeros(len(W))])
                
                # 使用更稳定的求解方法
                solution = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
                p = solution[:n]
            else:
                # 无约束情况
                p = -np.linalg.solve(G, G.dot(x) + c)
        except np.linalg.LinAlgError:
            # 如果求解失败，使用梯度下降方向
            grad = G.dot(x) + c
            p = -grad / (np.linalg.norm(grad) + 1e-10)
        
        print(f"搜索方向 p = [{p[0]:.6f}, {p[1]:.6f}], 范数 = {np.linalg.norm(p):.6f}")
        
        # 步骤5-9: 如果p≈0，检查最优性
        if np.linalg.norm(p) < 1e-8:
            print("p ≈ 0，检查最优性条件")
            
            if len(W) == 0:
                print("无约束，达到最优")
                history.append({'x': x.copy(), 'p': p, 'status': 'optimal'})
                break
            
            # 计算拉格朗日乘子
            try:
                if len(W) > 0:
                    A_eq = A[W]
                    # 使用最小二乘求解拉格朗日乘子
                    lambdas = np.linalg.lstsq(A_eq.dot(A_eq.T), A_eq.dot(G.dot(x) + c), rcond=None)[0]
                    print(f"拉格朗日乘子 λ = {lambdas}")
                    
                    # 检查不等式约束的乘子非负
                    if np.all(lambdas >= -1e-8):
                        print("所有乘子非负，达到最优")
                        history.append({'x': x.copy(), 'p': p, 'status': 'optimal'})
                        break
                    else:
                        # 移除最负的乘子对应的约束
                        j = np.argmin(lambdas)
                        removed_constraint = W[j]
                        W = [W[i] for i in range(len(W)) if i != j]
                        print(f"移除约束 {removed_constraint}，新工作集 W = {W}")
                        history.append({'x': x.copy(), 'p': p, 'status': 'remove_constraint'})
                else:
                    history.append({'x': x.copy(), 'p': p, 'status': 'optimal'})
                    break
            except:
                print("无法求解乘子，继续")
                history.append({'x': x.copy(), 'p': p, 'status': 'error'})
                break
        
        # 步骤10-16: 如果p≠0，计算步长
        else:
            # 计算最大可行步长
            alpha = 1.0
            blocking_constraint = None
            
            for i in range(m):
                if i not in W:
                    a_i = A[i]
                    denom = a_i.dot(p)
                    if denom < -1e-10:  # 只考虑可能违反的约束
                        residual = b[i] - a_i.dot(x)
                        ratio = residual / denom
                        if ratio > 0 and ratio < alpha:
                            alpha = ratio
                            blocking_constraint = i
            
            print(f"步长 α = {alpha:.6f}, 阻塞约束 = {blocking_constraint}")
            
            # 更新x
            x_old = x.copy()
            x = x + alpha * p
            
            # 更新工作集
            if alpha < 1.0 - 1e-8 and blocking_constraint is not None:
                W.append(blocking_constraint)
                print(f"添加约束 {blocking_constraint}，新工作集 W = {W}")
            
            history.append({'x': x.copy(), 'p': p, 'alpha': alpha, 
                          'blocking': blocking_constraint, 'status': 'step'})
            
            # 检查收敛
            if np.linalg.norm(x - x_old) < 1e-8:
                print("收敛")
                break
    
    return x, history

# 第20题的问题数据
G = np.array([[2, -2], [-2, 4]])  # 二次项矩阵
c = np.array([-2, -6])            # 线性项

# 约束: 0.5*x1 + 0.5*x2 <= 1, -x1 + 2*x2 <= 2, x1 >= 0, x2 >= 0
A = np.array([[0.5, 0.5],   # 约束1: 0.5x1 + 0.5x2 <= 1
              [-1, 2],      # 约束2: -x1 + 2x2 <= 2
              [-1, 0],      # 约束3: x1 >= 0
              [0, -1]])     # 约束4: x2 >= 0
b = np.array([1, 2, 0, 0])

print("=== 第20题: 凸二次规划有效集方法 ===\n")
print("问题: min x1^2 + 2x2^2 - 2x1 - 6x2 - 2x1x2")
print("约束: 0.5x1 + 0.5x2 <= 1, -x1 + 2x2 <= 2, x1 >= 0, x2 >= 0")
print(f"G = {G}")
print(f"c = {c}")

# 三个不同的起始点
starting_points = [
    {"name": "内点", "x0": np.array([0.5, 0.5]), "W0": []},
    {"name": "顶点", "x0": np.array([0.0, 0.0]), "W0": [2, 3]},  # x1=0, x2=0 约束活跃
    {"name": "非顶点边界点", "x0": np.array([1.0, 0.0]), "W0": [3]}  # x2=0 约束活跃
]

results = {}

for start in starting_points:
    print(f"\n{'='*50}")
    print(f"起始点: {start['name']}")
    print(f"初始点: x0 = {start['x0']}")
    print(f"初始工作集: W0 = {start['W0']}")
    print(f"{'='*50}")
    
    x_opt, history = active_set_qp(G, c, A, b, start['x0'], start['W0'])
    
    # 计算最优值
    f_opt = 0.5 * x_opt.dot(G.dot(x_opt)) + c.dot(x_opt)
    
    print(f"\n最优解: x* = [{x_opt[0]:.6f}, {x_opt[1]:.6f}]")
    print(f"最优值: f(x*) = {f_opt:.6f}")
    
    # 检查约束满足情况
    constraints = A.dot(x_opt) - b
    print("约束满足情况:")
    for i in range(len(b)):
        status = "活跃" if abs(constraints[i]) < 1e-8 else "满足" if constraints[i] <= 1e-8 else "违反"
        print(f"  约束 {i+1}: {A[i]}·x <= {b[i]}, 残差: {constraints[i]:.6f} ({status})")
    
    results[start['name']] = {
        'x_opt': x_opt,
        'f_opt': f_opt,
        'history': history
    }

print(f"\n{'='*60}")
print("总结:")
for name, result in results.items():
    print(f"{name}:")
    print(f"  最优解: x* = [{result['x_opt'][0]:.6f}, {result['x_opt'][1]:.6f}]")
    print(f"  最优值: f(x*) = {result['f_opt']:.6f}")
    print(f"  迭代次数: {len(result['history'])}")

# 验证所有结果是否一致
all_same = True
ref_x = results['内点']['x_opt']
ref_f = results['内点']['f_opt']
for name, result in results.items():
    if np.linalg.norm(result['x_opt'] - ref_x) > 1e-6 or abs(result['f_opt'] - ref_f) > 1e-6:
        all_same = False
        break

print(f"\n所有起始点结果一致: {all_same}")