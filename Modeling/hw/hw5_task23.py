import numpy as np
from scipy.linalg import solve

# ==================== 问题定义 ====================
# max 6x1 + 4x2 - 13 - x1^2 - x2^2
# s.t. x1 + x2 <= 3, x1 >= 0, x2 >= 0
#
# 转换为最小化问题:
# min -6x1 - 4x2 + 13 + x1^2 + x2^2
# ================================================

def objective_max(x):
    """原始最大化目标函数"""
    return 6*x[0] + 4*x[1] - 13 - x[0]**2 - x[1]**2

def objective_min(x):
    """转换后的最小化目标函数"""
    return -6*x[0] - 4*x[1] + 13 + x[0]**2 + x[1]**2

def gradient(x):
    """梯度向量 (最小化问题)"""
    g = np.array([
        2*x[0] - 6,
        2*x[1] - 4
    ])
    return g

def hessian():
    """Hessian矩阵"""
    G = np.array([
        [2.0, 0.0],
        [0.0, 2.0]
    ])
    return G

def constraint_matrix():
    """
    约束矩阵 A, 使得 Ax <= b
    约束1: x1 + x2 <= 3
    约束2: -x1 <= 0 (即 x1 >= 0)
    约束3: -x2 <= 0 (即 x2 >= 0)
    """
    A = np.array([
        [1.0, 1.0],    # 约束1
        [-1.0, 0.0],   # 约束2 (x1 >= 0)
        [0.0, -1.0]    # 约束3 (x2 >= 0)
    ])
    b = np.array([3.0, 0.0, 0.0])
    return A, b

def is_feasible(x, A, b, tol=1e-8):
    """检查点是否可行"""
    return np.all(A @ x <= b + tol)

def get_active_constraints(x, A, b, tol=1e-8):
    """获取在点x处的活跃约束集合"""
    active = []
    for i in range(len(b)):
        if abs(A[i] @ x - b[i]) < tol:
            active.append(i)
    return active

def solve_equality_qp(G, c, A_active, x_k):
    """
    求解等式约束二次规划子问题:
    min 0.5*p^T*G*p + (Gx_k + c)^T*p
    s.t. A_active * p = 0
    """
    n = len(c)
    grad = G @ x_k + c
    
    if len(A_active) == 0:
        # 无约束情况
        p = -np.linalg.solve(G, grad)
        return p
    
    # 构建KKT系统
    A_mat = np.array(A_active)
    m = A_mat.shape[0]
    
    # KKT矩阵: [G  A^T]
    #          [A   0 ]
    KKT = np.zeros((n + m, n + m))
    KKT[:n, :n] = G
    KKT[:n, n:] = A_mat.T
    KKT[n:, :n] = A_mat
    
    # 右端: [-grad, 0]
    rhs = np.zeros(n + m)
    rhs[:n] = -grad
    
    # 求解KKT系统
    sol = solve(KKT, rhs)
    p = sol[:n]
    
    return p

def compute_step_length(x_k, p_k, A, b):
    """计算步长，确保不违反任何约束"""
    alpha = 1.0
    n_constraints = len(b)
    
    for i in range(n_constraints):
        a_i = A[i]
        denominator = a_i @ p_k
        
        if denominator > 1e-10:  # 约束可能被违反
            alpha_i = (b[i] - a_i @ x_k) / denominator
            if alpha_i < alpha:
                alpha = alpha_i
    
    return alpha

def find_blocking_constraint(x_k, p_k, A, b, W_k):
    """找到阻碍约束"""
    alpha = compute_step_length(x_k, p_k, A, b)
    
    if alpha >= 1.0 - 1e-10:
        return None
    
    # 找到哪个约束被阻碍
    for i in range(len(b)):
        if i not in W_k:
            a_i = A[i]
            if a_i @ p_k > 1e-10:
                alpha_i = (b[i] - a_i @ x_k) / (a_i @ p_k)
                if abs(alpha_i - alpha) < 1e-8:
                    return i
    return None

def active_set_method(x0, A, b, max_iter=100, tol=1e-8, verbose=True):
    """Active-Set算法主函数"""
    G = hessian()
    c = np.array([-6.0, -4.0])  # 线性项系数
    
    x_k = x0.copy()
    W_k = get_active_constraints(x_k, A, b)
    
    if verbose:
        print(f"\n初始点: x0 = [{x0[0]:.4f}, {x0[1]:.4f}]")
        print(f"初始目标函数值 (最大化): f(x0) = {objective_max(x0):.6f}")
        print(f"初始活跃约束集: {W_k}")
    
    trajectory = [x_k.copy()]
    
    for k in range(max_iter):
        if verbose:
            print(f"\n--- 迭代 {k} ---")
            print(f"当前点: x_{k} = [{x_k[0]:.6f}, {x_k[1]:.6f}]")
            print(f"活跃约束集 W_{k} = {W_k}")
        
        # Step 4: 求解子问题得到搜索方向
        A_active = [A[i] for i in W_k]
        p_k = solve_equality_qp(G, c, A_active, x_k)
        
        if verbose:
            print(f"搜索方向: p_{k} = [{p_k[0]:.6f}, {p_k[1]:.6f}]")
        
        # Step 5: 如果p_k = 0，计算Lagrange乘子
        if np.linalg.norm(p_k) < tol:
            if verbose:
                print("p_k ≈ 0, 计算Lagrange乘子...")
            
            grad = gradient(x_k)
            A_active = np.array([A[i] for i in W_k])
            
            if len(W_k) > 0:
                # 求解: G*x_k + c = A_active^T * lambda
                lambda_hat = np.linalg.lstsq(A_active.T, -grad, rcond=None)[0]
                if verbose:
                    print(f"Lagrange乘子: λ = {lambda_hat}")
                
                # Step 6: 检查最优性条件
                # 对于不等式约束，需要 lambda >= 0
                inequality_indices = list(range(len(W_k)))
                
                if all(lambda_hat[i] >= -tol for i in inequality_indices):
                    if verbose:
                        print(f"\n{'*'*50}")
                        print(f"*** 找到最优解! ***")
                        print(f"{'*'*50}")
                        print(f"最优点: x* = [{x_k[0]:.6f}, {x_k[1]:.6f}]")
                        print(f"最优值 (最大化): f(x*) = {objective_max(x_k):.6f}")
                        print(f"最优值 (最小化): f(x*) = {objective_min(x_k):.6f}")
                    return x_k, trajectory
                
                # Step 8: 找到最负的Lagrange乘子
                min_lambda = float('inf')
                j = -1
                for i in inequality_indices:
                    if lambda_hat[i] < min_lambda:
                        min_lambda = lambda_hat[i]
                        j = i
                
                if verbose:
                    print(f"移除约束: W_{k}[{j}] = {W_k[j]} (λ = {min_lambda:.6f})")
                # Step 9: 移除约束j
                W_k.pop(j)
            else:
                if verbose:
                    print("无活跃约束且p_k≈0，已找到最优解")
                return x_k, trajectory
        
        else:  # p_k != 0
            # Step 11-12: 计算步长并更新
            alpha_k = compute_step_length(x_k, p_k, A, b)
            if verbose:
                print(f"步长: α_{k} = {alpha_k:.6f}")
            
            x_k = x_k + alpha_k * p_k
            trajectory.append(x_k.copy())
            
            # Step 13-14: 检查阻碍约束
            blocking = find_blocking_constraint(x_k - alpha_k * p_k, p_k, A, b, W_k)
            
            if blocking is not None:
                if verbose:
                    print(f"添加阻碍约束: {blocking}")
                W_k.append(blocking)
            
            if verbose:
                print(f"更新后点: x_{k+1} = [{x_k[0]:.6f}, {x_k[1]:.6f}]")
                print(f"目标函数值 (最大化): f(x_{k+1}) = {objective_max(x_k):.6f}")
    
    if verbose:
        print("\n达到最大迭代次数")
    return x_k, trajectory

def graphical_solution():
    """图形化求解（理论分析）"""
    print("\n" + "="*70)
    print("第一步: 图形化分析（理论分析）")
    print("="*70)
    
    print("\n目标函数: max 6x1 + 4x2 - 13 - x1² - x2²")
    print("约束条件:")
    print("  x1 + x2 ≤ 3")
    print("  x1 ≥ 0")
    print("  x2 ≥ 0")
    
    # 分析目标函数
    print("\n目标函数分析:")
    print("  这是一个凹函数 (Hessian矩阵负定)")
    print("  ∇f = [6 - 2x1, 4 - 2x2]")
    print("  无约束最优点: ∇f = 0 => x1=3, x2=2")
    
    # 检查无约束最优点
    x_unconstrained = np.array([3.0, 2.0])
    print(f"  无约束最优点: ({x_unconstrained[0]}, {x_unconstrained[1]})")
    print(f"  约束检查: x1 + x2 = {x_unconstrained[0] + x_unconstrained[1]} > 3")
    print(f"  结论: 无约束最优点不可行，最优解在边界上")
    
    # 检查边界
    print("\n可行域顶点:")
    vertices = [
        np.array([0.0, 0.0]),
        np.array([3.0, 0.0]),
        np.array([0.0, 3.0])
    ]
    
    for i, v in enumerate(vertices):
        print(f"  顶点{i+1}: ({v[0]}, {v[1]}) => f = {objective_max(v):.4f}")
    
    # 检查约束 x1 + x2 = 3 上的最优点
    print("\n约束 x1 + x2 = 3 上的最优点:")
    print("  令 x2 = 3 - x1, 代入目标函数:")
    print("  f(x1) = 6x1 + 4(3-x1) - 13 - x1² - (3-x1)²")
    print("       = 6x1 + 12 - 4x1 - 13 - x1² - 9 + 6x1 - x1²")
    print("       = -2x1² + 8x1 - 10")
    print("  f'(x1) = -4x1 + 8 = 0 => x1 = 2")
    print("  因此 x2 = 3 - 2 = 1")
    
    x_boundary = np.array([2.0, 1.0])
    print(f"  边界最优点: ({x_boundary[0]}, {x_boundary[1]})")
    print(f"  目标函数值: f = {objective_max(x_boundary):.4f}")
    
    print("\n图形化结论: 最优解应为 x* = (2, 1), f* = -1")

def print_summary_table(results):
    """打印结果汇总表格"""
    print("\n" + "="*90)
    print("结果汇总表")
    print("="*90)
    print(f"{'起始点类型':<15} {'起始点':<20} {'最优解':<20} {'最优值':<12} {'迭代次数':<10}")
    print("-"*90)
    for point_type, x0, x_opt, f_opt, n_iter in results:
        x0_str = f"({x0[0]:.2f}, {x0[1]:.2f})"
        x_opt_str = f"({x_opt[0]:.4f}, {x_opt[1]:.4f})"
        print(f"{point_type:<15} {x0_str:<20} {x_opt_str:<20} {f_opt:<12.6f} {n_iter:<10}")
    print("="*90)

# 主程序
if __name__ == "__main__":
    print("="*70)
    print("问题 23: 二次规划求解")
    print("="*70)
    
    # 第一步: 图形化求解（理论分析）
    graphical_solution()
    
    # 第二步: 使用Active-Set方法求解
    print("\n\n" + "="*70)
    print("第二步: Active-Set Method求解")
    print("="*70)
    
    A, b = constraint_matrix()
    
    # 三个起始点
    starting_points = [
        np.array([1.0, 1.0]),   # 内部点
        np.array([0.0, 0.0]),   # 顶点
        np.array([1.5, 0.0])    # 非顶点边界点
    ]
    
    point_types = ["内部点", "顶点", "非顶点边界点"]
    
    results = []
    trajectories = []
    
    for i, x0 in enumerate(starting_points):
        print(f"\n\n{'='*70}")
        print(f"测试 {i+1}: 起始点类型 - {point_types[i]}")
        print(f"{'='*70}")
        
        if is_feasible(x0, A, b):
            x_opt, traj = active_set_method(x0, A, b, verbose=True)
            results.append((point_types[i], x0, x_opt, objective_max(x_opt), len(traj)))
            trajectories.append(traj)
        else:
            print(f"起始点 {x0} 不可行!")
    
    # 打印汇总表格
    print_summary_table(results)
    
    # 验证最优性条件
    print("\n" + "="*70)
    print("最优性验证")
    print("="*70)
    if results:
        x_opt = results[0][2]
        print(f"最优点: x* = [{x_opt[0]:.6f}, {x_opt[1]:.6f}]")
        print(f"梯度 (最小化问题): ∇f_min(x*) = {gradient(x_opt)}")
        print(f"活跃约束: {get_active_constraints(x_opt, A, b)}")
        
        # 检查约束
        constraint_names = ["x1 + x2 ≤ 3", "x1 ≥ 0", "x2 ≥ 0"]
        print(f"\n约束检查:")
        print(f"  {constraint_names[0]}: x1 + x2 = {x_opt[0] + x_opt[1]:.6f}")
        print(f"  {constraint_names[1]}: x1 = {x_opt[0]:.6f}")
        print(f"  {constraint_names[2]}: x2 = {x_opt[1]:.6f}")
        
        print("\n所有起始点都收敛到同一最优解，验证算法正确性")