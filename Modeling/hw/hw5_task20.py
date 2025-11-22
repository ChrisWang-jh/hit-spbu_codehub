import numpy as np
from scipy.linalg import solve

def objective(x):
    """目标函数: x1^2 + 2x2^2 - 2x1 - 6x2 - 2x1*x2"""
    return x[0]**2 + 2*x[1]**2 - 2*x[0] - 6*x[1] - 2*x[0]*x[1]

def gradient(x):
    """梯度向量"""
    g = np.array([
        2*x[0] - 2 - 2*x[1],
        4*x[1] - 6 - 2*x[0]
    ])
    return g

def hessian():
    """Hessian矩阵 (常数)"""
    G = np.array([
        [2, -2],
        [-2, 4]
    ])
    return G

def constraint_matrix():
    """约束矩阵 A, 使得 Ax <= b"""
    # 约束1: 0.5*x1 + 0.5*x2 <= 1
    # 约束2: -x1 + 2*x2 <= 2
    # 约束3: -x1 <= 0 (即 x1 >= 0)
    # 约束4: -x2 <= 0 (即 x2 >= 0)
    A = np.array([
        [0.5, 0.5],   # 约束1
        [-1, 2],      # 约束2
        [-1, 0],      # 约束3 (x1 >= 0)
        [0, -1]       # 约束4 (x2 >= 0)
    ])
    b = np.array([1, 2, 0, 0])
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

def active_set_method(x0, A, b, max_iter=100, tol=1e-8):
    """Active-Set算法主函数"""
    G = hessian()
    c = np.array([-2, -6])  # 线性项系数
    
    x_k = x0.copy()
    W_k = get_active_constraints(x_k, A, b)
    
    print(f"\n初始点: x0 = {x0}")
    print(f"初始目标函数值: f(x0) = {objective(x0):.6f}")
    print(f"初始活跃约束集: {W_k}")
    
    trajectory = [x_k.copy()]
    
    for k in range(max_iter):
        print(f"\n--- 迭代 {k} ---")
        print(f"当前点: x_{k} = {x_k}")
        print(f"活跃约束集 W_{k} = {W_k}")
        
        # Step 4: 求解子问题得到搜索方向
        A_active = [A[i] for i in W_k]
        p_k = solve_equality_qp(G, c, A_active, x_k)
        
        print(f"搜索方向: p_{k} = {p_k}")
        
        # Step 5: 如果p_k = 0，计算Lagrange乘子
        if np.linalg.norm(p_k) < tol:
            print("p_k = 0, 计算Lagrange乘子...")
            
            grad = gradient(x_k)
            A_active = np.array([A[i] for i in W_k])
            
            if len(W_k) > 0:
                # 求解: G*x_k + c = A_active^T * lambda
                lambda_hat = np.linalg.lstsq(A_active.T, -grad, rcond=None)[0]
                print(f"Lagrange乘子: λ = {lambda_hat}")
                
                # Step 6: 检查最优性条件
                # 对于不等式约束，需要 lambda >= 0
                inequality_indices = [i for i, idx in enumerate(W_k) if idx < 4]
                
                if all(lambda_hat[i] >= -tol for i in inequality_indices):
                    print(f"\n*** 找到最优解! ***")
                    print(f"最优点: x* = {x_k}")
                    print(f"最优值: f(x*) = {objective(x_k):.6f}")
                    return x_k, trajectory
                
                # Step 8: 找到最负的Lagrange乘子
                min_lambda = float('inf')
                j = -1
                for i in inequality_indices:
                    if lambda_hat[i] < min_lambda:
                        min_lambda = lambda_hat[i]
                        j = i
                
                print(f"移除约束: W_{k}[{j}] = {W_k[j]}")
                # Step 9: 移除约束j
                W_k.pop(j)
            else:
                print("无活跃约束且p_k=0，已找到最优解")
                return x_k, trajectory
        
        else:  # p_k != 0
            # Step 11-12: 计算步长并更新
            alpha_k = compute_step_length(x_k, p_k, A, b)
            print(f"步长: α_{k} = {alpha_k:.6f}")
            
            x_k = x_k + alpha_k * p_k
            trajectory.append(x_k.copy())
            
            # Step 13-14: 检查阻碍约束
            blocking = find_blocking_constraint(x_k - alpha_k * p_k, p_k, A, b, W_k)
            
            if blocking is not None:
                print(f"添加阻碍约束: {blocking}")
                W_k.append(blocking)
            
            print(f"更新后点: x_{k+1} = {x_k}")
            print(f"目标函数值: f(x_{k+1}) = {objective(x_k):.6f}")
    
    print("\n达到最大迭代次数")
    return x_k, trajectory


# 主程序
if __name__ == "__main__":
    A, b = constraint_matrix()
    
    # 三个起始点
    starting_points = [
        np.array([0.5, 0.5]),   # 内部点
        np.array([0.0, 0.0]),   # 顶点
        np.array([0.0, 0.5])    # 非顶点边界点
    ]
    
    point_types = ["内部点", "顶点", "非顶点边界点"]
    
    print("="*70)
    print("Active-Set Method for Convex QP")
    print("="*70)
    print("\n目标函数: min x1² + 2x2² - 2x1 - 6x2 - 2x1*x2")
    print("约束条件:")
    print("  0.5*x1 + 0.5*x2 <= 1")
    print("  -x1 + 2*x2 <= 2")
    print("  x1, x2 >= 0")
    print("="*70)
    
    results = []
    
    for i, x0 in enumerate(starting_points):
        print(f"\n\n{'='*70}")
        print(f"测试 {i+1}: 起始点类型 - {point_types[i]}")
        print(f"{'='*70}")
        
        if is_feasible(x0, A, b):
            x_opt, traj = active_set_method(x0, A, b)
            results.append((point_types[i], x0, x_opt, objective(x_opt), len(traj)))
        else:
            print(f"起始点 {x0} 不可行!")
    
    # 总结结果
    print("\n\n" + "="*70)
    print("结果总结")
    print("="*70)
    for point_type, x0, x_opt, f_opt, n_iter in results:
        print(f"\n{point_type}:")
        print(f"  起始点: {x0}")
        print(f"  最优解: x* = {x_opt}")
        print(f"  最优值: f(x*) = {f_opt:.6f}")
        print(f"  迭代次数: {n_iter}")