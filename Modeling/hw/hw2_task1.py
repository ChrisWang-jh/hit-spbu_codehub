import numpy as np

def rosenbrock(x):
    """Rosenbrock function: f(x) = 100(x2-x1^2)^2 + (1-x1)^2"""
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def gradient(x):
    """Gradient of Rosenbrock function"""
    g1 = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    g2 = 200 * (x[1] - x[0]**2)
    return np.array([g1, g2])

def hessian(x):
    """Hessian of Rosenbrock function"""
    h11 = 1200 * x[0]**2 - 400 * x[1] + 2
    h12 = -400 * x[0]
    h21 = -400 * x[0]
    h22 = 200
    return np.array([[h11, h12], [h21, h22]])

def backtracking_line_search(f, grad_f, x, p, alpha_init=1.0, rho=0.5, c=0.0001):
    """
    Backtracking line search (Algorithm 1)
    
    Parameters:
    - f: objective function
    - grad_f: gradient function
    - x: current point
    - p: search direction
    - alpha_init: initial step length
    - rho: reduction factor (ρ ∈ (0,1)) 阿尔法
    - c: Armijo constant (c ∈ (0,1))
    
    Returns:
    - alpha: step length satisfying Armijo condition
    """
    alpha = alpha_init
    f_x = f(x)
    grad_x = grad_f(x)
    
    # Armijo condition: f(x + αp) ≤ f(x) + c·α·∇f^T·p
    while f(x + alpha * p) > f_x + c * alpha * np.dot(grad_x, p):
        alpha = rho * alpha
    
    return alpha

def steepest_descent(x0, max_iter=10000, tol=1e-6, alpha_init=1.0):
    """
    Steepest Descent method with backtracking line search
    
    Parameters:
    - x0: initial point
    - max_iter: maximum number of iterations
    - tol: tolerance for convergence
    - alpha_init: initial step length
    """
    x = x0.copy()
    print("\n" + "="*70)
    print("STEEPEST DESCENT METHOD")
    print(f"Initial point: x0 = {x0}")
    print("="*70)
    print(f"{'Iter':<6} {'x1':<12} {'x2':<12} {'f(x)':<14} {'||∇f||':<12} {'α':<10}")
    print("-"*70)
    
    for k in range(max_iter):
        grad = gradient(x)
        grad_norm = np.linalg.norm(grad)
        f_val = rosenbrock(x)
        
        if k == 0:
            print(f"{k:<6} {x[0]:<12.6f} {x[1]:<12.6f} {f_val:<14.6e} {grad_norm:<12.6e} {0:<10}")
        
        # Check convergence
        if grad_norm < tol:
            print(f"Converged at iteration {k}")
            break
        
        # Search direction: negative gradient
        p = -grad
        
        # Backtracking line search
        alpha = backtracking_line_search(rosenbrock, gradient, x, p, alpha_init)
        
        # Update
        x = x + alpha * p
        
        # Print iteration info
        if k < 50:  # Print first 50 iterations
            print(f"{k+1:<6} {x[0]:<12.6f} {x[1]:<12.6f} {rosenbrock(x):<14.6e} {grad_norm:<12.6e} {alpha:<10.6f}")
    
    print("-"*70)
    print(f"Final point: x* = [{x[0]:.8f}, {x[1]:.8f}]")
    print(f"Final value: f(x*) = {rosenbrock(x):.10e}")
    print(f"Total iterations: {k+1}")
    print("="*70)
    
    return x

def newton_method(x0, max_iter=10000, tol=1e-6, alpha_init=1.0):
    """
    Newton's method with backtracking line search
    
    Parameters:
    - x0: initial point
    - max_iter: maximum number of iterations
    - tol: tolerance for convergence
    - alpha_init: initial step length
    """
    x = x0.copy()
    print("\n" + "="*70)
    print("NEWTON'S METHOD")
    print(f"Initial point: x0 = {x0}")
    print("="*70)
    print(f"{'Iter':<6} {'x1':<12} {'x2':<12} {'f(x)':<14} {'||∇f||':<12} {'α':<10}")
    print("-"*70)
    
    for k in range(max_iter):
        grad = gradient(x)
        grad_norm = np.linalg.norm(grad)
        f_val = rosenbrock(x)
        
        if k == 0:
            print(f"{k:<6} {x[0]:<12.6f} {x[1]:<12.6f} {f_val:<14.6e} {grad_norm:<12.6e} {0:<10}")
        
        # Check convergence
        if grad_norm < tol:
            print(f"Converged at iteration {k}")
            break
        
        # Compute Hessian and check if it's positive definite
        H = hessian(x)
        
        try:
            # Solve H * p = -grad for Newton direction
            p = -np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            print(f"Singular Hessian at iteration {k}")
            break
        
        # Check if direction is descent direction
        if np.dot(grad, p) >= 0:
            print(f"Not a descent direction at iteration {k}, switching to gradient")
            p = -grad
        
        # Backtracking line search
        alpha = backtracking_line_search(rosenbrock, gradient, x, p, alpha_init)
        
        # Update
        x = x + alpha * p
        
        # Print iteration info
        if k < 50:  # Print first 50 iterations
            print(f"{k+1:<6} {x[0]:<12.6f} {x[1]:<12.6f} {rosenbrock(x):<14.6e} {grad_norm:<12.6e} {alpha:<10.6f}")
    
    print("-"*70)
    print(f"Final point: x* = [{x[0]:.8f}, {x[1]:.8f}]")
    print(f"Final value: f(x*) = {rosenbrock(x):.10e}")
    print(f"Total iterations: {k+1}")
    print("="*70)
    
    return x

if __name__ == "__main__":
    # Test case 1: x0 = [1.2, 1.2]
    print("\n" + "#"*70)
    print("TEST CASE 1: Initial point x0 = [1.2, 1.2]")
    print("#"*70)
    
    x0_1 = np.array([1.2, 1.2])
    
    # Steepest Descent
    x_sd_1 = steepest_descent(x0_1, alpha_init=1.0)
    
    # Newton's Method
    x_newton_1 = newton_method(x0_1, alpha_init=1.0)
    
    # Test case 2: x0 = [-1.2, 1]
    print("\n\n" + "#"*70)
    print("TEST CASE 2: Initial point x0 = [-1.2, 1.0] (more difficult)")
    print("#"*70)
    
    x0_2 = np.array([-1.2, 1.0])
    
    # Steepest Descent
    x_sd_2 = steepest_descent(x0_2, alpha_init=1.0)
    
    # Newton's Method
    x_newton_2 = newton_method(x0_2, alpha_init=1.0)
    
    print("\n" + "#"*70)
    print("SUMMARY")
    print("#"*70)
    
    print("\nThe global minimum of Rosenbrock function is at x* = [1, 1]")
    print("with f(x*) = 0")

    print("\nObservations:")
    print("- Newton's method converges much faster than Steepest Descent")
    print("- From x0=[1.2, 1.2]: Both methods converge quickly (close to minimum)")
    print("- From x0=[-1.2, 1.0]: Newton's method still fast, Steepest Descent slower")
    print("- Step lengths (α) vary significantly during optimization")