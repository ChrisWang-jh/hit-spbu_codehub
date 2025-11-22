import numpy as np

def rosenbrock(x):
    """Rosenbrock function: f(x) = 100(x2 - x1^2)^2 + (1 - x1)^2"""
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def gradient(x):
    """Gradient of Rosenbrock function"""
    g1 = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    g2 = 200 * (x[1] - x[0]**2)
    return np.array([g1, g2])

def hessian(x):
    """Hessian of Rosenbrock function (exact)"""
    h11 = 1200 * x[0]**2 - 400 * x[1] + 2
    h12 = -400 * x[0]
    h21 = -400 * x[0]
    h22 = 200
    return np.array([[h11, h12], [h21, h22]])

def compute_cauchy_point(g, B, delta):
    """
    Compute the Cauchy point (steepest descent step within trust region)
    p_c = -tau * (Delta / ||g||) * g
    where tau minimizes the quadratic model along the steepest descent direction
    """
    g_norm = np.linalg.norm(g)
    if g_norm < 1e-10:
        return np.zeros_like(g)
    
    # Compute tau
    Bg = B @ g
    gBg = g @ Bg
    
    if gBg <= 0:
        tau = 1.0
    else:
        tau = min(1.0, (g_norm**3) / (delta * gBg))
    
    p_c = -tau * (delta / g_norm) * g
    return p_c

def compute_newton_step(g, B):
    """
    Compute the Newton step: p_B = -B^{-1} * g
    """
    try:
        p_B = np.linalg.solve(B, -g)
        return p_B, True
    except np.linalg.LinAlgError:
        return np.zeros_like(g), False

def dogleg_step(g, B, delta):
    """
    Compute the dogleg step
    
    The dogleg path consists of:
    1. From origin to Cauchy point: p(tau) = tau * p_c, tau in [0, 1]
    2. From Cauchy point to Newton point: p(tau) = p_c + (tau - 1)(p_B - p_c), tau in [1, 2]
    
    Parameters:
    - g: gradient
    - B: Hessian (or approximation)
    - delta: trust region radius
    
    Returns:
    - p: dogleg step
    """
    # Compute Cauchy point
    p_c = compute_cauchy_point(g, B, delta)
    
    # Compute Newton step
    p_B, success = compute_newton_step(g, B)
    
    if not success:
        # If Newton step fails, use Cauchy point
        return p_c
    
    # If Newton step is within trust region, use it
    if np.linalg.norm(p_B) <= delta:
        return p_B
    
    # If Cauchy point is outside trust region, scale it
    if np.linalg.norm(p_c) >= delta:
        return (delta / np.linalg.norm(p_c)) * p_c
    
    # Otherwise, find the dogleg point on the line from p_c to p_B
    # Solve: ||p_c + tau(p_B - p_c)||^2 = delta^2 for tau in [0, 1]
    p_diff = p_B - p_c
    a = np.dot(p_diff, p_diff)
    b = 2 * np.dot(p_c, p_diff)
    c = np.dot(p_c, p_c) - delta**2
    
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        # Should not happen, but return Cauchy point as fallback
        return p_c
    
    tau = (-b + np.sqrt(discriminant)) / (2*a)
    tau = max(0, min(1, tau))  # Ensure tau in [0, 1]
    
    p = p_c + tau * p_diff
    return p

def compute_rho(x, p, g, B):
    """
    Compute the ratio rho = (f(x) - f(x+p)) / (m(0) - m(p))
    where m is the quadratic model: m(p) = f(x) + g^T*p + 0.5*p^T*B*p
    """
    f_x = rosenbrock(x)
    f_x_new = rosenbrock(x + p)
    
    actual_reduction = f_x - f_x_new
    predicted_reduction = -(g @ p + 0.5 * p @ (B @ p))
    
    if abs(predicted_reduction) < 1e-14:
        return 0
    
    rho = actual_reduction / predicted_reduction
    return rho

def dogleg_trust_region(x0, delta_max=2.0, delta_0=1.0, eta=0.125, 
                        max_iter=1000, tol=1e-6, verbose=True):
    """
    Dogleg method with trust region
    
    Algorithm (Trust Region):
    (a) Given Delta_max > 0, Delta_0 in (0, Delta_max), and eta in [0, 1/4)
    (b) For k = 0, 1, 2, ...:
        i. Obtain p_k by (approximately) solving the subproblem
        ii. Evaluate rho_k
        iii. If rho_k < 1/4, set Delta_{k+1} = 1/4 * Delta_k
        iv. Else if rho_k > 3/4 and ||p_k|| = Delta_k, set Delta_{k+1} = min(2*Delta_k, Delta_max)
        v. Else set Delta_{k+1} = Delta_k
        vi. If rho_k > eta, set x_{k+1} = x_k + p_k, else x_{k+1} = x_k
    
    Parameters:
    - x0: initial point
    - delta_max: maximum trust region radius
    - delta_0: initial trust region radius
    - eta: acceptance threshold
    - max_iter: maximum iterations
    - tol: tolerance for convergence
    """
    x = x0.copy()
    delta = delta_0
    
    if verbose:
        print("\n" + "="*85)
        print("DOGLEG METHOD WITH TRUST REGION")
        print(f"Initial point: x0 = {x0}")
        print(f"Trust region parameters: Δ_max={delta_max}, Δ_0={delta_0}, η={eta}")
        print("="*85)
        print(f"{'Iter':<6} {'x1':<12} {'x2':<12} {'f(x)':<14} {'||∇f||':<12} {'||p||':<10} {'ρ':<10} {'Δ':<10}")
        print("-"*85)
    
    for k in range(max_iter):
        # Compute gradient and Hessian
        g = gradient(x)
        B = hessian(x)  # Using exact Hessian as B_k
        g_norm = np.linalg.norm(g)
        f_val = rosenbrock(x)
        
        if k == 0 and verbose:
            print(f"{k:<6} {x[0]:<12.6f} {x[1]:<12.6f} {f_val:<14.6e} {g_norm:<12.6e} {'-':<10} {'-':<10} {delta:<10.4f}")
        
        # Check convergence
        if g_norm < tol:
            if verbose:
                print(f"\nConverged at iteration {k}: ||∇f|| = {g_norm:.6e} < {tol}")
            break
        
        # Solve trust region subproblem using dogleg method
        p = dogleg_step(g, B, delta)
        p_norm = np.linalg.norm(p)
        
        # Compute rho
        rho = compute_rho(x, p, g, B)
        
        # Update trust region radius (steps iii, iv, v)
        if rho < 0.25:
            delta = 0.25 * delta
        elif rho > 0.75 and abs(p_norm - delta) < 1e-6:
            delta = min(2 * delta, delta_max)
        # else: delta remains the same
        
        # Update x (step vi)
        if rho > eta:
            x = x + p
            accepted = "✓"
        else:
            accepted = "✗"
        
        if verbose:
            print(f"{k+1:<6} {x[0]:<12.6f} {x[1]:<12.6f} {rosenbrock(x):<14.6e} {g_norm:<12.6e} {p_norm:<10.6f} {rho:<10.4f} {delta:<10.4f} {accepted}")
        
        # Safety check
        if delta < 1e-12:
            if verbose:
                print(f"\nTrust region radius too small: Δ = {delta:.6e}")
            break
    
    if verbose:
        print("-"*85)
        print(f"Final point: x* = [{x[0]:.8f}, {x[1]:.8f}]")
        print(f"Final value: f(x*) = {rosenbrock(x):.10e}")
        print(f"Final gradient norm: ||∇f|| = {np.linalg.norm(gradient(x)):.6e}")
        print(f"Total iterations: {k+1}")
        print("="*85)
    
    return x

def experiment_with_parameters():
    """
    Experiment with different trust region update rules
    """
    print("\n" + "#"*85)
    print("EXPERIMENTS WITH DIFFERENT TRUST REGION PARAMETERS")
    print("#"*85)
    
    # Test different starting points
    test_cases = [
        {"x0": np.array([1.2, 1.2]), "name": "x0 = [1.2, 1.2]"},
        {"x0": np.array([-1.2, 1.0]), "name": "x0 = [-1.2, 1.0] (difficult)"},
        {"x0": np.array([0.0, 0.0]), "name": "x0 = [0.0, 0.0]"},
    ]
    
    # Different parameter sets
    param_sets = [
        {"delta_max": 2.0, "delta_0": 1.0, "eta": 0.125, "name": "Standard"},
        {"delta_max": 2.0, "delta_0": 0.5, "eta": 0.125, "name": "Smaller initial Δ"},
        {"delta_max": 5.0, "delta_0": 2.0, "eta": 0.125, "name": "Larger Δ"},
        {"delta_max": 2.0, "delta_0": 1.0, "eta": 0.05, "name": "Stricter acceptance (η=0.05)"},
        {"delta_max": 2.0, "delta_0": 1.0, "eta": 0.2, "name": "Looser acceptance (η=0.2)"},
    ]
    
    for tc in test_cases:
        print(f"\n{'='*85}")
        print(f"Starting point: {tc['name']}")
        print(f"{'='*85}")
        
        for params in param_sets:
            print(f"\n{'-'*85}")
            print(f"Parameter set: {params['name']}")
            print(f"  Δ_max = {params['delta_max']}, Δ_0 = {params['delta_0']}, η = {params['eta']}")
            print(f"{'-'*85}")
            
            x_final = dogleg_trust_region(
                tc['x0'], 
                delta_max=params['delta_max'],
                delta_0=params['delta_0'],
                eta=params['eta'],
                max_iter=100,
                verbose=False
            )
            
            print(f"Result: x* = [{x_final[0]:.6f}, {x_final[1]:.6f}], f(x*) = {rosenbrock(x_final):.6e}")

if __name__ == "__main__":
    # Standard test with default parameters
    print("\n" + "#"*85)
    print("STANDARD TEST WITH DEFAULT PARAMETERS")
    print("#"*85)
    
    # Test case 1: x0 = [1.2, 1.2]
    print("\n" + "="*85)
    print("TEST CASE 1: x0 = [1.2, 1.2]")
    print("="*85)
    x0_1 = np.array([1.2, 1.2])
    x_final_1 = dogleg_trust_region(x0_1, delta_max=2.0, delta_0=1.0, eta=0.125)
    
    # Test case 2: x0 = [-1.2, 1.0]
    print("\n" + "="*85)
    print("TEST CASE 2: x0 = [-1.2, 1.0] (more difficult)")
    print("="*85)
    x0_2 = np.array([-1.2, 1.0])
    x_final_2 = dogleg_trust_region(x0_2, delta_max=2.0, delta_0=1.0, eta=0.125)
    
    # Run experiments with different parameters
    experiment_with_parameters()
    
    print("\n" + "#"*85)
    print("SUMMARY")
    print("#"*85)
    print("\nDogleg Method Key Features:")
    print("- Combines Cauchy point (steepest descent) and Newton step")
    print("- Automatically adjusts trust region radius based on model quality (ρ)")
    print("- More robust than pure Newton method")
    print("- Efficient for well-conditioned problems")
    print("\nTrust Region Update Rules:")
    print("- If ρ < 1/4: reduce trust region (Δ_{k+1} = 1/4 * Δ_k)")
    print("- If ρ > 3/4 and ||p|| = Δ: expand trust region (Δ_{k+1} = min(2Δ_k, Δ_max))")
    print("- Otherwise: keep trust region unchanged")
    print("- Accept step only if ρ > η")