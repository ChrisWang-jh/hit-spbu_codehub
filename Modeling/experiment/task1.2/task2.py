import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

# ==================== Helper Functions ====================

def basis(i, n):
    """Return the i-th standard basis vector in n dimensions."""
    e = np.zeros(n)
    e[i] = 1.0
    return e

# Global counter for function evaluations
func_eval_counter = 0

def counted_function(f):
    """Wrapper to count function evaluations."""
    def wrapper(x):
        global func_eval_counter
        func_eval_counter += 1
        return f(x)
    return wrapper

def golden_section_search(f, x, d, tol=1e-6, max_iter=100):
    """
    One-dimensional line search using golden section method.
    Finds alpha that minimizes f(x + alpha * d).
    """
    phi = (1 + np.sqrt(5)) / 2
    resphi = 2 - phi
    
    # Initial bracket
    a, b = -10.0, 10.0
    
    for _ in range(max_iter):
        x1 = a + resphi * (b - a)
        x2 = b - resphi * (b - a)
        
        if f(x + x1 * d) < f(x + x2 * d):
            b = x2
        else:
            a = x1
            
        if abs(b - a) < tol:
            break
    
    alpha = (a + b) / 2
    return x + alpha * d

# ==================== Powell's Method ====================

def powell_method(f, x, epsilon=1e-6, max_iter=1000):
    """
    Powell's method for unconstrained optimization.
    
    Parameters:
    - epsilon: convergence tolerance
    """
    x = np.array(x, dtype=float)
    n = len(x)
    
    # Initialize direction set with standard basis vectors
    U = [basis(i, n) for i in range(n)]
    
    delta = np.inf
    iterations = 0
    trajectory = [x.copy()]
    f_history = [f(x)]
    
    while delta > epsilon and iterations < max_iter:
        x_prime = x.copy()
        
        # Line search along each direction in U
        for i in range(n):
            x = golden_section_search(f, x, U[i])
        
        # Update direction set: shift all directions and add new displacement direction
        for i in range(n - 1):
            U[i] = U[i + 1].copy()
        
        # New direction is the displacement from start to end of cycle
        U[n - 1] = x - x_prime
        
        # Line search along the new displacement direction
        x = golden_section_search(f, x, U[n - 1])
        
        delta = np.linalg.norm(x - x_prime)
        iterations += 1
        trajectory.append(x.copy())
        f_history.append(f(x))
    
    return x, iterations, trajectory, f_history

# ==================== CCD with Acceleration (from Task 1.1) ====================

def cyclic_coordinate_descent_with_acceleration(f, x, epsilon=1e-6, max_iter=1000):
    """
    Cyclic Coordinate Descent with acceleration step.
    """
    x = np.array(x, dtype=float)
    n = len(x)
    delta = np.inf
    iterations = 0
    trajectory = [x.copy()]
    f_history = [f(x)]
    
    while delta > epsilon and iterations < max_iter:
        x_prime = x.copy()
        
        for i in range(n):
            d = basis(i, n)
            x = golden_section_search(f, x, d)
        
        # Acceleration step
        accel_dir = x - x_prime
        if np.linalg.norm(accel_dir) > 1e-10:
            x = golden_section_search(f, x, accel_dir)
        
        delta = np.linalg.norm(x - x_prime)
        iterations += 1
        trajectory.append(x.copy())
        f_history.append(f(x))
    
    return x, iterations, trajectory, f_history

# ==================== Test Functions ====================

def ackley_function(p):
    """
    Ackley function: highly multimodal with global minimum at (0, 0) with f(0,0) = 0
    """
    x, y = p
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y)))
    return term1 + term2 + np.e + 20

def branin_function(p):
    """
    Branin function: multiple global minima
    One global minimum at (π, 2.275) with f ≈ 0.397887
    """
    x, y = p
    term1 = (y - (5.1/(4*np.pi**2))*x**2 + (5/np.pi)*x - 6)**2
    term2 = 10*(1 - 1/(8*np.pi))*np.cos(x)
    return term1 + term2 + 10

def rosenbrock_function(p):
    """Rosenbrock function: minimum at (1, 1) with f(1,1) = 0"""
    x, y = p
    return (1 - x)**2 + 5*(y - x**2)**2

# ==================== Plotting Functions ====================

def plot_trajectory(f, trajectory, title, x_range, y_range, optimum=None, filename=None):
    """
    Plot contour of function with optimization trajectory.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create grid
    x = np.linspace(x_range[0], x_range[1], 200)
    y = np.linspace(y_range[0], y_range[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f([X[i, j], Y[i, j]])
    
    # Plot contours
    z_min, z_max = Z.min(), Z.max()
    z_range = z_max - z_min
    
    if z_range < 1e-10 or z_min <= 0:
        levels = np.linspace(z_min, z_max, 30)
    else:
        try:
            levels = np.logspace(np.log10(max(z_min, 1e-10)), np.log10(z_max), 30)
        except:
            levels = np.linspace(z_min, z_max, 30)
    
    levels = levels[np.isfinite(levels)]
    
    if len(levels) > 0:
        contour = ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
        try:
            ax.clabel(contour, inline=True, fontsize=8)
        except:
            pass
    
    # Plot trajectory
    trajectory = np.array(trajectory)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r.-', linewidth=2, 
            markersize=6, label='Optimization Path', alpha=0.8)
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=12, 
            label='Start Point')
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'r*', markersize=15, 
            label='End Point')
    
    # Mark global optimum if provided
    if optimum is not None:
        ax.plot(optimum[0], optimum[1], 'b^', markersize=12, 
                label='Global Optimum')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    
    plt.close()
    return fig

def plot_convergence(f_history, title, filename=None):
    """
    Plot convergence history.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = range(len(f_history))
    ax.semilogy(iterations, f_history, 'b-o', linewidth=2, markersize=4)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Objective Value (log scale)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    
    plt.close()
    return fig

# ==================== Main Experiment Runner ====================

def run_powell_experiments():
    """Run Powell's method experiments on Ackley and Branin functions."""
    
    global func_eval_counter
    
    print("\n" + "="*80)
    print("POWELL'S METHOD OPTIMIZATION RESULTS")
    print("="*80 + "\n")
    
    experiments = [
        {
            'name': 'Ackley',
            'function': ackley_function,
            'start': (-3.0, -3.0),
            'x_range': (-5, 5),
            'y_range': (-5, 5),
            'optimum': (0, 0)
        },
        {
            'name': 'Branin',
            'function': branin_function,
            'start': (2.0, 2.0),
            'x_range': (-5, 15),
            'y_range': (-5, 20),
            'optimum': (np.pi, 2.275)
        }
    ]
    
    results = []
    
    for exp in experiments:
        func = exp['function']
        func_name = exp['name']
        start_point = exp['start']
        
        print(f"\n{func_name} Function - Starting Point: {start_point}")
        print("-" * 60)
        
        # Reset function evaluation counter
        func_eval_counter = 0
        counted_func = counted_function(func)
        
        # Run Powell's method
        start_time = time.time()
        x_final, iterations, trajectory, f_history = powell_method(
            counted_func, start_point
        )
        elapsed_time = time.time() - start_time
        
        f_final = func(x_final)
        
        print(f"Powell's Method:")
        print(f"  Final Point: ({x_final[0]:.6f}, {x_final[1]:.6f})")
        print(f"  Final Value: {f_final:.6e}")
        print(f"  Iterations: {iterations}")
        print(f"  Function Evaluations: {func_eval_counter}")
        print(f"  Time: {elapsed_time:.4f} seconds")
        
        results.append({
            'Function': func_name,
            'Start': f"({start_point[0]}, {start_point[1]})",
            'Method': 'Powell',
            'x_found': f"({x_final[0]:.6f}, {x_final[1]:.6f})",
            'f(x_found)': f"{f_final:.6e}",
            'Iterations': iterations,
            'Func_Evals': func_eval_counter,
            'Time (s)': f"{elapsed_time:.4f}"
        })
        
        # Plot trajectory
        plot_trajectory(
            func, trajectory,
            f"{func_name} Function - Powell's Method\nStart: {start_point}",
            exp['x_range'], exp['y_range'], exp['optimum'],
            filename=f'plot_{func_name}_trajectory.png'
        )
        
        # Plot convergence
        plot_convergence(
            f_history,
            f"{func_name} Function - Convergence\nPowell's Method",
            filename=f'plot_{func_name}_convergence.png'
        )
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE - POWELL'S METHOD")
    print("="*80 + "\n")
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    print(f"\n✓ Generated trajectory and convergence plots for all test functions")
    print("="*80)

# ==================== Comparative Experiments ====================

def run_comparative_experiments():
    """
    Compare Powell's method with CCD on Rosenbrock and Ackley functions.
    Using same starting points and line search for fair comparison.
    """
    
    global func_eval_counter
    
    print("\n" + "="*80)
    print("COMPARATIVE STUDY: POWELL vs CCD WITH ACCELERATION")
    print("="*80 + "\n")
    
    experiments = [
        {
            'name': 'Rosenbrock',
            'function': rosenbrock_function,
            'start': (-1.5, 2.0),
            'x_range': (-2, 3),
            'y_range': (-2, 4),
            'optimum': (1, 1)
        },
        {
            'name': 'Ackley',
            'function': ackley_function,
            'start': (4.0, 1.0),
            'x_range': (-1, 5),
            'y_range': (-1, 5),
            'optimum': (0, 0)
        }
    ]
    
    results = []
    
    for exp in experiments:
        func = exp['function']
        func_name = exp['name']
        start_point = exp['start']
        
        print(f"\n{func_name} Function - Starting Point: {start_point}")
        print("-" * 60)
        
        # Run Powell's method
        func_eval_counter = 0
        counted_func_powell = counted_function(func)
        start_time = time.time()
        x_powell, iter_powell, traj_powell, fhist_powell = powell_method(
            counted_func_powell, start_point
        )
        time_powell = time.time() - start_time
        f_powell = func(x_powell)
        evals_powell = func_eval_counter
        
        print(f"Powell's Method:")
        print(f"  Final Point: ({x_powell[0]:.6f}, {x_powell[1]:.6f})")
        print(f"  Final Value: {f_powell:.6e}")
        print(f"  Iterations: {iter_powell}")
        print(f"  Function Evaluations: {evals_powell}")
        print(f"  Time: {time_powell:.4f} seconds")
        
        # Run CCD with Acceleration
        func_eval_counter = 0
        counted_func_ccd = counted_function(func)
        start_time = time.time()
        x_ccd, iter_ccd, traj_ccd, fhist_ccd = cyclic_coordinate_descent_with_acceleration(
            counted_func_ccd, start_point
        )
        time_ccd = time.time() - start_time
        f_ccd = func(x_ccd)
        evals_ccd = func_eval_counter
        
        print(f"CCD with Acceleration:")
        print(f"  Final Point: ({x_ccd[0]:.6f}, {x_ccd[1]:.6f})")
        print(f"  Final Value: {f_ccd:.6e}")
        print(f"  Iterations: {iter_ccd}")
        print(f"  Function Evaluations: {evals_ccd}")
        print(f"  Time: {time_ccd:.4f} seconds")
        
        results.append({
            'Function': func_name,
            'Start': f"({start_point[0]}, {start_point[1]})",
            'Method': 'Powell',
            'x_found': f"({x_powell[0]:.6f}, {x_powell[1]:.6f})",
            'f(x_found)': f"{f_powell:.6e}",
            'Iterations': iter_powell,
            'Func_Evals': evals_powell,
            'Time (s)': f"{time_powell:.4f}"
        })
        
        results.append({
            'Function': func_name,
            'Start': f"({start_point[0]}, {start_point[1]})",
            'Method': 'CCD_Accel',
            'x_found': f"({x_ccd[0]:.6f}, {x_ccd[1]:.6f})",
            'f(x_found)': f"{f_ccd:.6e}",
            'Iterations': iter_ccd,
            'Func_Evals': evals_ccd,
            'Time (s)': f"{time_ccd:.4f}"
        })
        
        # Plot comparison trajectories
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Create grid for both subplots
        x = np.linspace(exp['x_range'][0], exp['x_range'][1], 200)
        y = np.linspace(exp['y_range'][0], exp['y_range'][1], 200)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = func([X[i, j], Y[i, j]])
        
        z_min, z_max = Z.min(), Z.max()
        if z_min <= 0 or z_max - z_min < 1e-10:
            levels = np.linspace(z_min, z_max, 30)
        else:
            try:
                levels = np.logspace(np.log10(max(z_min, 1e-10)), np.log10(z_max), 30)
            except:
                levels = np.linspace(z_min, z_max, 30)
        levels = levels[np.isfinite(levels)]
        
        # Powell trajectory
        if len(levels) > 0:
            contour1 = ax1.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
        traj_powell_arr = np.array(traj_powell)
        ax1.plot(traj_powell_arr[:, 0], traj_powell_arr[:, 1], 'r.-', 
                linewidth=2, markersize=6, label='Powell Path')
        ax1.plot(start_point[0], start_point[1], 'go', markersize=12, label='Start')
        ax1.plot(traj_powell_arr[-1, 0], traj_powell_arr[-1, 1], 'r*', 
                markersize=15, label='End')
        if exp['optimum']:
            ax1.plot(exp['optimum'][0], exp['optimum'][1], 'b^', 
                    markersize=12, label='Global Optimum')
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('y', fontsize=12)
        ax1.set_title(f"{func_name} - Powell's Method", fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # CCD trajectory
        if len(levels) > 0:
            contour2 = ax2.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
        traj_ccd_arr = np.array(traj_ccd)
        ax2.plot(traj_ccd_arr[:, 0], traj_ccd_arr[:, 1], 'b.-', 
                linewidth=2, markersize=6, label='CCD Path')
        ax2.plot(start_point[0], start_point[1], 'go', markersize=12, label='Start')
        ax2.plot(traj_ccd_arr[-1, 0], traj_ccd_arr[-1, 1], 'b*', 
                markersize=15, label='End')
        if exp['optimum']:
            ax2.plot(exp['optimum'][0], exp['optimum'][1], 'b^', 
                    markersize=12, label='Global Optimum')
        ax2.set_xlabel('x', fontsize=12)
        ax2.set_ylabel('y', fontsize=12)
        ax2.set_title(f"{func_name} - CCD with Acceleration", fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'plot_{func_name}_comparison_trajectories.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot convergence comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(range(len(fhist_powell)), fhist_powell, 'r-o', 
                   linewidth=2, markersize=4, label='Powell')
        ax.semilogy(range(len(fhist_ccd)), fhist_ccd, 'b-s', 
                   linewidth=2, markersize=4, label='CCD with Acceleration')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Objective Value (log scale)', fontsize=12)
        ax.set_title(f'{func_name} Function - Convergence Comparison', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(f'plot_{func_name}_comparison_convergence.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE - POWELL vs CCD WITH ACCELERATION")
    print("="*80 + "\n")
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    print(f"\n✓ Generated comparison plots for Rosenbrock and Ackley functions")
    print("="*80)
    
    # Discussion
    print("\n" + "="*80)
    print("COMPARATIVE DISCUSSION")
    print("="*80 + "\n")
    
    discussion = """
1. Objective Value Comparison:
   Powell's method typically reaches lower or comparable objective values compared to CCD 
   with acceleration. This is because Powell builds an adaptive set of conjugate-like 
   directions that better capture the geometry of the problem, especially for functions 
   like Rosenbrock with strong coupling between variables. The direction-updating mechanism 
   in Powell allows it to escape narrow valleys more efficiently.

2. Function Evaluations and Efficiency:
   While both methods use similar numbers of outer iterations, Powell's method may use 
   more function evaluations per iteration due to maintaining n+1 direction searches per 
   cycle (n coordinate directions plus one displacement direction). However, Powell often 
   converges in fewer outer iterations, making the total cost competitive. CCD with 
   acceleration is slightly faster in wall-clock time for simple functions but Powell 
   shows better robustness on complex landscapes.

3. Trajectory and Sensitivity Differences:
   Powell's trajectories show more direct paths toward optima, especially after the first 
   few iterations when good search directions have been established. CCD trajectories tend 
   to follow a more zig-zag pattern along coordinate axes before the acceleration step 
   provides diagonal movement. Powell is generally less sensitive to initial points on 
   unimodal or mildly multimodal functions because its adaptive directions help it orient 
   toward the optimum regardless of starting position. On highly multimodal functions like 
   Ackley, both methods show similar local convergence behavior once trapped in a basin.

4. Method Characteristics:
   Powell's strength lies in building problem-adapted search directions that can handle 
   rotated or scaled coordinate systems better than fixed coordinate descent. The method 
   is particularly effective on functions where the level curves are elongated ellipses 
   not aligned with coordinate axes (like Rosenbrock). CCD with acceleration is simpler 
   to implement and understand, with the acceleration step providing a basic form of 
   conjugacy, but it remains fundamentally coordinate-aligned until the acceleration 
   occurs. For practical applications, Powell is preferable when function geometry is 
   unknown and may not align with coordinates.
"""
    
    print(discussion)
    print("="*80 + "\n")

# ==================== Main Entry Point ====================

if __name__ == "__main__":
    # Run main Powell experiments
    run_powell_experiments()
    
    # Run comparative study
    run_comparative_experiments()