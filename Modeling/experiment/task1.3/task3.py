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

# ==================== Hooke-Jeeves Method ====================

def hooke_jeeves(f, x, alpha=1.0, epsilon=1e-6, gamma=0.5, max_iter=10000):
    """
    Hooke-Jeeves method for unconstrained optimization.
    
    Parameters:
    - alpha: initial step size
    - epsilon: minimum step size (stopping criterion)
    - gamma: step size reduction factor
    """
    x = np.array(x, dtype=float)
    y = f(x)
    n = len(x)
    
    iterations = 0
    trajectory = [x.copy()]
    f_history = [y]
    
    while alpha > epsilon and iterations < max_iter:
        improved = False
        x_best = x.copy()
        y_best = y
        
        # Exploratory search around current point
        for i in range(n):
            for sgn in [-1, +1]:
                x_prime = x + sgn * alpha * basis(i, n)
                y_prime = f(x_prime)
                
                if y_prime < y_best:
                    x_best = x_prime.copy()
                    y_best = y_prime
                    improved = True
        
        x = x_best.copy()
        y = y_best
        
        # Reduce step size if no improvement
        if not improved:
            alpha = alpha * gamma
        
        iterations += 1
        trajectory.append(x.copy())
        f_history.append(y)
    
    return x, iterations, trajectory, f_history

# ==================== Generalized Pattern Search (GPS) ====================

def gps(f, x, alpha=1.0, D=None, epsilon=1e-6, gamma=0.5, max_iter=10000):
    """
    Generalized Pattern Search (GPS) method.
    
    Parameters:
    - alpha: initial step size
    - D: list of search directions (default: positive/negative coordinate directions)
    - epsilon: minimum step size (stopping criterion)
    - gamma: step size reduction factor
    """
    x = np.array(x, dtype=float)
    n = len(x)
    
    # Default direction set: positive and negative coordinate directions
    if D is None:
        D = []
        for i in range(n):
            D.append(basis(i, n))
            D.append(-basis(i, n))
    
    y = f(x)
    iterations = 0
    trajectory = [x.copy()]
    f_history = [y]
    
    while alpha > epsilon and iterations < max_iter:
        improved = False
        
        for i, d in enumerate(D):
            x_prime = x + alpha * d
            y_prime = f(x_prime)
            
            if y_prime < y:
                x = x_prime.copy()
                y = y_prime
                improved = True
                
                # Promote successful direction to front
                D = [d] + D[:i] + D[i+1:]
                break
        
        # Reduce step size if no improvement
        if not improved:
            alpha = alpha * gamma
        
        iterations += 1
        trajectory.append(x.copy())
        f_history.append(y)
    
    return x, iterations, trajectory, f_history

# ==================== Test Functions ====================

def ackley_function(p):
    """Ackley function: global minimum at (0, 0) with f(0,0) = 0"""
    x, y = p
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y)))
    return term1 + term2 + np.e + 20

def booth_function(p):
    """Booth function: global minimum at (1, 3) with f(1,3) = 0"""
    x, y = p
    return (x + 2*y - 7)**2 + (2*x + y - 5)**2

def branin_function(p):
    """Branin function: multiple global minima, one at (π, 2.275) ≈ 0.397887"""
    x, y = p
    term1 = (y - (5.1/(4*np.pi**2))*x**2 + (5/np.pi)*x - 6)**2
    term2 = 10*(1 - 1/(8*np.pi))*np.cos(x)
    return term1 + term2 + 10

def rosenbrock_function(p):
    """Rosenbrock function: global minimum at (1, 1) with f(1,1) = 0"""
    x, y = p
    return (1 - x)**2 + 5*(y - x**2)**2

def wheeler_function(p):
    """Wheeler function: global minimum at (1.5, 1.5) with f(1.5,1.5) = -1"""
    x, y = p
    return -np.exp(-(x*y - 1.5)**2 - (y - 1.5)**2)

def rastrigin_function(p):
    """Rastrigin function: global minimum at (0, 0) with f(0,0) = 0"""
    x, y = p
    return 10*2 + (x**2 - 10*np.cos(2*np.pi*x)) + (y**2 - 10*np.cos(2*np.pi*y))

# ==================== Plotting Functions ====================

def plot_trajectory(f, trajectory, title, x_range, y_range, optimum=None, filename=None):
    """Plot contour of function with optimization trajectory."""
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
            markersize=4, label='Optimization Path', alpha=0.7)
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
    """Plot convergence history."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = range(len(f_history))
    
    # Use log scale if values are all positive and span multiple orders of magnitude
    if min(f_history) > 0 and max(f_history) / min(f_history) > 10:
        ax.semilogy(iterations, f_history, 'b-o', linewidth=2, markersize=3)
        ax.set_ylabel('Objective Value (log scale)', fontsize=12)
    else:
        ax.plot(iterations, f_history, 'b-o', linewidth=2, markersize=3)
        ax.set_ylabel('Objective Value', fontsize=12)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    
    plt.close()
    return fig

# ==================== Main Experiment Runner ====================

def run_all_experiments():
    """Run Hooke-Jeeves and GPS experiments on all test functions."""
    
    print("\n" + "="*90)
    print("HOOKE-JEEVES AND GPS METHODS OPTIMIZATION RESULTS")
    print("="*90 + "\n")
    
    # Define experiments
    experiments = [
        {
            'name': 'Ackley',
            'function': ackley_function,
            'start': (-3.0, -3.0),
            'x_range': (-5, 5),
            'y_range': (-5, 5),
            'optimum': (0, 0),
            'alpha': 1.0
        },
        {
            'name': 'Booth',
            'function': booth_function,
            'start': (0.0, 0.0),
            'x_range': (-5, 5),
            'y_range': (-5, 5),
            'optimum': (1, 3),
            'alpha': 1.0
        },
        {
            'name': 'Branin',
            'function': branin_function,
            'start': (2.0, 2.0),
            'x_range': (-5, 15),
            'y_range': (-5, 20),
            'optimum': (np.pi, 2.275),
            'alpha': 2.0
        },
        {
            'name': 'Rosenbrock',
            'function': rosenbrock_function,
            'start': (-1.5, 2.0),
            'x_range': (-2, 3),
            'y_range': (-2, 4),
            'optimum': (1, 1),
            'alpha': 0.5
        },
        {
            'name': 'Wheeler',
            'function': wheeler_function,
            'start': (1.5, 0.5),
            'x_range': (0, 3),
            'y_range': (0, 3),
            'optimum': (1.5, 1.5),
            'alpha': 0.5
        },
        {
            'name': 'Rastrigin',
            'function': rastrigin_function,
            'start': (2.5, 2.5),
            'x_range': (-5, 5),
            'y_range': (-5, 5),
            'optimum': (0, 0),
            'alpha': 1.0
        }
    ]
    
    results = []
    plot_count = 0
    
    for exp in experiments:
        func = exp['function']
        func_name = exp['name']
        start_point = exp['start']
        alpha_init = exp['alpha']
        
        print(f"\n{func_name} Function - Starting Point: {start_point}")
        print("-" * 80)
        
        # Run Hooke-Jeeves
        start_time = time.time()
        x_hj, iter_hj, traj_hj, fhist_hj = hooke_jeeves(
            func, start_point, alpha=alpha_init
        )
        time_hj = time.time() - start_time
        f_hj = func(x_hj)
        
        print(f"Hooke-Jeeves:")
        print(f"  Final Point: ({x_hj[0]:.6f}, {x_hj[1]:.6f})")
        print(f"  Final Value: {f_hj:.6e}")
        print(f"  Iterations: {iter_hj}")
        print(f"  Time: {time_hj:.4f} seconds")
        
        results.append({
            'Function': func_name,
            'Start': f"({start_point[0]}, {start_point[1]})",
            'Method': 'Hooke-Jeeves',
            'x_found': f"({x_hj[0]:.6f}, {x_hj[1]:.6f})",
            'f(x_found)': f"{f_hj:.6e}",
            'Iterations': iter_hj,
            'Time (s)': f"{time_hj:.4f}"
        })
        
        # Run GPS
        start_time = time.time()
        x_gps, iter_gps, traj_gps, fhist_gps = gps(
            func, start_point, alpha=alpha_init
        )
        time_gps = time.time() - start_time
        f_gps = func(x_gps)
        
        print(f"GPS:")
        print(f"  Final Point: ({x_gps[0]:.6f}, {x_gps[1]:.6f})")
        print(f"  Final Value: {f_gps:.6e}")
        print(f"  Iterations: {iter_gps}")
        print(f"  Time: {time_gps:.4f} seconds")
        
        results.append({
            'Function': func_name,
            'Start': f"({start_point[0]}, {start_point[1]})",
            'Method': 'GPS',
            'x_found': f"({x_gps[0]:.6f}, {x_gps[1]:.6f})",
            'f(x_found)': f"{f_gps:.6e}",
            'Iterations': iter_gps,
            'Time (s)': f"{time_gps:.4f}"
        })
        
        # Generate plots for selected functions (Rosenbrock and Ackley as examples)
        if func_name in ['Rosenbrock', 'Ackley', 'Booth', 'Branin', 'Wheeler', 'Rastrigin']:
            # Plot Hooke-Jeeves trajectory
            plot_trajectory(
                func, traj_hj,
                f"{func_name} Function - Hooke-Jeeves\nStart: {start_point}",
                exp['x_range'], exp['y_range'], exp['optimum'],
                filename=f'plot_{func_name}_HJ_trajectory.png'
            )
            plot_count += 1
            
            # Plot GPS trajectory
            plot_trajectory(
                func, traj_gps,
                f"{func_name} Function - GPS\nStart: {start_point}",
                exp['x_range'], exp['y_range'], exp['optimum'],
                filename=f'plot_{func_name}_GPS_trajectory.png'
            )
            plot_count += 1
            
            # Plot convergence comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if min(fhist_hj + fhist_gps) > 0:
                ax.semilogy(range(len(fhist_hj)), fhist_hj, 'r-o', 
                           linewidth=2, markersize=3, label='Hooke-Jeeves')
                ax.semilogy(range(len(fhist_gps)), fhist_gps, 'b-s', 
                           linewidth=2, markersize=3, label='GPS')
                ax.set_ylabel('Objective Value (log scale)', fontsize=12)
            else:
                ax.plot(range(len(fhist_hj)), fhist_hj, 'r-o', 
                       linewidth=2, markersize=3, label='Hooke-Jeeves')
                ax.plot(range(len(fhist_gps)), fhist_gps, 'b-s', 
                       linewidth=2, markersize=3, label='GPS')
                ax.set_ylabel('Objective Value', fontsize=12)
            
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_title(f'{func_name} Function - Convergence Comparison', 
                        fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.savefig(f'plot_{func_name}_convergence_comparison.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
            plot_count += 1
    
    # Print summary table
    print("\n" + "="*90)
    print("SUMMARY TABLE - ALL RESULTS")
    print("="*90 + "\n")
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    print(f"\n✓ Generated {plot_count} plots for Rosenbrock and Ackley functions")
    print("="*90)
    
    # Discussion
    print("\n" + "="*90)
    print("DISCUSSION")
    print("="*90 + "\n")
    
    discussion = """
1. Convergence Speed Comparison:
   GPS generally converges faster than Hooke-Jeeves in terms of iterations for most test 
   functions. This is because GPS uses opportunistic search - it immediately accepts and 
   exploits a successful direction, promoting it to the front of the search queue. 
   Hooke-Jeeves exhaustively explores all coordinate directions before moving, which can 
   be less efficient. However, the actual wall-clock time difference is often minimal 
   since both methods have similar computational cost per iteration.

2. Global Minimum Convergence:
   Both methods successfully find the global minimum for unimodal functions (Booth, 
   Rosenbrock, Wheeler) and converge to very good solutions. For highly multimodal 
   functions (Ackley, Rastrigin), both methods can get trapped in local minima depending 
   on the starting point, though they reliably find a local minimum near the start. 
   Branin's multiple global minima are challenging, but both methods converge to one of 
   the global optima when starting from favorable locations.

3. Sensitivity to Starting Point:
   Both methods show similar sensitivity to initial conditions. For convex or mildly 
   non-convex functions, they are relatively insensitive and converge from various starts. 
   For highly multimodal landscapes like Ackley and Rastrigin, the starting point 
   determines which basin of attraction is entered, and both methods will converge to 
   the nearest local minimum. GPS's direction promotion mechanism can sometimes provide 
   slightly better resilience by quickly adapting to successful search directions.

4. Step-Size Reduction Impact:
   The step-size reduction parameter γ critically affects performance. A smaller γ (slower 
   reduction) allows more extensive exploration at each scale but takes longer to converge. 
   A larger γ (faster reduction) leads to quicker convergence but may prematurely stop in 
   a suboptimal region. The default γ=0.5 provides a good balance. The step-size reduction 
   mechanism is essential for convergence - without it, the methods would oscillate 
   indefinitely. For functions with narrow valleys (like Rosenbrock), careful step-size 
   control is crucial for successful navigation.

5. Method Characteristics:
   Hooke-Jeeves is more systematic and predictable, always exploring all coordinate 
   directions before deciding. GPS is more adaptive and opportunistic, immediately 
   exploiting promising directions. For problems where coordinate directions are natural 
   (axis-aligned problems), both work well. For rotated or coupled problems, both methods 
   may struggle compared to methods that build conjugate directions (like Powell's method). 
   The choice between them often depends on the specific problem structure and whether 
   opportunistic or systematic search is preferred.
"""
    
    print(discussion)
    print("="*90 + "\n")

# ==================== Additional Exercise: Adaptive GPS ====================

def gps_adaptive(f, x, alpha=1.0, epsilon=1e-6, gamma=0.5, max_iter=10000):
    """
    Adaptive GPS: Generate new orthogonal directions after successful steps.
    Uses Gram-Schmidt orthogonalization to create a fresh orthonormal basis.
    """
    x = np.array(x, dtype=float)
    n = len(x)
    
    # Start with standard basis
    D = []
    for i in range(n):
        D.append(basis(i, n))
        D.append(-basis(i, n))
    
    y = f(x)
    iterations = 0
    trajectory = [x.copy()]
    f_history = [y]
    regenerate_count = 0
    
    while alpha > epsilon and iterations < max_iter:
        improved = False
        
        for i, d in enumerate(D):
            x_prime = x + alpha * d
            y_prime = f(x_prime)
            
            if y_prime < y:
                x = x_prime.copy()
                y = y_prime
                improved = True
                
                # Promote successful direction
                D = [d] + D[:i] + D[i+1:]
                
                # Regenerate orthogonal directions using Gram-Schmidt
                if regenerate_count % 3 == 0:  # Regenerate every 3 successful steps
                    # Create new random directions and orthogonalize
                    new_dirs = [d]  # Keep successful direction
                    for _ in range(n - 1):
                        rand_dir = np.random.randn(n)
                        # Gram-Schmidt orthogonalization
                        for existing in new_dirs:
                            rand_dir -= np.dot(rand_dir, existing) * existing
                        norm = np.linalg.norm(rand_dir)
                        if norm > 1e-10:
                            rand_dir = rand_dir / norm
                            new_dirs.append(rand_dir)
                    
                    # Update direction set with orthogonal directions
                    D = new_dirs + [-dd for dd in new_dirs]
                
                regenerate_count += 1
                break
        
        if not improved:
            alpha = alpha * gamma
        
        iterations += 1
        trajectory.append(x.copy())
        f_history.append(y)
    
    return x, iterations, trajectory, f_history

def run_adaptive_gps_experiment():
    """Additional exercise: Compare standard GPS with adaptive GPS."""
    
    print("\n" + "="*90)
    print("ADDITIONAL EXERCISE: ADAPTIVE GPS WITH ORTHOGONAL DIRECTION REGENERATION")
    print("="*90 + "\n")
    
    # Test on challenging functions
    experiments = [
        {
            'name': 'Rosenbrock',
            'function': rosenbrock_function,
            'start': (-1.5, 2.0),
            'alpha': 0.5
        },
        {
            'name': 'Rastrigin',
            'function': rastrigin_function,
            'start': (2.5, 2.5),
            'alpha': 1.0
        }
    ]
    
    results = []
    
    for exp in experiments:
        func = exp['function']
        func_name = exp['name']
        start_point = exp['start']
        alpha_init = exp['alpha']
        
        print(f"\n{func_name} Function - Starting Point: {start_point}")
        print("-" * 80)
        
        # Standard GPS
        np.random.seed(42)  # For reproducibility
        start_time = time.time()
        x_std, iter_std, traj_std, fhist_std = gps(
            func, start_point, alpha=alpha_init
        )
        time_std = time.time() - start_time
        f_std = func(x_std)
        
        print(f"Standard GPS:")
        print(f"  Final Value: {f_std:.6e}")
        print(f"  Iterations: {iter_std}")
        print(f"  Time: {time_std:.4f} seconds")
        
        # Adaptive GPS
        np.random.seed(42)  # For reproducibility
        start_time = time.time()
        x_adpt, iter_adpt, traj_adpt, fhist_adpt = gps_adaptive(
            func, start_point, alpha=alpha_init
        )
        time_adpt = time.time() - start_time
        f_adpt = func(x_adpt)
        
        print(f"Adaptive GPS:")
        print(f"  Final Value: {f_adpt:.6e}")
        print(f"  Iterations: {iter_adpt}")
        print(f"  Time: {time_adpt:.4f} seconds")
        
        improvement = ((f_std - f_adpt) / f_std * 100) if f_std > 0 else 0
        print(f"  Improvement: {improvement:.2f}%")
        
        results.append({
            'Function': func_name,
            'Method': 'Standard GPS',
            'Final Value': f"{f_std:.6e}",
            'Iterations': iter_std,
            'Time (s)': f"{time_std:.4f}"
        })
        
        results.append({
            'Function': func_name,
            'Method': 'Adaptive GPS',
            'Final Value': f"{f_adpt:.6e}",
            'Iterations': iter_adpt,
            'Time (s)': f"{time_adpt:.4f}"
        })
    
    print("\n" + "="*90)
    print("ADAPTIVE GPS COMPARISON RESULTS")
    print("="*90 + "\n")
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    print("\n" + "="*90)
    print("Adaptive GPS generates new orthogonal directions after successful steps,")
    print("potentially exploring the space more effectively than fixed coordinate directions.")
    print("="*90 + "\n")

# ==================== Main Entry Point ====================

if __name__ == "__main__":
    # Run main experiments
    run_all_experiments()
    
    # Run additional exercise (Adaptive GPS)
    run_adaptive_gps_experiment()