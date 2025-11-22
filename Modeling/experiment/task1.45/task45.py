import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from collections import OrderedDict
import heapq

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

# ==================== Nelder-Mead Method ====================

def nelder_mead(f, S, epsilon=1e-6, alpha=1.0, beta=2.0, gamma=0.5, max_iter=1000):
    """
    Nelder-Mead simplex method for unconstrained optimization.
    
    Parameters:
    - alpha: reflection coefficient (default 1.0)
    - beta: expansion coefficient (default 2.0)
    - gamma: contraction coefficient (default 0.5)
    """
    S = [np.array(p, dtype=float) for p in S]
    y_arr = np.array([f(s) for s in S])
    
    delta = np.inf
    iterations = 0
    trajectory = []  # Track centroid positions
    f_history = []
    simplex_history = []  # Track all simplex vertices
    
    while delta > epsilon and iterations < max_iter:
        # Sort by function values (lowest to highest)
        p = np.argsort(y_arr)
        S = [S[i] for i in p]
        y_arr = y_arr[p]
        
        xl, yl = S[0], y_arr[0]      # best
        xh, yh = S[-1], y_arr[-1]    # worst
        xs, ys = S[-2], y_arr[-2]    # second worst
        
        # Centroid of all points except worst
        xm = np.mean(S[:-1], axis=0)
        
        # Reflection
        xr = xm + alpha * (xm - xh)
        yr = f(xr)
        
        if yr < yl:
            # Expansion
            xe = xm + beta * (xr - xm)
            ye = f(xe)
            if ye < yr:
                S[-1], y_arr[-1] = xe, ye
            else:
                S[-1], y_arr[-1] = xr, yr
        elif yr >= ys:
            # Contraction
            if yr < yh:
                xh, yh = xr, yr
                S[-1], y_arr[-1] = xr, yr
            
            xc = xm + gamma * (xh - xm)
            yc = f(xc)
            
            if yc > yh:
                # Shrink toward best point
                for i in range(1, len(S)):
                    S[i] = (S[i] + xl) / 2
                    y_arr[i] = f(S[i])
            else:
                S[-1], y_arr[-1] = xc, yc
        else:
            # Accept reflection
            S[-1], y_arr[-1] = xr, yr
        
        # Convergence check: standard deviation of function values
        delta = np.std(y_arr, ddof=0)
        iterations += 1
        
        trajectory.append(xm.copy())
        f_history.append(np.min(y_arr))
        simplex_history.append([s.copy() for s in S])
    
    # Return best point
    best_idx = np.argmin(y_arr)
    return S[best_idx], iterations, trajectory, f_history, simplex_history

def create_initial_simplex(x0, delta=0.05):
    """
    Create initial simplex from starting point x0.
    Returns n+1 points for n-dimensional problem.
    """
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    S = [x0.copy()]
    
    for i in range(n):
        x = x0.copy()
        x[i] += delta
        S.append(x)
    
    return S

# ==================== DIRECT Method ====================

class Interval:
    """Represents a hyperrectangle in DIRECT algorithm."""
    def __init__(self, c, y, depths):
        self.c = np.array(c, dtype=float)  # center
        self.y = float(y)                   # function value
        self.depths = np.array(depths, dtype=int)  # division depth per coordinate
    
    def __lt__(self, other):
        return self.y < other.y

def min_depth(interval):
    """Return minimum depth value."""
    return np.min(interval.depths)

def vertex_dist(interval):
    """Euclidean distance from center to vertex."""
    return np.linalg.norm(0.5 * 3.0 ** (-interval.depths))

def reparameterize_to_unit_hypercube(f, a, b):
    """Map function from [a,b]^n to unit hypercube [0,1]^n."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    delta = b - a
    return lambda x: f(x * delta + a)

def rev_unit_hypercube_parameterization(x, a, b):
    """Map point from unit hypercube back to original space."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return x * (b - a) + a

def add_interval(intervals, interval):
    """Add interval to data structure."""
    d = vertex_dist(interval)
    if d not in intervals:
        intervals[d] = []
    heapq.heappush(intervals[d], interval)

def get_opt_intervals(intervals, epsilon, y_best):
    """Select potentially optimal intervals."""
    stack = []
    
    for d in sorted(intervals.keys()):
        pq = intervals[d]
        if len(pq) > 0:
            # Get interval with best function value at this distance
            interval = min(pq, key=lambda x: x.y)
            y = interval.y
            
            # Check dominance against last two in stack
            while len(stack) > 1:
                interval1 = stack[-1]
                interval2 = stack[-2]
                x1, y1 = vertex_dist(interval1), interval1.y
                x2, y2 = vertex_dist(interval2), interval2.y
                x = vertex_dist(interval)
                
                # Slope of lower bound line
                if x2 != x:
                    slope = (y2 - y) / (x2 - x)
                    if y1 <= slope * (x1 - x) + y + epsilon:
                        break
                pop_interval = stack.pop()
            
            # Check if new interval is worse than last in stack
            if len(stack) > 0 and interval.y > stack[-1].y + epsilon:
                continue
            
            stack.append(interval)
    
    return stack

def divide(f, interval):
    """Divide interval along minimal-depth coordinates."""
    c = interval.c
    d = min_depth(interval)
    n = len(c)
    
    # Find directions with minimal depth
    dirs = np.where(interval.depths == d)[0]
    
    # New centers along each direction
    cs = []
    for i in dirs:
        c_plus = c + 3.0**(-d-1) * basis(i, n)
        c_minus = c - 3.0**(-d-1) * basis(i, n)
        cs.append((c_plus, c_minus))
    
    # Evaluate function at new centers
    vs = [(f(C[0]), f(C[1])) for C in cs]
    
    # Get minimum values
    minvals = [min(V[0], V[1]) for V in vs]
    
    # Create new intervals
    new_intervals = []
    depths = interval.depths.copy()
    
    # Sort by minimum values
    for j in np.argsort(minvals):
        depths[dirs[j]] += 1
        C, V = cs[j], vs[j]
        new_intervals.append(Interval(C[0], V[0], depths.copy()))
        new_intervals.append(Interval(C[1], V[1], depths.copy()))
    
    # Add original center with updated depths
    new_intervals.append(Interval(c, interval.y, depths.copy()))
    
    return new_intervals

def direct(f, a, b, epsilon=1e-4, max_iter=150):
    """
    DIRECT (DIviding RECTangles) global optimization algorithm.
    
    Parameters:
    - f: objective function
    - a: lower bounds (array)
    - b: upper bounds (array)
    - max_iter: maximum iterations
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    n = len(a)
    
    # Reparameterize to unit hypercube
    g = reparameterize_to_unit_hypercube(f, a, b)
    
    # Initialize
    intervals = OrderedDict()
    c = np.full(n, 0.5)
    initial_interval = Interval(c, g(c), np.zeros(n, dtype=int))
    add_interval(intervals, initial_interval)
    
    c_best = initial_interval.c.copy()
    y_best = initial_interval.y
    
    all_centers = [rev_unit_hypercube_parameterization(c_best, a, b)]
    f_history = [y_best]
    
    for k in range(max_iter):
        # Select potentially optimal intervals
        S = get_opt_intervals(intervals, epsilon, y_best)
        
        # Divide selected intervals
        to_add = []
        for interval in S:
            new_intervals = divide(g, interval)
            to_add.extend(new_intervals)
            
            # Remove old interval
            d = vertex_dist(interval)
            if d in intervals:
                intervals[d] = [iv for iv in intervals[d] if iv != interval]
                if len(intervals[d]) == 0:
                    del intervals[d]
        
        # Add new intervals
        for interval in to_add:
            add_interval(intervals, interval)
            
            # Track all centers
            center_original = rev_unit_hypercube_parameterization(interval.c, a, b)
            all_centers.append(center_original)
            
            # Update best
            if interval.y < y_best:
                c_best = interval.c.copy()
                y_best = interval.y
        
        f_history.append(y_best)
    
    # Convert best point back to original space
    x_best = rev_unit_hypercube_parameterization(c_best, a, b)
    
    return x_best, max_iter, all_centers, f_history

# ==================== Test Functions ====================

def ackley_function(p):
    """Ackley function: global minimum at (0, 0) with f(0,0) = 0"""
    x, y = p
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y)))
    return term1 + term2 + np.e + 20

def branin_function(p):
    """Branin function: multiple global minima"""
    x, y = p
    term1 = (y - (5.1/(4*np.pi**2))*x**2 + (5/np.pi)*x - 6)**2
    term2 = 10*(1 - 1/(8*np.pi))*np.cos(x)
    return term1 + term2 + 10

def rosenbrock_function(p):
    """Rosenbrock function: global minimum at (1, 1) with f(1,1) = 0"""
    x, y = p
    return (1 - x)**2 + 5*(y - x**2)**2

def rastrigin_function(p):
    """Rastrigin function: global minimum at (0, 0) with f(0,0) = 0"""
    x, y = p
    return 20 + (x**2 - 10*np.cos(2*np.pi*x)) + (y**2 - 10*np.cos(2*np.pi*y))

# ==================== Plotting Functions ====================

def plot_nelder_mead_trajectory(f, trajectory, simplex_history, title, x_range, y_range, 
                                optimum=None, filename=None):
    """Plot Nelder-Mead trajectory with simplex vertices."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create grid
    x = np.linspace(x_range[0], x_range[1], 200)
    y = np.linspace(y_range[0], y_range[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                Z[i, j] = f([X[i, j], Y[i, j]])
            except:
                Z[i, j] = np.nan
    
    # Plot contours
    z_min, z_max = np.nanmin(Z), np.nanmax(Z)
    if z_min <= 0 or z_max - z_min < 1e-10:
        levels = np.linspace(z_min, z_max, 20)
    else:
        try:
            levels = np.logspace(np.log10(max(z_min, 1e-10)), np.log10(z_max), 20)
        except:
            levels = np.linspace(z_min, z_max, 20)
    
    levels = levels[np.isfinite(levels)]
    
    if len(levels) > 0:
        contour = ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
    
    # Plot simplex evolution (every few iterations)
    step = max(1, len(simplex_history) // 10)
    for k in range(0, len(simplex_history), step):
        simplex = np.array(simplex_history[k])
        # Close the simplex by adding first point at end
        simplex_closed = np.vstack([simplex, simplex[0]])
        alpha_val = 0.3 * (k / len(simplex_history))
        ax.plot(simplex_closed[:, 0], simplex_closed[:, 1], 'gray', 
               alpha=alpha_val, linewidth=1)
    
    # Plot final simplex
    if len(simplex_history) > 0:
        final_simplex = np.array(simplex_history[-1])
        final_simplex_closed = np.vstack([final_simplex, final_simplex[0]])
        ax.plot(final_simplex_closed[:, 0], final_simplex_closed[:, 1], 'r-', 
               linewidth=2, label='Final Simplex')
    
    # Plot centroid trajectory
    trajectory = np.array(trajectory)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b.-', linewidth=2, 
            markersize=4, label='Centroid Path', alpha=0.7)
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=12, 
            label='Start')
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'r*', markersize=15, 
            label='End')
    
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

def plot_direct_samples(f, all_centers, best_point, title, x_range, y_range, 
                       optimum=None, filename=None):
    """Plot DIRECT sampled centers."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create grid
    x = np.linspace(x_range[0], x_range[1], 200)
    y = np.linspace(y_range[0], y_range[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                Z[i, j] = f([X[i, j], Y[i, j]])
            except:
                Z[i, j] = np.nan
    
    # Plot contours
    z_min, z_max = np.nanmin(Z), np.nanmax(Z)
    if z_min <= 0 or z_max - z_min < 1e-10:
        levels = np.linspace(z_min, z_max, 20)
    else:
        try:
            levels = np.logspace(np.log10(max(z_min, 1e-10)), np.log10(z_max), 20)
        except:
            levels = np.linspace(z_min, z_max, 20)
    
    levels = levels[np.isfinite(levels)]
    
    if len(levels) > 0:
        contour = ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
    
    # Plot sampled centers
    all_centers = np.array(all_centers)
    ax.scatter(all_centers[:, 0], all_centers[:, 1], c='blue', s=10, 
              alpha=0.3, label='Sampled Centers')
    
    # Plot best point
    ax.plot(best_point[0], best_point[1], 'r*', markersize=15, 
            label='Best Found')
    
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

# ==================== Main Experiment Runner ====================

def run_all_experiments():
    """Run Nelder-Mead and DIRECT experiments."""
    
    global func_eval_counter
    
    print("\n" + "="*100)
    print("NELDER-MEAD AND DIRECT METHODS OPTIMIZATION RESULTS")
    print("="*100 + "\n")
    
    # Nelder-Mead experiments
    nm_experiments = [
        {
            'name': 'Ackley',
            'function': ackley_function,
            'start': (0.0, 1.0),
            'x_range': (-2, 4),
            'y_range': (-3, 3),
            'optimum': (0, 0)
        },
        {
            'name': 'Branin',
            'function': branin_function,
            'start': (2.0, 2.0),
            'x_range': (0, 4),
            'y_range': (0, 4),
            'optimum': (np.pi, 2.275)
        },
        {
            'name': 'Rosenbrock',
            'function': rosenbrock_function,
            'start': (-1.5, 2.0),
            'x_range': (-5, 2),
            'y_range': (0, 4),
            'optimum': (1, 1)
        },
        {
            'name': 'Rastrigin',
            'function': rastrigin_function,
            'start': (2.5, 2.5),
            'x_range': (-1, 6),
            'y_range': (-1, 6),
            'optimum': (0, 0)
        }
    ]
    
    # DIRECT experiments
    direct_experiments = [
        {
            'name': 'Ackley',
            'function': ackley_function,
            'bounds': ([-2, -3], [4, 3]),
            'x_range': (-2, 4),
            'y_range': (-3, 3),
            'optimum': (0, 0)
        },
        {
            'name': 'Branin',
            'function': branin_function,
            'bounds': ([0, 0], [4, 4]),
            'x_range': (0, 4),
            'y_range': (0, 4),
            'optimum': (np.pi, 2.275)
        },
        {
            'name': 'Rosenbrock',
            'function': rosenbrock_function,
            'bounds': ([-5, 0], [2, 4]),
            'x_range': (-5, 2),
            'y_range': (0, 4),
            'optimum': (1, 1)
        },
        {
            'name': 'Rastrigin',
            'function': rastrigin_function,
            'bounds': ([-1, -1], [6, 6]),
            'x_range': (-1, 6),
            'y_range': (-1, 6),
            'optimum': (0, 0)
        }
    ]
    
    results = []
    
    # Run Nelder-Mead experiments
    print("="*50)
    print("NELDER-MEAD EXPERIMENTS")
    print("="*50 + "\n")
    
    for exp in nm_experiments:
        func = exp['function']
        func_name = exp['name']
        start_point = exp['start']
        
        print(f"\n{func_name} Function - Starting Point: {start_point}")
        print("-" * 80)
        
        # Create initial simplex
        S = create_initial_simplex(start_point, delta=0.05)
        
        # Run Nelder-Mead
        func_eval_counter = 0
        counted_func = counted_function(func)
        
        start_time = time.time()
        x_nm, iter_nm, traj_nm, fhist_nm, simplex_hist = nelder_mead(
            counted_func, S, epsilon=1e-6, max_iter=1000
        )
        time_nm = time.time() - start_time
        
        f_nm = func(x_nm)
        evals_nm = func_eval_counter
        
        print(f"Nelder-Mead:")
        print(f"  Final Point: ({x_nm[0]:.6f}, {x_nm[1]:.6f})")
        print(f"  Final Value: {f_nm:.6e}")
        print(f"  Iterations: {iter_nm}")
        print(f"  Function Evaluations: {evals_nm}")
        print(f"  Time: {time_nm:.4f} seconds")
        
        results.append({
            'Function': func_name,
            'Start/Domain': f"start: {start_point}",
            'Method': 'Nelder-Mead',
            'x_found': f"({x_nm[0]:.6f}, {x_nm[1]:.6f})",
            'f(x_found)': f"{f_nm:.6e}",
            'Iterations': iter_nm,
            'f_evals': evals_nm,
            'time_s': f"{time_nm:.4f}"
        })
        
        # Plot trajectory
        plot_nelder_mead_trajectory(
            func, traj_nm, simplex_hist,
            f"{func_name} - Nelder-Mead\nStart: {start_point}",
            exp['x_range'], exp['y_range'], exp['optimum'],
            filename=f'plot_{func_name}_nelder_trajectory.png'
        )
    
    # Run DIRECT experiments
    print("\n" + "="*50)
    print("DIRECT EXPERIMENTS")
    print("="*50 + "\n")
    
    for exp in direct_experiments:
        func = exp['function']
        func_name = exp['name']
        a, b = exp['bounds']
        
        print(f"\n{func_name} Function - Bounds: [{a}, {b}]")
        print("-" * 80)
        
        # Run DIRECT
        func_eval_counter = 0
        counted_func = counted_function(func)
        
        start_time = time.time()
        x_direct, iter_direct, centers, fhist_direct = direct(
            counted_func, a, b, epsilon=1e-4, max_iter=150
        )
        time_direct = time.time() - start_time
        
        f_direct = func(x_direct)
        evals_direct = func_eval_counter
        
        print(f"DIRECT:")
        print(f"  Final Point: ({x_direct[0]:.6f}, {x_direct[1]:.6f})")
        print(f"  Final Value: {f_direct:.6e}")
        print(f"  Iterations: {iter_direct}")
        print(f"  Function Evaluations: {evals_direct}")
        print(f"  Time: {time_direct:.4f} seconds")
        
        results.append({
            'Function': func_name,
            'Start/Domain': f"bounds: [{a}, {b}]",
            'Method': 'DIRECT',
            'x_found': f"({x_direct[0]:.6f}, {x_direct[1]:.6f})",
            'f(x_found)': f"{f_direct:.6e}",
            'Iterations': iter_direct,
            'f_evals': evals_direct,
            'time_s': f"{time_direct:.4f}"
        })
        
        # Plot sampled centers
        plot_direct_samples(
            func, centers, x_direct,
            f"{func_name} - DIRECT\nBounds: [{a}, {b}]",
            exp['x_range'], exp['y_range'], exp['optimum'],
            filename=f'plot_{func_name}_direct_samples.png'
        )
    
    # Print summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE - ALL RESULTS")
    print("="*100 + "\n")
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    print(f"\n✓ Generated trajectory plots for all test functions")
    print("="*100)
    
    # Discussion
    print("\n" + "="*100)
    print("COMPARATIVE DISCUSSION")
    print("="*100 + "\n")
    
    discussion = """
1. Final Objective Comparison:
   DIRECT consistently finds better global minima on highly multimodal functions (Ackley and 
   Rastrigin) because it systematically explores the entire search space through recursive 
   domain subdivision. Nelder-Mead, being a local search method, often gets trapped in local 
   minima on these functions depending on the starting point. For smoother functions like 
   Rosenbrock and Branin, both methods perform reasonably well, though DIRECT's global 
   exploration can discover better solutions when the local basin is small or the landscape 
   is complex.

2. Function Evaluations and Efficiency:
   Nelder-Mead typically uses significantly fewer function evaluations (hundreds) compared to 
   DIRECT (thousands), as it operates locally and converges quickly within a basin of attraction. 
   However, DIRECT's evaluations are necessary for its global search guarantee - it must sample 
   the entire domain to identify promising regions. The trade-off is clear: Nelder-Mead is 
   computationally cheaper per run but may miss the global optimum, while DIRECT is more 
   expensive but provides better global coverage. Wall-clock time shows similar patterns, with 
   Nelder-Mead being faster but less reliable on multimodal problems.

3. Method Preferences:
   Nelder-Mead is preferable when: (a) the function is unimodal or has a single dominant basin, 
   (b) a good starting point is available near the optimum, (c) computational budget is limited, 
   (d) local refinement and fast convergence are priorities. DIRECT is preferable when: (a) the 
   function is highly multimodal with many local minima, (b) no prior knowledge about the 
   landscape is available, (c) finding the global optimum is critical, (d) the search domain is 
   bounded and relatively low-dimensional. For practical applications, a hybrid approach often 
   works best: use DIRECT to identify promising regions, then refine with Nelder-Mead.

4. Convergence Characteristics:
   Nelder-Mead shows rapid initial convergence followed by slow refinement as the simplex 
   contracts around the minimum. The simplex adaptation allows it to navigate valleys and 
   ridges effectively within a local region. DIRECT exhibits more uniform progress, steadily 
   improving as it subdivides regions with good function values. Its convergence can appear 
   slower initially but becomes very effective as subdivision focuses on promising areas. The 
   selection rule in DIRECT balances exploration (large intervals) with exploitation (good 
   function values), providing a principled approach to global optimization that Nelder-Mead's 
   local heuristics cannot match.
"""
    
    print(discussion)
    print("="*100 + "\n")

# ==================== Main Entry Point ====================

if __name__ == "__main__":
    # Run main experiments
    run_all_experiments()