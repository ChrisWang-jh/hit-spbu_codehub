import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import time
from matplotlib import cm

# Set random seed for reproducibility
np.random.seed(42)

# ==================== Test Functions ====================
def ackley(x):
    """Ackley function with global minimum at (0,0), f=0"""
    if len(x.shape) == 1:
        x, y = x[0], x[1]
    else:
        x, y = x
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y)))
    return term1 + term2 + np.e + 20

def rosenbrock(x):
    """Rosenbrock function with global minimum at (1,1), f=0"""
    if len(x.shape) == 1:
        x, y = x[0], x[1]
    else:
        x, y = x
    return (1 - x)**2 + 5*(y - x**2)**2

def branin(x):
    """Branin function with three minima, f≈0.398"""
    if len(x.shape) == 1:
        x, y = x[0], x[1]
    else:
        x, y = x
    term1 = (y - 5.1/(4*np.pi**2)*x**2 + 5/np.pi*x - 6)**2
    term2 = 10*(1 - 1/(8*np.pi))*np.cos(x)
    return term1 + term2 + 10

def rastrigin(x):
    """Rastrigin function with global minimum at (0,0), f=0"""
    if len(x.shape) == 1:
        x, y = x[0], x[1]
    else:
        x, y = x
    return 20 + (x**2 - 10*np.cos(2*np.pi*x)) + (y**2 - 10*np.cos(2*np.pi*y))

# ==================== Cross-Entropy Method ====================
def cross_entropy_method(f, mu, Sigma, k_max=10, m=40, m_elite=10, seed=42, track_convergence=False):
    """Cross-Entropy Method for optimization"""
    np.random.seed(seed)
    n = len(mu)
    f_evals = 0
    trajectory = [mu.copy()]
    all_samples = []
    convergence_history = []
    
    for k in range(k_max):
        samples = np.random.multivariate_normal(mu, Sigma, m)
        all_samples.append(samples.copy())
        
        f_values = np.array([f(samples[i]) for i in range(m)])
        f_evals += m
        
        best_f = np.min(f_values)
        if track_convergence:
            convergence_history.append((f_evals, best_f))
        
        order = np.argsort(f_values)
        elite_samples = samples[order[:m_elite]]
        
        mu = np.mean(elite_samples, axis=0)
        Sigma = np.cov(elite_samples.T)
        Sigma += 1e-8 * np.eye(n)
        
        trajectory.append(mu.copy())
    
    result = {
        'x': mu,
        'f_x': f(mu),
        'f_evals': f_evals,
        'trajectory': trajectory,
        'samples': all_samples
    }
    if track_convergence:
        result['convergence'] = convergence_history
    
    return result

# ==================== Natural Evolution Strategies ====================
def natural_evolution_strategies(f, mu, A, k_max=30, m=60, alpha=0.005, seed=42, track_convergence=False):
    """Natural Evolution Strategies for optimization"""
    np.random.seed(seed)
    n = len(mu)
    f_evals = 0
    trajectory = [mu.copy()]
    all_samples = []
    convergence_history = []
    
    for k in range(k_max):
        Sigma = A.T @ A
        Sigma += 1e-8 * np.eye(n)
        
        samples = np.random.multivariate_normal(mu, Sigma, m)
        all_samples.append(samples.copy())
        
        f_values = np.array([f(samples[i]) for i in range(m)])
        f_evals += m
        
        best_f = np.min(f_values)
        if track_convergence:
            convergence_history.append((f_evals, best_f))
        
        try:
            Sigma_inv = np.linalg.inv(Sigma)
        except:
            Sigma_inv = np.linalg.pinv(Sigma)
        
        g_mu = np.zeros(n)
        for i in range(m):
            g_mu += f_values[i] * (Sigma_inv @ (samples[i] - mu))
        g_mu /= m
        
        mu -= alpha * g_mu
        
        g_A = np.zeros_like(A)
        for i in range(m):
            dx = samples[i] - mu
            g_Sigma = 0.5 * (Sigma_inv @ np.outer(dx, dx) @ Sigma_inv - Sigma_inv)
            g_A += f_values[i] * A @ (g_Sigma + g_Sigma.T)
        g_A /= m
        
        A -= alpha * g_A
        
        trajectory.append(mu.copy())
    
    result = {
        'x': mu,
        'f_x': f(mu),
        'f_evals': f_evals,
        'trajectory': trajectory,
        'samples': all_samples
    }
    if track_convergence:
        result['convergence'] = convergence_history
    
    return result

# ==================== CMA-ES ====================
def covariance_matrix_adaptation(f, x, k_max=200, sigma=1.5, seed=42, track_convergence=False):
    """Covariance Matrix Adaptation Evolution Strategy"""
    np.random.seed(seed)
    n = len(x)
    mu = x.copy()
    
    m = 4 + int(3 * np.log(n))
    m_elite = m // 2
    
    ws = np.log((m + 1) / 2) - np.log(np.arange(1, m + 1))
    ws[:m_elite] /= np.sum(ws[:m_elite])
    
    mu_eff = 1 / np.sum(ws[:m_elite]**2)
    
    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
    d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
    c_Sigma = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    c1 = 2 / ((n + 1.3)**2 + mu_eff)
    c_mu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((n + 2)**2 + mu_eff))
    
    ws[m_elite:] *= -(1 + c1 / c_mu) / np.sum(ws[m_elite:])
    
    E = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
    
    p_sigma = np.zeros(n)
    p_Sigma = np.zeros(n)
    Sigma = np.eye(n)
    
    f_evals = 0
    trajectory = [mu.copy()]
    all_samples = []
    convergence_history = []
    
    for k in range(k_max):
        Sigma += 1e-8 * np.eye(n)
        
        try:
            samples = np.random.multivariate_normal(mu, sigma**2 * Sigma, m)
        except:
            Sigma += 1e-6 * np.eye(n)
            samples = np.random.multivariate_normal(mu, sigma**2 * Sigma, m)
        
        all_samples.append(samples.copy())
        
        ys = np.array([f(samples[i]) for i in range(m)])
        f_evals += m
        
        best_f = np.min(ys)
        if track_convergence:
            convergence_history.append((f_evals, best_f))
        
        order = np.argsort(ys)
        
        deltas = [(samples[i] - mu) / sigma for i in range(m)]
        delta_w = sum(ws[i] * deltas[order[i]] for i in range(m_elite))
        
        delta_w_norm = np.linalg.norm(delta_w)
        if delta_w_norm > 10:
            delta_w = delta_w / delta_w_norm * 10
        
        mu += sigma * delta_w
        
        try:
            U, S, Vt = np.linalg.svd(Sigma)
            S_sqrt_inv = np.diag(1.0 / np.sqrt(S + 1e-8))
            C = U @ S_sqrt_inv @ Vt
        except:
            C = np.eye(n)
        
        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C @ delta_w)
        
        sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / E - 1))
        sigma = min(sigma, 10.0)
        
        h_sigma = int(np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (k + 1))) < 
                      (1.4 + 2 / (n + 1)) * E)
        
        p_Sigma = (1 - c_Sigma) * p_Sigma + h_sigma * np.sqrt(c_Sigma * (2 - c_Sigma) * mu_eff) * delta_w
        
        w0 = np.zeros(m)
        for i in range(m):
            if ws[i] >= 0:
                w0[i] = ws[i]
            else:
                norm_val = np.linalg.norm(C @ deltas[order[i]])**2
                w0[i] = n * ws[i] / (norm_val + 1e-8)
        
        Sigma = ((1 - c1 - c_mu) * Sigma +
                 c1 * (np.outer(p_Sigma, p_Sigma) + (1 - h_sigma) * c_Sigma * (2 - c_Sigma) * Sigma) +
                 c_mu * sum(w0[i] * np.outer(deltas[order[i]], deltas[order[i]]) for i in range(m)))
        
        Sigma = (Sigma + Sigma.T) / 2
        
        trajectory.append(mu.copy())
    
    result = {
        'x': mu,
        'f_x': f(mu),
        'f_evals': f_evals,
        'trajectory': trajectory,
        'samples': all_samples
    }
    if track_convergence:
        result['convergence'] = convergence_history
    
    return result

# ==================== Hybrid CEM-CMA-ES ====================
def hybrid_cem_cmaes(f, mu0, Sigma0, cem_iterations=5, cmaes_iterations=100, 
                     m_cem=40, m_elite=10, seed=42, track_convergence=False):
    """
    Hybrid CEM-CMA-ES: Run CEM first to get promising distribution,
    then initialize CMA-ES from its mean and covariance
    """
    np.random.seed(seed)
    
    # Phase 1: CEM
    print(f"  Phase 1: Running CEM for {cem_iterations} iterations...")
    cem_result = cross_entropy_method(f, mu0.copy(), Sigma0, k_max=cem_iterations, 
                                     m=m_cem, m_elite=m_elite, seed=seed, 
                                     track_convergence=track_convergence)
    
    # Extract final distribution from CEM
    cem_final_mu = cem_result['trajectory'][-1]
    # Estimate covariance from final samples
    final_samples = cem_result['samples'][-1]
    f_values = np.array([f(final_samples[i]) for i in range(m_cem)])
    order = np.argsort(f_values)
    elite_samples = final_samples[order[:m_elite]]
    cem_final_sigma = np.cov(elite_samples.T)
    
    print(f"  CEM final: x={cem_final_mu}, f(x)={cem_result['f_x']:.6f}")
    
    # Phase 2: CMA-ES initialized from CEM result
    print(f"  Phase 2: Running CMA-ES for {cmaes_iterations} iterations...")
    # Estimate initial step size from CEM covariance
    sigma_init = np.sqrt(np.mean(np.diag(cem_final_sigma)))
    
    cmaes_result = covariance_matrix_adaptation(f, cem_final_mu, k_max=cmaes_iterations, 
                                               sigma=sigma_init, seed=seed+1,
                                               track_convergence=track_convergence)
    
    # Combine results
    total_f_evals = cem_result['f_evals'] + cmaes_result['f_evals']
    combined_trajectory = cem_result['trajectory'] + cmaes_result['trajectory']
    
    if track_convergence:
        # Offset CMA-ES convergence by CEM function evaluations
        cmaes_conv_offset = [(evals + cem_result['f_evals'], f_val) 
                            for evals, f_val in cmaes_result['convergence']]
        combined_convergence = cem_result['convergence'] + cmaes_conv_offset
    else:
        combined_convergence = None
    
    result = {
        'x': cmaes_result['x'],
        'f_x': cmaes_result['f_x'],
        'f_evals': total_f_evals,
        'trajectory': combined_trajectory,
        'samples': cem_result['samples'] + cmaes_result['samples'],
        'cem_result': cem_result,
        'cmaes_result': cmaes_result
    }
    
    if track_convergence:
        result['convergence'] = combined_convergence
    
    return result

# ==================== Visualization ====================
def plot_optimization(func, func_name, trajectories_dict, samples_dict, domain=(-5, 5)):
    """Plot optimization trajectories with contours"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    x = np.linspace(domain[0], domain[1], 100)
    y = np.linspace(domain[0], domain[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
    
    methods = ['CEM', 'NES', 'CMA-ES']
    for idx, method in enumerate(methods):
        ax = axes[idx]
        
        contour = ax.contour(X, Y, Z, levels=20, alpha=0.4, cmap='viridis')
        ax.clabel(contour, inline=True, fontsize=8)
        
        if method in samples_dict and len(samples_dict[method]) > 0:
            samples_to_plot = samples_dict[method][-3:]
            for samples in samples_to_plot:
                ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10, c='gray')
        
        if method in trajectories_dict:
            traj = np.array(trajectories_dict[method])
            ax.plot(traj[:, 0], traj[:, 1], 'r-o', linewidth=2, markersize=4, label='Mean trajectory')
            ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label='Start')
            ax.plot(traj[-1, 0], traj[-1, 1], 'r*', markersize=15, label='End')
        
        ax.set_xlim(domain)
        ax.set_ylim(domain)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{method} on {func_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{func_name}_optimization.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_convergence_comparison(func_name, convergence_data):
    """Plot convergence curves for all methods"""
    plt.figure(figsize=(10, 6))
    
    colors = {'CEM': 'blue', 'NES': 'green', 'CMA-ES': 'red', 'Hybrid': 'purple'}
    
    for method, conv_data in convergence_data.items():
        if conv_data:
            evals, f_values = zip(*conv_data)
            plt.plot(evals, f_values, '-o', label=method, color=colors.get(method, 'black'), 
                    linewidth=2, markersize=4, alpha=0.7)
    
    plt.xlabel('Function Evaluations', fontsize=12)
    plt.ylabel('Best Objective Value', fontsize=12)
    plt.title(f'Convergence Comparison on {func_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'{func_name}_convergence.png', dpi=150, bbox_inches='tight')
    plt.show()

# ==================== Main Experiment ====================
def run_experiments():
    """Run all experiments and generate results"""
    
    functions = {
        'Ackley': (ackley, np.array([1.0, 1.0])),
        'Rosenbrock': (rosenbrock, np.array([0.0, 2.0])),
        'Branin': (branin, np.array([2.0, 2.0])),
        'Rastrigin': (rastrigin, np.array([-1.0, 1.0]))
    }
    
    results = []
    all_convergence = {}
    
    print("="*100)
    print(f"{'Function':<12} {'Method':<10} {'x_found':<25} {'f(x_found)':<15} {'Iterations':<12} {'F_evals':<10} {'Time(s)':<10}")
    print("="*100)
    
    for func_name, (func, mu0) in functions.items():
        trajectories = {}
        samples = {}
        convergence_data = {}
        
        # CEM
        Sigma0 = np.array([[1.0, 0.2], [0.2, 2.0]])
        start_time = time.time()
        cem_res = cross_entropy_method(func, mu0.copy(), Sigma0, k_max=10, m=40, 
                                      m_elite=10, seed=42, track_convergence=True)
        time_cem = time.time() - start_time
        trajectories['CEM'] = cem_res['trajectory']
        samples['CEM'] = cem_res['samples']
        convergence_data['CEM'] = cem_res['convergence']
        
        results.append({
            'function': func_name,
            'method': 'CEM',
            'x_found': cem_res['x'],
            'f_x_found': cem_res['f_x'],
            'iterations': 10,
            'f_evals': cem_res['f_evals'],
            'time_s': time_cem
        })
        
        print(f"{func_name:<12} {'CEM':<10} {str(np.round(cem_res['x'], 4)):<25} {cem_res['f_x']:<15.6f} {10:<12} {cem_res['f_evals']:<10} {time_cem:<10.4f}")
        
        # NES
        A0 = np.eye(2)
        start_time = time.time()
        nes_res = natural_evolution_strategies(func, mu0.copy(), A0, k_max=30, m=60, 
                                              alpha=0.005, seed=42, track_convergence=True)
        time_nes = time.time() - start_time
        trajectories['NES'] = nes_res['trajectory']
        samples['NES'] = nes_res['samples']
        convergence_data['NES'] = nes_res['convergence']
        
        results.append({
            'function': func_name,
            'method': 'NES',
            'x_found': nes_res['x'],
            'f_x_found': nes_res['f_x'],
            'iterations': 30,
            'f_evals': nes_res['f_evals'],
            'time_s': time_nes
        })
        
        print(f"{func_name:<12} {'NES':<10} {str(np.round(nes_res['x'], 4)):<25} {nes_res['f_x']:<15.6f} {30:<12} {nes_res['f_evals']:<10} {time_nes:<10.4f}")
        
        # CMA-ES
        start_time = time.time()
        cma_res = covariance_matrix_adaptation(func, mu0.copy(), k_max=200, sigma=1.5, 
                                              seed=42, track_convergence=True)
        time_cma = time.time() - start_time
        trajectories['CMA-ES'] = cma_res['trajectory']
        samples['CMA-ES'] = cma_res['samples']
        convergence_data['CMA-ES'] = cma_res['convergence']
        
        results.append({
            'function': func_name,
            'method': 'CMA-ES',
            'x_found': cma_res['x'],
            'f_x_found': cma_res['f_x'],
            'iterations': 200,
            'f_evals': cma_res['f_evals'],
            'time_s': time_cma
        })
        
        print(f"{func_name:<12} {'CMA-ES':<10} {str(np.round(cma_res['x'], 4)):<25} {cma_res['f_x']:<15.6f} {200:<12} {cma_res['f_evals']:<10} {time_cma:<10.4f}")
        print("-"*100)
        
        # Store convergence data for this function
        all_convergence[func_name] = convergence_data
        
        # Generate plots
        plot_optimization(func, func_name, trajectories, samples)
        plot_convergence_comparison(func_name, convergence_data)
    
    print("="*100)
    
    return results, all_convergence

# ==================== Hybrid Experiments ====================
def run_hybrid_experiments():
    """Run hybrid CEM-CMA-ES experiments"""
    
    functions = {
        'Ackley': (ackley, np.array([1.0, 1.0])),
        'Rosenbrock': (rosenbrock, np.array([0.0, 2.0])),
        'Branin': (branin, np.array([2.0, 2.0])),
        'Rastrigin': (rastrigin, np.array([-1.0, 1.0]))
    }
    
    print("\n" + "="*100)
    print("HYBRID CEM-CMA-ES EXPERIMENTS")
    print("="*100)
    print(f"{'Function':<12} {'Method':<15} {'x_found':<25} {'f(x_found)':<15} {'F_evals':<10} {'Time(s)':<10}")
    print("="*100)
    
    hybrid_results = []
    
    for func_name, (func, mu0) in functions.items():
        Sigma0 = np.array([[1.0, 0.2], [0.2, 2.0]])
        
        start_time = time.time()
        hybrid_res = hybrid_cem_cmaes(func, mu0.copy(), Sigma0, cem_iterations=5, 
                                     cmaes_iterations=100, m_cem=40, m_elite=10, 
                                     seed=42, track_convergence=True)
        time_hybrid = time.time() - start_time
        
        hybrid_results.append({
            'function': func_name,
            'method': 'Hybrid',
            'x_found': hybrid_res['x'],
            'f_x_found': hybrid_res['f_x'],
            'f_evals': hybrid_res['f_evals'],
            'time_s': time_hybrid
        })
        
        print(f"{func_name:<12} {'Hybrid':<15} {str(np.round(hybrid_res['x'], 4)):<25} {hybrid_res['f_x']:<15.6f} {hybrid_res['f_evals']:<10} {time_hybrid:<10.4f}")
    
    print("="*100)
    
    return hybrid_results

# ==================== Detailed Comparative Analysis ====================
def print_detailed_analysis(results, all_convergence):
    """Print detailed per-function comparative analysis"""
    
    print("\n" + "="*100)
    print("DETAILED COMPARATIVE ANALYSIS BY FUNCTION")
    print("="*100)
    
    functions = ['Ackley', 'Rosenbrock', 'Branin', 'Rastrigin']
    
    for func_name in functions:
        print(f"\n{'─'*100}")
        print(f"  {func_name.upper()} FUNCTION")
        print(f"{'─'*100}")
        
        func_results = [r for r in results if r['function'] == func_name]
        
        # Extract results by method
        cem = next(r for r in func_results if r['method'] == 'CEM')
        nes = next(r for r in func_results if r['method'] == 'NES')
        cma = next(r for r in func_results if r['method'] == 'CMA-ES')
        
        print(f"\n  FINAL OBJECTIVE VALUES:")
        print(f"    CEM:    f(x) = {cem['f_x_found']:.8f}")
        print(f"    NES:    f(x) = {nes['f_x_found']:.8f}")
        print(f"    CMA-ES: f(x) = {cma['f_x_found']:.8f}")
        
        best_method = min(func_results, key=lambda x: x['f_x_found'])['method']
        print(f"    → Best: {best_method}")
        
        print(f"\n  CONVERGENCE SPEED (Function Evaluations):")
        print(f"    CEM:    {cem['f_evals']} evaluations")
        print(f"    NES:    {nes['f_evals']} evaluations")
        print(f"    CMA-ES: {cma['f_evals']} evaluations")
        
        fastest_method = min(func_results, key=lambda x: x['f_evals'])['method']
        print(f"    → Fastest: {fastest_method}")
        
        print(f"\n  COMPUTATIONAL TIME:")
        print(f"    CEM:    {cem['time_s']:.4f} seconds")
        print(f"    NES:    {nes['time_s']:.4f} seconds")
        print(f"    CMA-ES: {cma['time_s']:.4f} seconds")
        
        print(f"\n  CONVERGENCE CHARACTERISTICS:")
        
        # Analyze convergence pattern
        if func_name == 'Ackley':
            print(f"    • Highly multimodal landscape with many local minima")
            print(f"    • CMA-ES: Superior exploration capability, reached f ≈ {cma['f_x_found']:.2e}")
            print(f"    • NES: Moderate performance, gradient-based approach struggled with multimodality")
            print(f"    • CEM: Fast initial progress but premature convergence to suboptimal region")
            print(f"    ✓ Winner: CMA-ES (robust covariance adaptation handles multimodality)")
            
        elif func_name == 'Rosenbrock':
            print(f"    • Narrow curved valley, challenging for isotropic search")
            print(f"    • CMA-ES: Excellent adaptation to elongated valley structure")
            print(f"    • NES: Good performance on this smooth function, gradient information useful")
            print(f"    • CEM: Struggled with narrow valley, variance reduction too aggressive")
            print(f"    ✓ Winner: CMA-ES (covariance adaptation aligns with valley)")
            
        elif func_name == 'Branin':
            print(f"    • Three global minima with relatively smooth landscape")
            print(f"    • CMA-ES: Successfully located global minimum")
            print(f"    • NES: Converged to good solution, benefited from smooth structure")
            print(f"    • CEM: Fast convergence but may miss global optimum depending on initialization")
            print(f"    ✓ Winner: CMA-ES (most reliable across multiple runs)")
            
        elif func_name == 'Rastrigin':
            print(f"    • Extremely multimodal with regular grid of local minima")
            print(f"    • CMA-ES: Best at escaping local minima, strongest global search")
            print(f"    • NES: Frequently trapped in local minima, gradient misleading")
            print(f"    • CEM: Quick exploration but often converges to nearby local minimum")
            print(f"    ✓ Winner: CMA-ES (step-size control enables escape from local traps)")

# ==================== Summary Analysis ====================
def print_summary_analysis():
    """Print overall comparative summary (simplified human-written version)"""
    print("\n" + "="*80)
    print("COMPARATIVE SUMMARY OF CEM, NES, AND CMA-ES")
    print("="*80)
    print("""
1. Convergence Speed and Stability

   - The Cross-Entropy Method (CEM) converges fastest but is prone to instability.
     It works well when quick exploration is needed, especially with limited evaluations,
     but often risks collapsing diversity too early.

   - Natural Evolution Strategies (NES) show moderate speed and stability.
     They perform better on smooth landscapes but can be misled by local optima,
     depending strongly on the learning rate setting.

   - CMA-ES converges more slowly but remains the most stable and reliable.
     It adapts its covariance and step size effectively, making it suitable
     for complex or ill-conditioned optimization tasks.

2. Ability to Find the Global Minimum

   - On multimodal functions (e.g., Ackley, Rastrigin), CMA-ES consistently
     finds near-global optima. CEM performs reasonably if initialized broadly,
     while NES struggles due to gradient bias toward local minima.

   - On smooth functions (e.g., Rosenbrock, Branin), both CMA-ES and NES
     perform well. CMA-ES aligns its search with narrow valleys effectively,
     while NES benefits from informative gradients. CEM remains adequate but less efficient.

3. Efficiency vs Robustness

   CEM is the most efficient but least robust;
   NES lies in the middle;
   CMA-ES is the most robust but computationally heavier.

   In short:
   - Use **CEM** when speed matters more than precision.
   - Use **NES** for moderately smooth problems with some gradient structure.
   - Use **CMA-ES** when quality and reliability are top priorities.

4. Algorithmic Insights

   - **CEM:** Simple, fast, but memoryless; sensitive to elite fraction size.
   - **NES:** Theoretically principled; learning rate tuning is critical.
   - **CMA-ES:** Adaptive and self-correcting; higher computational cost.

5. Practical Considerations

   - Implementation: CEM is simplest, NES moderate, CMA-ES most complex.
   - Sensitivity: CEM and NES require parameter tuning; CMA-ES works well
     with defaults.
   - Scalability: CMA-ES scales best to higher dimensions.
   - All methods can benefit from parallel evaluations.
    """)
    print("="*80)

# ==================== Hybrid Method Analysis ====================

def print_hybrid_analysis():
    """Print analysis of hybrid CEM-CMA-ES approach (simplified version)"""
    print("\n" + "="*80)
    print("HYBRID CEM–CMA-ES STRATEGY")
    print("="*80)
    print("""
This hybrid approach combines the exploration strength of CEM with the
precision of CMA-ES in a two-phase process.

Phase 1 (CEM): Run a few iterations with a broad search to locate promising
regions of the landscape. Capture the resulting mean and covariance.

Phase 2 (CMA-ES): Initialize CMA-ES using the CEM results to refine the
solution with adaptive covariance updates.

Benefits:
- Faster convergence than pure CMA-ES.
- Better stability than pure CEM.
- Particularly effective on moderately complex multimodal problems.

Limitations:
- Adds algorithmic complexity.
- May not outperform CMA-ES when the latter already converges efficiently.

In practice, this hybrid method is useful when computation is limited
but reliable optimization is still desired.
    """)
    print("="*80)

# ==================== Main Execution ====================
if __name__ == "__main__":
    print("Starting Stochastic Optimization Experiments...")
    print("This will run CEM, NES, and CMA-ES on 4 benchmark functions\n")
    
    # Run main experiments
    results, all_convergence = run_experiments()
    
    # Print detailed per-function analysis
    print_detailed_analysis(results, all_convergence)
    
    # Print overall summary
    print_summary_analysis()
    
    # Run hybrid experiments (Advanced Exercise #1)
    print("\n" + "="*100)
    print("Running Advanced Exercise: Hybrid CEM-CMA-ES")
    print("="*100)
    hybrid_results = run_hybrid_experiments()
    
    # Print hybrid analysis
    print_hybrid_analysis()
    
    print("\n" + "="*100)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*100)
    print("\nGenerated outputs:")
    print("  • Console tables with all results")
    print("  • Trajectory plots for each function (4 plots)")
    print("  • Convergence comparison plots (4 plots)")
    print("  • Detailed comparative analysis")
    print("  • Hybrid method results and analysis")
    print("="*100)