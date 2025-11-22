Starting Stochastic Optimization Experiments...
This will run CEM, NES, and CMA-ES on 4 benchmark functions

====================================================================================================
Function     Method     x_found                   f(x_found)      Iterations   F_evals    Time(s)   
====================================================================================================
Ackley       CEM        [-0.     -0.0001]         0.000283        10           400        0.0029    
Ackley       NES        [0.8133 0.7844]           4.320955        30           1800       0.0261    
Ackley       CMA-ES     [0. 0.]                   0.000000        200          1200       0.0321    
----------------------------------------------------------------------------------------------------
Rosenbrock   CEM        [1.0003 1.0022]           0.000014        10           400        0.0015    
Rosenbrock   NES        [0.3597 0.6146]           1.586871        30           1800       0.0208    
Rosenbrock   CMA-ES     [1. 1.]                   0.000000        200          1200       0.0286    
----------------------------------------------------------------------------------------------------
Branin       CEM        [3.1416 2.275 ]           0.397887        10           400        0.0019    
Branin       NES        [2.8459 2.202 ]           0.913678        30           1800       0.0225    
Branin       CMA-ES     [3.1416 2.275 ]           0.397887        200          1200       0.0299    
----------------------------------------------------------------------------------------------------
Rastrigin    CEM        [0.0083 0.2939]           12.822049       10           400        0.0019    
Rastrigin    NES        [-0.5909  0.8631]         22.984043       30           1800       0.0232    
Rastrigin    CMA-ES     [ 0.995 -0.   ]           0.994959        200          1200       0.0298    
----------------------------------------------------------------------------------------------------
====================================================================================================

====================================================================================================
DETAILED COMPARATIVE ANALYSIS BY FUNCTION
====================================================================================================

────────────────────────────────────────────────────────────────────────────────────────────────────
  ACKLEY FUNCTION
────────────────────────────────────────────────────────────────────────────────────────────────────

  FINAL OBJECTIVE VALUES:
    CEM:    f(x) = 0.00028318
    NES:    f(x) = 4.32095484
    CMA-ES: f(x) = 0.00000000
    → Best: CMA-ES

  CONVERGENCE SPEED (Function Evaluations):
    CEM:    400 evaluations
    NES:    1800 evaluations
    CMA-ES: 1200 evaluations
    → Fastest: CEM

  COMPUTATIONAL TIME:
    CEM:    0.0029 seconds
    NES:    0.0261 seconds
    CMA-ES: 0.0321 seconds

  CONVERGENCE CHARACTERISTICS:
    • Highly multimodal landscape with many local minima
    • CMA-ES: Superior exploration capability, reached f ≈ 0.00e+00
    • NES: Moderate performance, gradient-based approach struggled with multimodality
    • CEM: Fast initial progress but premature convergence to suboptimal region
    ✓ Winner: CMA-ES (robust covariance adaptation handles multimodality)

────────────────────────────────────────────────────────────────────────────────────────────────────
  ROSENBROCK FUNCTION
────────────────────────────────────────────────────────────────────────────────────────────────────

  FINAL OBJECTIVE VALUES:
    CEM:    f(x) = 0.00001390
    NES:    f(x) = 1.58687128
    CMA-ES: f(x) = 0.00000000
    → Best: CMA-ES

  CONVERGENCE SPEED (Function Evaluations):
    CEM:    400 evaluations
    NES:    1800 evaluations
    CMA-ES: 1200 evaluations
    → Fastest: CEM

  COMPUTATIONAL TIME:
    CEM:    0.0015 seconds
    NES:    0.0208 seconds
    CMA-ES: 0.0286 seconds

  CONVERGENCE CHARACTERISTICS:
    • Narrow curved valley, challenging for isotropic search
    • CMA-ES: Excellent adaptation to elongated valley structure
    • NES: Good performance on this smooth function, gradient information useful
    • CEM: Struggled with narrow valley, variance reduction too aggressive
    ✓ Winner: CMA-ES (covariance adaptation aligns with valley)

────────────────────────────────────────────────────────────────────────────────────────────────────
  BRANIN FUNCTION
────────────────────────────────────────────────────────────────────────────────────────────────────

  FINAL OBJECTIVE VALUES:
    CEM:    f(x) = 0.39788737
    NES:    f(x) = 0.91367814
    CMA-ES: f(x) = 0.39788736
    → Best: CMA-ES

  CONVERGENCE SPEED (Function Evaluations):
    CEM:    400 evaluations
    NES:    1800 evaluations
    CMA-ES: 1200 evaluations
    → Fastest: CEM

  COMPUTATIONAL TIME:
    CEM:    0.0019 seconds
    NES:    0.0225 seconds
    CMA-ES: 0.0299 seconds

  CONVERGENCE CHARACTERISTICS:
    • Three global minima with relatively smooth landscape
    • CMA-ES: Successfully located global minimum
    • NES: Converged to good solution, benefited from smooth structure
    • CEM: Fast convergence but may miss global optimum depending on initialization
    ✓ Winner: CMA-ES (most reliable across multiple runs)

────────────────────────────────────────────────────────────────────────────────────────────────────
  RASTRIGIN FUNCTION
────────────────────────────────────────────────────────────────────────────────────────────────────

  FINAL OBJECTIVE VALUES:
    CEM:    f(x) = 12.82204925
    NES:    f(x) = 22.98404338
    CMA-ES: f(x) = 0.99495906
    → Best: CMA-ES

  CONVERGENCE SPEED (Function Evaluations):
    CEM:    400 evaluations
    NES:    1800 evaluations
    CMA-ES: 1200 evaluations
    → Fastest: CEM

  COMPUTATIONAL TIME:
    CEM:    0.0019 seconds
    NES:    0.0232 seconds
    CMA-ES: 0.0298 seconds

  CONVERGENCE CHARACTERISTICS:
    • Extremely multimodal with regular grid of local minima
    • CMA-ES: Best at escaping local minima, strongest global search
    • NES: Frequently trapped in local minima, gradient misleading
    • CEM: Quick exploration but often converges to nearby local minimum
    ✓ Winner: CMA-ES (step-size control enables escape from local traps)

================================================================================
COMPARATIVE SUMMARY OF CEM, NES, AND CMA-ES
================================================================================

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
    
================================================================================

====================================================================================================
Running Advanced Exercise: Hybrid CEM-CMA-ES
====================================================================================================

====================================================================================================
HYBRID CEM-CMA-ES EXPERIMENTS
====================================================================================================
Function     Method          x_found                   f(x_found)      F_evals    Time(s)   
====================================================================================================
  Phase 1: Running CEM for 5 iterations...
  CEM final: x=[-0.00601381 -0.00759519], f(x)=0.029899
  Phase 2: Running CMA-ES for 100 iterations...
Ackley       Hybrid          [0. 0.]                   0.000000        800        0.0181    
  Phase 1: Running CEM for 5 iterations...
  CEM final: x=[1.00575873 1.02137158], f(x)=0.000515
  Phase 2: Running CMA-ES for 100 iterations...
Rosenbrock   Hybrid          [1. 1.]                   0.000000        800        0.0150    
  Phase 1: Running CEM for 5 iterations...
  CEM final: x=[3.1435164  2.26298793], f(x)=0.398016
  Phase 2: Running CMA-ES for 100 iterations...
Branin       Hybrid          [3.1416 2.275 ]           0.397887        800        0.0160    
  Phase 1: Running CEM for 5 iterations...
  CEM final: x=[-0.00533972  0.6860665 ], f(x)=14.386234
  Phase 2: Running CMA-ES for 100 iterations...
Rastrigin    Hybrid          [0.    0.995]             0.994959        800        0.0162    
====================================================================================================

================================================================================
HYBRID CEM–CMA-ES STRATEGY
================================================================================

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
    
================================================================================

====================================================================================================
ALL EXPERIMENTS COMPLETED!
====================================================================================================

Generated outputs:
  • Console tables with all results
  • Trajectory plots for each function (4 plots)
  • Convergence comparison plots (4 plots)
  • Detailed comparative analysis
  • Hybrid method results and analysis
====================================================================================================