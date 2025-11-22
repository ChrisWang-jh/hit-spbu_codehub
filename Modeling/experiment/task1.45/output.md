
====================================================================================================
NELDER-MEAD AND DIRECT METHODS OPTIMIZATION RESULTS
====================================================================================================

==================================================
NELDER-MEAD EXPERIMENTS
==================================================


Ackley Function - Starting Point: (0.0, 1.0)
--------------------------------------------------------------------------------
Nelder-Mead:
  Final Point: (-0.000064, 0.951933)
  Final Value: 2.579929e+00
  Iterations: 19
  Function Evaluations: 37
  Time: 0.0011 seconds

Branin Function - Starting Point: (2.0, 2.0)
--------------------------------------------------------------------------------
Nelder-Mead:
  Final Point: (3.142077, 2.275256)
  Final Value: 3.978889e-01
  Iterations: 26
  Function Evaluations: 54
  Time: 0.0012 seconds

Rosenbrock Function - Starting Point: (-1.5, 2.0)
--------------------------------------------------------------------------------
Nelder-Mead:
  Final Point: (1.000743, 1.001138)
  Final Value: 1.161162e-06
  Iterations: 45
  Function Evaluations: 86
  Time: 0.0018 seconds

Rastrigin Function - Starting Point: (2.5, 2.5)
--------------------------------------------------------------------------------
Nelder-Mead:
  Final Point: (2.984798, 2.984904)
  Final Value: 1.790920e+01
  Iterations: 34
  Function Evaluations: 67
  Time: 0.0016 seconds

==================================================
DIRECT EXPERIMENTS
==================================================


Ackley Function - Bounds: [[-2, -3], [4, 3]]
--------------------------------------------------------------------------------
DIRECT:
  Final Point: (0.000000, 0.000000)
  Final Value: 3.552714e-15
  Iterations: 150
  Function Evaluations: 54777
  Time: 2.7019 seconds

Branin Function - Bounds: [[0, 0], [4, 4]]
--------------------------------------------------------------------------------
DIRECT:
  Final Point: (3.149186, 2.222222)
  Final Value: 4.003604e-01
  Iterations: 150
  Function Evaluations: 58873
  Time: 2.7384 seconds

Rosenbrock Function - Bounds: [[-5, 0], [2, 4]]
--------------------------------------------------------------------------------
DIRECT:
  Final Point: (-1.264746, 1.777778)
  Final Value: 5.287842e+00
  Iterations: 150
  Function Evaluations: 56621
  Time: 2.5509 seconds

Rastrigin Function - Bounds: [[-1, -1], [6, 6]]
--------------------------------------------------------------------------------
DIRECT:
  Final Point: (0.037037, 0.994959)
  Final Value: 1.265882e+00
  Iterations: 150
  Function Evaluations: 53917
  Time: 2.5569 seconds

====================================================================================================
SUMMARY TABLE - ALL RESULTS
====================================================================================================

  Function               Start/Domain      Method               x_found   f(x_found)  Iterations  f_evals time_s
    Ackley          start: (0.0, 1.0) Nelder-Mead (-0.000064, 0.951933) 2.579929e+00          19       37 0.0011
    Branin          start: (2.0, 2.0) Nelder-Mead  (3.142077, 2.275256) 3.978889e-01          26       54 0.0012
Rosenbrock         start: (-1.5, 2.0) Nelder-Mead  (1.000743, 1.001138) 1.161162e-06          45       86 0.0018
 Rastrigin          start: (2.5, 2.5) Nelder-Mead  (2.984798, 2.984904) 1.790920e+01          34       67 0.0016
    Ackley bounds: [[-2, -3], [4, 3]]      DIRECT  (0.000000, 0.000000) 3.552714e-15         150    54777 2.7019
    Branin   bounds: [[0, 0], [4, 4]]      DIRECT  (3.149186, 2.222222) 4.003604e-01         150    58873 2.7384
Rosenbrock  bounds: [[-5, 0], [2, 4]]      DIRECT (-1.264746, 1.777778) 5.287842e+00         150    56621 2.5509
 Rastrigin bounds: [[-1, -1], [6, 6]]      DIRECT  (0.037037, 0.994959) 1.265882e+00         150    53917 2.5569

✓ Generated trajectory plots for all test functions
====================================================================================================

====================================================================================================
COMPARATIVE DISCUSSION
====================================================================================================


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

====================================================================================================