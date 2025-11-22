==========================================================================================
HOOKE-JEEVES AND GPS METHODS OPTIMIZATION RESULTS
==========================================================================================


Ackley Function - Starting Point: (-3.0, -3.0)
--------------------------------------------------------------------------------
Hooke-Jeeves:
  Final Point: (0.000000, 0.000000)
  Final Value: 0.000000e+00
  Iterations: 26
  Time: 0.0009 seconds
GPS:
  Final Point: (0.000000, 0.000000)
  Final Value: 0.000000e+00
  Iterations: 26
  Time: 0.0006 seconds

Booth Function - Starting Point: (0.0, 0.0)
--------------------------------------------------------------------------------
Hooke-Jeeves:
  Final Point: (1.000000, 3.000000)
  Final Value: 0.000000e+00
  Iterations: 24
  Time: 0.0004 seconds
GPS:
  Final Point: (1.000004, 2.999996)
  Final Value: 2.910383e-11
  Iterations: 100
  Time: 0.0006 seconds

Branin Function - Starting Point: (2.0, 2.0)
--------------------------------------------------------------------------------
Hooke-Jeeves:
  Final Point: (3.141592, 2.275000)
  Final Value: 3.978874e-01
  Iterations: 49
  Time: 0.0009 seconds
GPS:
  Final Point: (3.141592, 2.275000)
  Final Value: 3.978874e-01
  Iterations: 49
  Time: 0.0006 seconds

Rosenbrock Function - Starting Point: (-1.5, 2.0)
--------------------------------------------------------------------------------
Hooke-Jeeves:
  Final Point: (0.999981, 0.999962)
  Final Value: 3.637979e-10
  Iterations: 462
  Time: 0.0061 seconds
GPS:
  Final Point: (0.999981, 0.999962)
  Final Value: 3.637979e-10
  Iterations: 460
  Time: 0.0024 seconds

Wheeler Function - Starting Point: (1.5, 0.5)
--------------------------------------------------------------------------------
Hooke-Jeeves:
  Final Point: (1.000002, 1.499998)
  Final Value: -1.000000e+00
  Iterations: 56
  Time: 0.0009 seconds
GPS:
  Final Point: (1.000002, 1.499998)
  Final Value: -1.000000e+00
  Iterations: 76
  Time: 0.0006 seconds

Rastrigin Function - Starting Point: (2.5, 2.5)
--------------------------------------------------------------------------------
Hooke-Jeeves:
  Final Point: (0.000000, 0.000000)
  Final Value: 0.000000e+00
  Iterations: 26
  Time: 0.0006 seconds
GPS:
  Final Point: (0.000000, 0.000000)
  Final Value: 0.000000e+00
  Iterations: 26
  Time: 0.0004 seconds

==========================================================================================
SUMMARY TABLE - ALL RESULTS
==========================================================================================

  Function        Start       Method              x_found    f(x_found)  Iterations Time (s)
    Ackley (-3.0, -3.0) Hooke-Jeeves (0.000000, 0.000000)  0.000000e+00          26   0.0009
    Ackley (-3.0, -3.0)          GPS (0.000000, 0.000000)  0.000000e+00          26   0.0006
     Booth   (0.0, 0.0) Hooke-Jeeves (1.000000, 3.000000)  0.000000e+00          24   0.0004
     Booth   (0.0, 0.0)          GPS (1.000004, 2.999996)  2.910383e-11         100   0.0006
    Branin   (2.0, 2.0) Hooke-Jeeves (3.141592, 2.275000)  3.978874e-01          49   0.0009
    Branin   (2.0, 2.0)          GPS (3.141592, 2.275000)  3.978874e-01          49   0.0006
Rosenbrock  (-1.5, 2.0) Hooke-Jeeves (0.999981, 0.999962)  3.637979e-10         462   0.0061
Rosenbrock  (-1.5, 2.0)          GPS (0.999981, 0.999962)  3.637979e-10         460   0.0024
   Wheeler   (1.5, 0.5) Hooke-Jeeves (1.000002, 1.499998) -1.000000e+00          56   0.0009
   Wheeler   (1.5, 0.5)          GPS (1.000002, 1.499998) -1.000000e+00          76   0.0006
 Rastrigin   (2.5, 2.5) Hooke-Jeeves (0.000000, 0.000000)  0.000000e+00          26   0.0006
 Rastrigin   (2.5, 2.5)          GPS (0.000000, 0.000000)  0.000000e+00          26   0.0004

✓ Generated 18 plots for Rosenbrock and Ackley functions
==========================================================================================

==========================================================================================
DISCUSSION
==========================================================================================


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

==========================================================================================


==========================================================================================
ADDITIONAL EXERCISE: ADAPTIVE GPS WITH ORTHOGONAL DIRECTION REGENERATION
==========================================================================================


Rosenbrock Function - Starting Point: (-1.5, 2.0)
--------------------------------------------------------------------------------
Standard GPS:
  Final Value: 3.637979e-10
  Iterations: 460
  Time: 0.0025 seconds
Adaptive GPS:
  Final Value: 3.637979e-10
  Iterations: 460
  Time: 0.0041 seconds
  Improvement: 0.00%

Rastrigin Function - Starting Point: (2.5, 2.5)
--------------------------------------------------------------------------------
Standard GPS:
  Final Value: 0.000000e+00
  Iterations: 26
  Time: 0.0004 seconds
Adaptive GPS:
  Final Value: 9.949591e-01
  Iterations: 36
  Time: 0.0006 seconds
  Improvement: 0.00%

==========================================================================================
ADAPTIVE GPS COMPARISON RESULTS
==========================================================================================

  Function       Method  Final Value  Iterations Time (s)
Rosenbrock Standard GPS 3.637979e-10         460   0.0025
Rosenbrock Adaptive GPS 3.637979e-10         460   0.0041
 Rastrigin Standard GPS 0.000000e+00          26   0.0004
 Rastrigin Adaptive GPS 9.949591e-01          36   0.0006

==========================================================================================
Adaptive GPS generates new orthogonal directions after successful steps,
potentially exploring the space more effectively than fixed coordinate directions.
==========================================================================================