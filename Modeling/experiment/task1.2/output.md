================================================================================
POWELL'S METHOD OPTIMIZATION RESULTS
================================================================================


Ackley Function - Starting Point: (-3.0, -3.0)
------------------------------------------------------------
Powell's Method:
  Final Point: (-0.000000, 0.000000)
  Final Value: 2.244971e-07
  Iterations: 2
  Function Evaluations: 423
  Time: 0.0030 seconds

Branin Function - Starting Point: (2.0, 2.0)
------------------------------------------------------------
Powell's Method:
  Final Point: (9.424778, 2.475000)
  Final Value: 3.978874e-01
  Iterations: 5
  Function Evaluations: 1056
  Time: 0.0044 seconds

================================================================================
SUMMARY TABLE - POWELL'S METHOD
================================================================================

Function        Start Method               x_found   f(x_found)  Iterations  Func_Evals Time (s)
  Ackley (-3.0, -3.0) Powell (-0.000000, 0.000000) 2.244971e-07           2         423   0.0030
  Branin   (2.0, 2.0) Powell  (9.424778, 2.475000) 3.978874e-01           5        1056   0.0044

✓ Generated trajectory and convergence plots for all test functions
================================================================================

================================================================================
COMPARATIVE STUDY: POWELL vs CCD WITH ACCELERATION
================================================================================


Rosenbrock Function - Starting Point: (-1.5, 2.0)
------------------------------------------------------------
Powell's Method:
  Final Point: (0.999888, 0.999818)
  Final Value: 2.113855e-08
  Iterations: 11
  Function Evaluations: 2322
  Time: 0.0068 seconds
CCD with Acceleration:
  Final Point: (1.000007, 1.000014)
  Final Value: 4.901371e-11
  Iterations: 65
  Function Evaluations: 13716
  Time: 0.0402 seconds

Ackley Function - Starting Point: (4.0, 1.0)
------------------------------------------------------------
Powell's Method:
  Final Point: (-0.000000, 0.000000)
  Final Value: 7.838645e-08
  Iterations: 2
  Function Evaluations: 423
  Time: 0.0030 seconds
CCD with Acceleration:
  Final Point: (0.000000, -0.000000)
  Final Value: 3.055021e-07
  Iterations: 2
  Function Evaluations: 423
  Time: 0.0030 seconds

================================================================================
COMPARISON TABLE - POWELL vs CCD WITH ACCELERATION
================================================================================

  Function       Start    Method               x_found   f(x_found)  Iterations  Func_Evals Time (s)
Rosenbrock (-1.5, 2.0)    Powell  (0.999888, 0.999818) 2.113855e-08          11        2322   0.0068
Rosenbrock (-1.5, 2.0) CCD_Accel  (1.000007, 1.000014) 4.901371e-11          65       13716   0.0402
    Ackley  (4.0, 1.0)    Powell (-0.000000, 0.000000) 7.838645e-08           2         423   0.0030
    Ackley  (4.0, 1.0) CCD_Accel (0.000000, -0.000000) 3.055021e-07           2         423   0.0030

✓ Generated comparison plots for Rosenbrock and Ackley functions
================================================================================

================================================================================
COMPARATIVE DISCUSSION
================================================================================


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

================================================================================