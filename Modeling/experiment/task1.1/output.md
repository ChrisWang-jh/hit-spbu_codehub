THE FOLLOWING ARE THE OUTPUT OF task1.py

================================================================================
CYCLIC COORDINATE DESCENT OPTIMIZATION RESULTS
================================================================================


Booth Function - Starting Point: (0.0, 0.0)
------------------------------------------------------------
Basic CCD:
  Final Point: (1.000001, 2.999999)
  Final Value: 4.197070e-12
  Iterations: 33
Accelerated CCD:
  Final Point: (1.000000, 3.000000)
  Final Value: 7.094877e-15
  Iterations: 18

Booth Function - Starting Point: (-3.0, 4.0)
------------------------------------------------------------
Basic CCD:
  Final Point: (0.999999, 3.000001)
  Final Value: 1.732175e-12
  Iterations: 31
Accelerated CCD:
  Final Point: (1.000000, 3.000000)
  Final Value: 7.353268e-14
  Iterations: 15

Rosenbrock Function - Starting Point: (-1.5, 2.0)
------------------------------------------------------------
Basic CCD:
  Final Point: (1.000010, 1.000021)
  Final Value: 1.037385e-10
  Iterations: 205
Accelerated CCD:
  Final Point: (1.000007, 1.000014)
  Final Value: 4.901371e-11
  Iterations: 65

Rosenbrock Function - Starting Point: (2.0, -1.0)
------------------------------------------------------------
Basic CCD:
  Final Point: (0.999989, 0.999978)
  Final Value: 1.191697e-10
  Iterations: 206
Accelerated CCD:
  Final Point: (0.999994, 0.999987)
  Final Value: 3.824529e-11
  Iterations: 58

Pathological Function - Starting Point: (2.0, 2.0)
------------------------------------------------------------
Basic CCD:
  Final Point: (1.159265, 0.862294)
  Final Value: 1.273724e+00
  Iterations: 1000
Accelerated CCD:
  Final Point: (-0.000156, -0.290996)
  Final Value: -9.086218e-01
  Iterations: 1000

Pathological Function - Starting Point: (-2.0, -2.0)
------------------------------------------------------------
Basic CCD:
  Final Point: (-0.581058, -0.869324)
  Final Value: 1.854551e-01
  Iterations: 1000
Accelerated CCD:
  Final Point: (-0.582639, 0.288633)
  Final Value: -5.433306e-01
  Iterations: 1000

================================================================================
SUMMARY TABLE
================================================================================

    Function  Start Point       Algorithm            Final Point   Final Value  Iterations
       Booth   (0.0, 0.0)       Basic CCD   (1.000001, 2.999999)  4.197070e-12          33
       Booth   (0.0, 0.0) Accelerated CCD   (1.000000, 3.000000)  7.094877e-15          18
       Booth  (-3.0, 4.0)       Basic CCD   (0.999999, 3.000001)  1.732175e-12          31
       Booth  (-3.0, 4.0) Accelerated CCD   (1.000000, 3.000000)  7.353268e-14          15
  Rosenbrock  (-1.5, 2.0)       Basic CCD   (1.000010, 1.000021)  1.037385e-10         205
  Rosenbrock  (-1.5, 2.0) Accelerated CCD   (1.000007, 1.000014)  4.901371e-11          65
  Rosenbrock  (2.0, -1.0)       Basic CCD   (0.999989, 0.999978)  1.191697e-10         206
  Rosenbrock  (2.0, -1.0) Accelerated CCD   (0.999994, 0.999987)  3.824529e-11          58
Pathological   (2.0, 2.0)       Basic CCD   (1.159265, 0.862294)  1.273724e+00        1000
Pathological   (2.0, 2.0) Accelerated CCD (-0.000156, -0.290996) -9.086218e-01        1000
Pathological (-2.0, -2.0)       Basic CCD (-0.581058, -0.869324)  1.854551e-01        1000
Pathological (-2.0, -2.0) Accelerated CCD  (-0.582639, 0.288633) -5.433306e-01        1000

✓ Generated 12 contour plots for all test functions
================================================================================

================================================================================
DISCUSSION
================================================================================


1. Convergence to Global Minimum:
   - Both algorithms successfully converge to the global minimum for the Booth function 
     from both starting points, as this is a convex quadratic function.
   - Both algorithms successfully converge to the global minimum for the Rosenbrock function.
   - The Pathological function has multiple local minima, so convergence to the global 
     minimum is not guaranteed and depends heavily on the starting point.

2. Effect of Acceleration Step:
   - The acceleration step significantly reduces the number of iterations needed, 
     especially for functions with elongated valleys (like Rosenbrock).
   - Acceleration provides a direction combining all coordinate movements, allowing 
     the algorithm to move more directly toward the optimum rather than in a 
     zig-zag pattern along coordinate axes.

3. Sensitivity to Starting Point:
   - The Booth and Rosenbrock functions show low sensitivity due to their convex nature.
   - The Pathological function shows high sensitivity. 
     Since it has many local minima, the choice of starting points can affect the outcome.
   - For non-convex functions, the choice of starting point can determine whether 
     the algorithm finds the global or a local minimum.

4. Why Methods Converge Well for Some Functions:
   - CCD works best on separable or nearly-separable functions where coordinate-wise 
     optimization is effective. The Booth function has strong coupling but is convex, 
     so convergence is guaranteed. The Rosenbrock function has a narrow valley aligned 
     diagonally, making pure coordinate descent inefficient - this is where acceleration 
     helps. The Pathological function's multiple local minima and complex landscape make 
     convergence to the global optimum difficult regardless of the method used.

================================================================================