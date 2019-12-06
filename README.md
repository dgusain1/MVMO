## Mean Variance Mapping Optimization Algorithm
MVMO is a Python package to perform heuristic optimization on constrained and unconsrained optimization problems whose convexity and/or linearity may not be fully known. It is based on swarm optimization principles, and uses a continuously updated mean and variance of best solutions from optimization process. Note: since this is a heuristic algorithm, it does not provide the optimal solution, but near optimal solution. This is however done in a very quick time compared to traditional optimization solvers.

## Installation
MVMO can be installed from PyPI using:

```
pip install MVMO
```
MVMO requires numpy and pandas to work.

## Usage

### Initialisation
The MVMO optimizer can be called with arguments *iterations*, *num_mutation*, and *population_size*. This defines key parameters for MVMO.

### Defining objective function
MVMO by default optimizes the objective function for minimum. For maximisation, the objective function will need to be modified. The MVMO package provides the following test function benchmarks from [Wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization#cite_note-11):
1. Rosenbrock
2. Matyas
3. Booth
4. Himm
5. Easom

### Constraint definition
Constraints can be inequality or equality. The constraints are passed as a dictionary object with keywords `ineq`, `eq`, or `func` to symbolise whether the specified constraint is inequality, equality or a function definition. Inequality and equality contraints are specified in `str` format, and must follow the convention:

```math
g(x) <= 0   #inequality constraint
h(x) = 0    #equality constraint
```
Complex constraints can also be defined as python functions. An example of how to use the MVMO package for constrained optimization is shown later. It uses all three constraint defintions. 

### Optimization
The `optimize()` method can be called on the `optimizer` to perform optimization. It returns the following upon completion of optimization:
1. `res`: Provides best objective function value, and where it was obtained. [obf_value, x]
2. `conv`: Provides the list of objective function values over optimization process. This can beused to plot convergence graph.
3. `sol`: provides the internal mean and variance of stored solutions that was used for optimization. The size of this matrix is **x** X **num_population**.

The convergence graph can be plotted with `MVMO.plot(conv)`.

The following example shows minimization of constrained [Rosenbrock](https://en.wikipedia.org/wiki/Test_functions_for_optimization#cite_note-11) function:

```python
from MVMO import MVMO
from MVMO import test_functions
function = test_functions.rosen
optimizer = MVMO(iterations=5000, num_mutation=3, population_size=10)

def func_constr(X):
	return True if X[0]**2 + X[1]**2 < 1 else False
	
bds = [(-1.5,1.5), (-0.5,2.5)]
constr = {'ineq':"(X[0] - 1)**3 - X[1] + 1",
		  'eq':"X[0]+X[1]-2",
		  'func':func_constr}
res, conv, sol = optimizer.optimize(obj_fun=function, bounds=bds, constraints=constr)

MVMO.plot(conv)
```

