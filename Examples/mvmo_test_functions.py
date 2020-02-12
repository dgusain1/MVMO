# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 12:36:20 2019

@author: digvijaygusain
"""

from MVMO import MVMO
from MVMO import test_functions
function_rosen = test_functions.rosen
function_easom = test_functions.easom

optimizer = MVMO(iterations=5000, num_mutation=3, population_size=10)

def func_constr(X):
	return True if X[0]**2 + X[1]**2 < 1 else False
	
bds = [(-1.5,1.5), (-0.5,2.5)]
constr = {'ineq':"(X[0] - 1)**3 - X[1] + 1",
		  'eq':"X[0]+X[1]-2",
		  'func':func_constr}
res, conv, sol = optimizer.optimize(obj_fun=function_rosen, bounds=bds, constraints=constr)

print(f'Rosenbrock optimum value = {res[0]} at X = {res[1]}')

#res, conv, sol = optimizer.optimize(obj_fun=function_easom, bounds=[(-100,100)*2])
#print(f'eASOM optimum value = {res[0]} at X = {res[1]}')