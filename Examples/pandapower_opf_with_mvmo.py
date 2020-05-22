# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 12:36:20 2019

@author: digvijaygusain
"""

import numpy as np, pandas as pd, pandapower as pp, pandapower.networks as pn
from MVMO import MVMO

net = pn.case9()
pp.runopp(net, init='pf')
cost_pp = net.res_cost
cost = net.poly_cost.iloc[:,[2,3,4]]

def function(x):
    net.gen.iloc[:,4] = x
    
    try:
        pp.runpp(net)
        gen = net.res_ext_grid.iloc[:,0].tolist() + net.res_gen.iloc[:,0].tolist()
        return abs(sum(cost.iloc[:,0] + pd.Series(gen)*(cost.iloc[:,1]) + (pd.Series(gen)**2)*(cost.iloc[:,2])))
    except:
        return False

def constr(x):
    #power gen constraints
    def react_power():
        up = net.res_gen.iloc[:,1]>net.gen.loc[:,'max_q_mvar']
        low = net.res_gen.iloc[:,1]<net.gen.loc[:,'min_q_mvar']
        return False if up.any() or low.any() else True
        
    #voltage constraints
    def voltage():
        up = net.res_bus.iloc[:,0]>net.bus.loc[:,'max_vm_pu']
        low = net.res_bus.iloc[:,0]<net.bus.loc[:,'min_vm_pu']
        return False if up.any() or low.any() else True
    
    #line loading
    def line_loading():
        up = net.res_line.loc[:,'loading_percent']>net.line.loc[:,'max_loading_percent']
        return False if up.any() else True
    
    a = np.array([react_power(), voltage(), line_loading()])
    return True if a.all() else False

optimizer = MVMO(iterations=200, num_mutation=2, population_size=5)
    
bds = list(zip(net.gen.min_p_mw.tolist(), net.gen.max_p_mw.tolist()))
constr = {'func':constr}
res, conv, sol, extras = optimizer.optimize(obj_fun=function, bounds=bds, constraints=constr)
#MVMO.plot(conv)
    
#check pandapower solution to MVMO solution
mvmo_cost = res[0]
print('*'*30)
print(f"Pandapower OPF cost = USD {cost_pp}.")
print(f"MVMO OPF cost = USD {mvmo_cost}.")
print(f"Difference OPF cost = USD {cost_pp-mvmo_cost}")
