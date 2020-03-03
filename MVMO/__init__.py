""" Heuristic Optimization Algorithm developed in Python by Digvijay Gusain.
    Usage: (Optimizing the Rosenbrock function constrained with a cubic and a line [Wikipedia])
    
        from MVMO import MVMO
        from MVMO import test_functions
        function = test_functions.rosen
        optimizer = MVMO(iterations=5000, num_mutation=3, population_size=10)
        
        def constraint(X):
            return True if X[0]**2 + X[1]**2 < 1 else False
            
        bds = [(-1.5,1.5), (-0.5,2.5)]
        constr = {'ineq':"(X[0] - 1)**3 - X[1] + 1",
                  'eq':"X[0]+X[1]-2",
                  'func':constraint}
        res, conv, sol = optimizer.optimize(obj_fun=function, bounds=bds, constraints=constr)
        MVMO.plot(conv)
    """
import numpy as np, pandas as pd
import time, sys
from tqdm import tqdm
from pyDOE import lhs
__version__ = "1.0.9"

this is crazy af 
class MVMO():
    
    def __init__(self, iterations=1000, num_mutation=1, population_size=5, logger=True, stop_iter_no_progresss = False, eps = 1e-4):
        #num_mutation can be variable.
        self.iterations = iterations
        self.num_mutation = num_mutation
        self.population_size = population_size
        self.logger = logger
        self.stop_if_no_progress = stop_iter_no_progresss
        self.eps = eps

    def mvmo(self, obj_fun, bounds, cons, x_initial):
        
        convergence = []
        min_b, max_b = np.asarray(bounds).T
        diff = max_b - min_b#np.fabs(min_b - max_b)

        # problem dimension
        D = len(bounds)

        # create storage df
        solutions_d = []
        metrics_d = {}

        if x_initial:
            x0 = (np.asarray(x_initial) - min_b) / diff
            
        else:
            # generate initial random solution
            x0 = lhs(1,D).T[0] #np.random.uniform(size=D)
        
        # denormalise solution
        x0_denorm = min_b + x0 * diff
        # evaluate initial fitness
        a = obj_fun(x0_denorm.tolist())
        #check if contraints are met
        sol_good = self.constraint_check(x0_denorm.tolist(), cons)
        
        if sol_good:            
            fitness = round(a, 4)
        else:
            fitness = 1e10 #penalty for constraint violation
            
        convergence.append(fitness)
    
            # fill the fitness dataframe with fitness value, solution vector, mean, shape, and d-factor
    
        solutions_d.append((fitness,tuple(x0.tolist())))
        # initial metric is set to 0.5 for mean

        scaling_factor_hist = []
        # TODO: How to define initial scaling factors???
        for i in tqdm(range(self.iterations),disable=False):
            #check for exit
            if self.stop_if_no_progress and i > 500 and np.var(convergence[-500:]) < self.eps:
                print(f"Exiting at iteration {i} because optimizer couldn't improve solutions any longer.")
                break
            
            # parent
            solutions_d.sort()
            x_parent = np.asarray(list(solutions_d[0][1]))
            num_mut = D if i < self.population_size+10 else self.num_mutation
            idxs = np.random.choice(
                list(range(D)), num_mut, replace=False)
            rand_mean = lhs(1,1)[0][0]
            for idx in idxs:
                
                # mean
                if len(solutions_d)>self.population_size+1:
                    x_bar = metrics_d['mean'][idx]
                    var = metrics_d['variance'][idx]
                else:
                    x_bar, var = rand_mean, 0.5
                    
                
                xi_star = lhs(1,1)[0][0]
                
                #scaling factor can be variable. This affects converegnce so play with it. 
                # maybe increase quadratically or someway with number of iterations.
                #when no improvement in solutions is observed, change it back to one for more explorstion\
                scaling_factor = 1 + (i+1) 
                
                if i > 500 and np.var(convergence[-500:]) < 1e-5:
                    scaling_factor = 2
                
                s_old = -np.log(var) * scaling_factor
                
                #this 0.5 can also be adaptive
                if xi_star < 0.5:
                    s_new = s_old/(1 - x_bar)
                    hm = x_bar - x_bar/(0.5*s_new + 1)
                    hf = x_bar * (1 - np.exp(-xi_star * s_new))
                    hc = (x_bar - hm) * 5 * xi_star
                    xi_new = hf + hc
                else:
                    s_new = s_old/x_bar
                    hm = (1 - x_bar)/(0.5*s_new + 1)
                    hb = (1 - x_bar) / ((1 - xi_star) * s_new + 1) + x_bar
                    hc = hm * 5 * (1 - xi_star)
                    xi_new = hb - hc
                    
                x_parent[idx] = xi_new
            
            scaling_factor_hist.append(scaling_factor)
            x_denorm = min_b + x_parent * diff
             
            tmp=x_denorm.tolist()
            
            a = obj_fun(tmp)
            
            sol_good = self.constraint_check(x_denorm, cons)
            
            if sol_good:                
                fitness = round(a, 4)
            else:
                fitness = 1e10 #penalty for constraint violation
            
            
            if len(solutions_d) < self.population_size+1:
                solutions_d.append((fitness,tuple(x_parent.tolist())))
                solutions_d.sort()
                convergence.append(solutions_d[0][0])
                sol_d_tmp = pd.DataFrame.from_dict(dict(solutions_d),orient='columns')
                    
                metrics_d['variance'] = [
                    np.var(sol_d_tmp.iloc[x, :]) for x in range(len(sol_d_tmp))]
                metrics_d['mean'] = [
                    np.mean(sol_d_tmp.iloc[x, :]) for x in range(len(sol_d_tmp))]
                
            else:
                solutions_d.sort()
                max_value=solutions_d[-1][0]
                if fitness < max_value:
                    solutions_d.append((fitness,tuple(x_parent.tolist())))
                    solutions_d.sort()
                    solutions_d.pop(-1)
                    convergence.append(solutions_d[0][0])
                    sol_d_tmp = pd.DataFrame.from_dict(dict(solutions_d),orient='columns')
                    
                    metrics_d['variance'] = [
                    np.var(sol_d_tmp.iloc[x, :]) for x in range(len(sol_d_tmp))]
                    metrics_d['mean'] = [
                        np.mean(sol_d_tmp.iloc[x, :]) for x in range(len(sol_d_tmp))]
                else:
                    convergence.append(convergence[-1])
            
        
        solutions_d.sort()
        res = min_b + \
            np.asarray(list(solutions_d[0][1])) * diff
        res = [round(x,5) for x in res]
        final_of = obj_fun(res)
        return [final_of, res], convergence, pd.DataFrame.from_dict(dict(solutions_d),orient='index'), {'metrics':metrics_d, 'scaling_factors':scaling_factor_hist}
    
    def constraint_check(self, solution, constraints):
        if len(constraints) == 0:
            return True
        else:
            X = solution
            for key, value in constraints.items():
                if key != 'func':
                    v = eval(value)
                    if key == 'ineq' and v >= 0:
                        return False
                    elif key == 'eq' and v != 0:
                        return False
                    else:
                        return True
                else:
                    return value(X)

    
    def optimize(self, obj_fun, bounds, constraints={}, x0=False):
        t1 = time.time()
        self.res, self.conv, self.sol, self.extras = self.mvmo(
            obj_fun=obj_fun, bounds=bounds, cons=constraints, x_initial=x0)
        t2 = time.time()
        if self.logger:
            sep = '*' * 50
            print("\n")
            print(sep)
            print(f"Optimal Solution found in {round(t2-t1, 2)}s")
            print(sep)
            print(f"\nFinal Objective Function Value: {self.res[0]}.")
            print(f"Optimal Objective Function Value at: {self.res[1]}.")

        return self.res, self.conv, self.sol, self.extras

    def plot(conv):
        import matplotlib.pyplot as plt
        plt.plot(conv, "C2", linewidth=1)
        plt.ylabel("Objective Function Fitness")
        plt.xlabel("Iterations")
        plt.title("Convergence Plot")
        plt.legend()
        plt.tight_layout()
        plt.show()
        

class test_functions():    
    def rosen(X):    
        x = X[0]
        y = X[1]
        a = 1. - x
        b = y - x*x
        return a*a + b*b*100.
    
    def obf(x):
        return sum(np.asarray(x)**2)+2
    
    def booth(x):
        return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2
    
    def matyas(x):
        return 0.26*(x[0]**2 + x[1]**2) - 0.48*x[0]*x[1]
    
    def himm(x):
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
    
    def easom(x):
        return -np.cos(x[0])*np.cos(x[1])*np.exp(-1*((x[0]-np.pi)**2 + (x[1]-np.pi)**2))