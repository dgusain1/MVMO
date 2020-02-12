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
import time
from tqdm import tqdm
__version__ = "1.0.5"


class MVMO():
    
    def __init__(self, iterations=1000, num_mutation=1, population_size=5, logger=True):
        #num_mutation can be variable.
        self.iterations = iterations
        self.num_mutation = num_mutation
        self.population_size = population_size
        self.logger = logger
        

    def mvmo(self, obj_fun, bounds, cons, x_initial):
        
        convergence = []
        min_b, max_b = np.asarray(bounds).T
        diff = np.fabs(min_b - max_b)

        # problem dimension
        D = len(bounds)

        # create storage df
        solutions_d = pd.DataFrame()
        metrics_d = {}

        if x_initial:
            x0 = (np.asarray(x_initial) - min_b) / (max_b - min_b)
            
        else:
            # generate initial random solution
            x0 = np.random.uniform(size=D)

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
        solutions_d[fitness] = x0.tolist()

        # initial metric is set to 0.5 for mean
        metrics_d['mean'] = 0.5 * np.ones(D)
        metrics_d['variance'] = 0.5 * np.ones(D)

        # TODO: How to define initial scaling factors???
        for i in tqdm(range(self.iterations)):
            # parent
            x_parent = solutions_d.loc[:, min(solutions_d.columns)].to_numpy()
            #this is strategy 4 where you select randomly. 
            #another strategy is fix one variable from vector. and mutate num_mutations-1 that are randomly selected from remaining.
            idxs = np.random.choice(
                list(range(len(bounds))), self.num_mutation)

            for idx in idxs:
                # mean
                x_bar = metrics_d['mean'][idx]

                xi_star = np.random.uniform(0, 1, 1)[0]
                var = metrics_d['variance'][idx]
                #scaling factor can be variable. This affects converegnce so play with it. 
                # maybe increase quadratically or someway with number of iterations.
                #when no improvement in solutions is observed, change it back to one for more explorstion\
                
                scaling_factor = 1 + 1/self.iterations * (20 - 1)
                # print(min(var,1e-5))
                s_old = -np.log(var) * scaling_factor
                
                #this 0.5 can also be adaptive
                
                if xi_star < 0.5:
                    s_new = s_old/(1 - x_bar)
                    hm = x_bar - x_bar/(0.5*s_new + 1)
                    hf = x_bar * (1 - np.exp(-xi_star * s_new))
                    hc = (x_bar - hm) * 2 * xi_star
                    xi_new = hf + hc
                else:
                    s_new = s_old/x_bar
                    hm = (1 - x_bar)/(0.5*s_new + 1)
                    hb = (1 - x_bar) / ((1 - xi_star) * s_new + 1) + x_bar
                    hc = hm * 2 * (1 - xi_star)
                    xi_new = hb - hc
                
                # Old method
                #variance = metrics_d['variance'][idx]
                # if variance < 1e-5:
                #    var = 1e-5
                #    scaling_factor = 1
                # else:
                #    var = variance
                #    scaling_factor = 1
                #s1 = -np.log(var) * scaling_factor
                #s2 = s1

                # hx = x_bar*(1-np.exp(-xi_star*s1)) + \
                #    (1-x_bar)*np.exp(-(1-xi_star)*s2)
                #h1 = x_bar*(1-np.exp(-1*s1)) + (1-x_bar)*np.exp(-(1-1)*s2)
                #h0 = x_bar*(1-np.exp(-0*s1)) + (1-x_bar)*np.exp(-(1-0)*s2)

                #xi_new = hx + (1-h1+h0)*xi_star - h0
                x_parent[idx] = xi_new

            x_denorm_t = min_b + np.asarray(x_parent) * diff
            x_denorm = x_denorm_t.tolist()

            a = obj_fun(x_denorm)
            if not a:
                continue

            sol_good = self.constraint_check(x_denorm, cons)
            if sol_good:                
                fitness = round(a, 4)
            else:
                fitness = 1e10 #penalty for constraint violation
            
            if fitness >= max(solutions_d.columns) or fitness in list(solutions_d.columns):
                convergence.append(convergence[-1])
            else:
                solutions_d[fitness] = x_parent

                if len(solutions_d.columns) > self.population_size:
                    solutions_d.pop(max(solutions_d.columns))

                convergence.append(min(solutions_d.columns))

                metrics_d['variance'] = [
                    np.var(solutions_d.iloc[x, :]) for x in range(len(solutions_d))]
                metrics_d['mean'] = [
                    np.mean(solutions_d.iloc[x, :]) for x in range(len(solutions_d))]

        res = min_b + \
            np.asarray(solutions_d.loc[:, min(
                solutions_d.columns)].tolist()) * diff
        return [convergence[-1], res], convergence, solutions_d
    
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
        self.res, self.conv, self.sol = self.mvmo(
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

        return self.res, self.conv, self.sol

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