# -*- coding:utf-8 -*-
'''
Created on 2016-10-02
StatLearning Homework 1 
reference here: Constrained minimization of multivariate scalar functions (minimize)
http://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#tutorial-sqlsp
@author: RENAIC225
'''
from scipy.optimize import minimize
import numpy as np

def func(x, sign=1.0):
    """ Objective function """
    return sign*(10 - x[0]**2 - x[1]**2)

def func_deriv(x, sign=1.0):
    """ Derivative of objective function """
    dfdx0 = sign*(-2*x[0] )
    dfdx1 = sign*(-2*x[1])
    return np.array([ dfdx0, dfdx1 ])
    
cons = (
        {'type': 'eq',  
         'fun': lambda x: np.array([x[0]+x[1]]), 
         'jac': lambda x: np.array([1., 1.])},
        {'type': 'ineq', 
         'fun': lambda x: np.array(-x[0]**2 + x[1] ), 
         'jac': lambda x: np.array([-2*x[0], 1.0])}  
        )

res = minimize(func, [-10., 10.], args=(1.0,), jac=func_deriv,
               constraints=cons, method='SLSQP', options={'disp': True})

print res.x

'''
SLSQP(Sequential Least SQuares Programming optimization algorithm)is utilized 
to solve this constrained nonlinear optimization problem.  
'''