
'''
© 2021 Hotae Lee <hotae.lee@berkeley.edu>
'''

# Import general packages.
import numpy as np
import scipy as sp
from scipy import sparse
import osqp
import matplotlib.pyplot as plt
import time
import os, sys
# Import optimization solvers
# import cvxpy as cp
import casadi as ca
from math import sin, cos, pi

class CFTOC:
    def __init__(self, x0, N, m):
        '''
        x0 : initial state
        N : horizon
        '''
        self.N = N
        self.x0 = x0
        self.n = len(x0)
        self.m = m        
        self.opti = ca.Opti()
    def set_cost(self, Q, QN, R):

    def set_model(self, model = None):
        # model function
        # We can bring model from model.py
        if model == None:
            def func(x,u):
                A = ca.DM(np.ones((self.n, self.n)).tolist())
                B = ca.DM(np.random.rand(self.n, self.m).tolist())
                output = A@x + B@u
                return output       
            # Add model constraints  
            for i in range(self.N-1):
                self.opti.subject_to(self.x[:,i+1] == func(self.x[:,i],self.u[:,i]))
        else:
            self.model = model
            for i in range(self.N-1):
                self.opti.subject_to(self.x[:,i+1] == model.f_casadi(self.x[:,i],self.u[:,i]))
                # self.opti.subject_to(self.x[:,i+1] == model.f_casadi_implicit(self.x[:,i],self.u[:,i], self.fc[:,i]))
    def set_bound(self):
        xmin = ca.DM((-500*np.zeros((self.n,1))).tolist())
        xmax = ca.DM((500*np.ones((self.n,1))).tolist())
        umin = ca.DM((-500*np.zeros((self.m,1))).tolist())
        umax = ca.DM((500*np.ones((self.m,1))).tolist())
        for i in range(self.N-1):
            self.opti.subject_to(self.opti.bounded(xmin, self.x[:,i], xmax))
            self.opti.subject_to(self.opti.bounded(umin, self.u[:,i], umax))
    def set_init_constr(self):
        self.opti.subject_to(self.x[:,0] == self.x0)
    def set_constr(self):
    	    def solve(self):
        # Set optimiztion options
        p_opts = {"expand": True}
        s_opts = {"max_iter": 100, "print_level": 1}
        self.opti.solver("ipopt", p_opts, s_opts)
        # initial values for optimization
        self.opti.set_initial(self.x[:,0], self.x0)

        # Solve optimization problems
        start_time = time.time()
        try:
            sol = self.opti.solve()
            x_cftoc = sol.value(self.x)
            u_cftoc = sol.value(self.u)
            is_opti = True
        except:
            # Sub-optimal(intermedite sol'n)
            x_cftoc = self.opti.debug.value(self.x)
            u_cftoc = self.opti.debug.value(self.u)
            is_opti = False
        
        solve_time = time.time() - start_time
        sol_dict = {}
        # print(x_cftoc[:,0])
        # print(x_cftoc)
        sol_dict['u_first'] = u_cftoc[:,0]
        sol_dict['x_traj'] = x_cftoc
        sol_dict['u_traj'] = u_cftoc
        sol_dict['is_opti'] = is_opti
        if is_opti == True:
            sol_dict['cost'] = sol.value(self.cost)
        return sol_dict