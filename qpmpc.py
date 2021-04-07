
'''
© 2021 Hotae Lee <hotae.lee@berkeley.edu>
'''

# Import packages.
import numpy as np
import osqp
import scipy as sp
from scipy import sparse

import matplotlib.pyplot as plt
import sys, os
# import casadi as ca
from math import cos, sin, pi
abspath = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(abspath)
# from utils.math_utils import rot2d, EulerDiscrete
from env import GridEnv



def generate_osqp_param(Ad,Bd,x0,xr,N = 10):
    [nx, nu] = Bd.shape

    # Constraints
    u0 = 0.
    umin = np.array([-4.,-4.]) - u0
    umax = np.array([4., 4.]) - u0
    xmin = np.array([-1,-1,-10,-10])
    xmax = np.array([10,10,10,10])

    # Objective function
    Q = sparse.diags([10., 10., 10., 10.])
    QN = Q
    R = 0.1*sparse.eye(2)


    # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
    # - quadratic objective
    P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                           sparse.kron(sparse.eye(N), R)], format='csc')
    # - linear objective
    q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr),
                   np.zeros(N*nu)])
    # - linear dynamics
    Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
    Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
    Aeq = sparse.hstack([Ax, Bu])
    leq = np.hstack([-x0, np.zeros(N*nx)])
    ueq = leq
    # - input and state constraints
    Aineq = sparse.eye((N+1)*nx + N*nu)
    lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
    uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
    # - OSQP constraints
    A = sparse.vstack([Aeq, Aineq], format='csc')
    l = np.hstack([leq, lineq])
    u = np.hstack([ueq, uineq])
    return P, q, A, l, u




pred_traj = []
# Discrete time model 
dt = 0.1
Ad = sparse.csc_matrix([
    [1,0,dt,0],
    [0,1,0,dt],
    [0,0,1,0],
    [0,0,0,1]])
Bd = sparse.csc_matrix([
    [0,0],
    [0,0],
    [dt,0],
    [0,dt]])
[nx, nu] = Bd.shape

# Prediction horizon
N = 10

# Initial and reference states
x0 = np.zeros(4)
x0 = np.array([3.5,4.5,0,1])
x0_pre = x0
# xr = np.array([0.,1.,1.,0.])

# Grid Env setting
env = GridEnv(20.,20.,1.,1.,x0) # N1,N2,lx,ly,x0
xr = env.set_target((2,3))

# Create an OSQP object
prob = osqp.OSQP()


# Generate the parameter for osqp
P, q, A, l, u = generate_osqp_param(Ad,Bd,x0,xr,N)

# Setup workspace
prob.setup(P, q, A, l, u, warm_start=True)

# Simulate in closed loop
nsim = 50
for i in range(nsim):
    # Solve
    res = prob.solve()

    # Check solver status
    if res.info.status != 'solved':
        raise ValueError('OSQP did not solve the problem!')

    # Apply first control input to the plant
    ctrl = res.x[-N*nu:-(N-1)*nu]
    # prediction    
    x0_pre = Ad.dot(x0) + Bd.dot(ctrl)
    pred_traj.append(x0_pre) 
    # actual 
    env.step(ctrl)
    x0 = env.xcur
    # print(x0)


    # Update initial state
    l[:nx] = -x0
    u[:nx] = -x0
    prob.update(l=l, u=u)

print(env.xtraj)
print(pred_traj)
env.visualize(pred_traj)



