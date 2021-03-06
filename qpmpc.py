
'''
© 2021 Hotae Lee <hotae.lee@berkeley.edu>
'''

# Import packages.
import numpy as np
import osqp
import scipy as sp
from scipy import sparse
import random


import matplotlib.pyplot as plt
import sys, os
# import casadi as ca
from math import cos, sin, pi
abspath = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(abspath)
# from utils.math_utils import rot2d, EulerDiscrete
from env import GridEnv

# Objective function
Q = sparse.diags([10., 10., 1., 1.])*0.1
QN = Q
R = 0.1*sparse.eye(2)

def generate_osqp_param(Ad,Bd,x0,xr,N = 20):
    [nx, nu] = Bd.shape

    # Constraints
    u0 = 0.
    umin = np.array([-14.,-14.]) - u0
    umax = np.array([14., 14.]) - u0
    xmin = np.array([-1,-1,-10,-10])
    xmax = np.array([10,10,10,10])


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

def compute_local_reward(Ad,Bd,x0,xr, N= 20):
    [nx, nu] = Bd.shape
    # Create an OSQP object
    prob = osqp.OSQP()
    # Generate the parameter for osqp
    P, q, A, l, u = generate_osqp_param(Ad,Bd,x0,xr,N)
    # Setup workspace
    prob.setup(P, q, A, l, u, verbose=False)
    # CFTOC CONTROLLER
    # Solve
    res = prob.solve()
    # Check solver status
    if res.info.status != 'solved':
        print(Ad,Bd,x0,xr)
        raise ValueError('OSQP did not solve the problem!')
    ctrl_cftoc = res.x[-N*nu:]
    x_cftoc = res.x[0:-N*nu]
    return res.info.obj_val +1/2*xr@Q@xr*(N+1)

def compute_actual_reward(x_traj, xr, u_traj):
    reward_sum = 0
    for x_state in x_traj:
        reward_sum += (x_state.T-xr.T)@Q@(x_state-xr)
    for u in u_traj:
        reward_sum += u.T@R@u
    return reward_sum


N_episode = 50
N_iter = 15
Gx_goal = 7
Gy_goal = 7

# Initial and reference states
x0 = np.zeros(4)
x0 = np.array([3.5,4.5,0,1])
x0_pre = x0

# Grid Env setting
env = GridEnv(12,12,1.,1.,x0) # N1,N2,lx,ly,x0
env.set_destination(Gx_goal, Gy_goal)
occ = [[2,3],[6,6]]
env.set_occupancy(occ)
_, Gx, Gy = env.get_info()

# Feasible sequence of grid cells stored
grid_seq = [[3,4],[2,4],[2,5],[2,6],[2,7],[3,7],[4,8],[5,8],[6,8],[7,8],[7,9],[8,8],[7,7]]
cost2go_seq = [330,300,270,240,200,160,130,100,70,40,20,5,0] #[80,70,60,50,40,30,20,10]
print("cell info:")
print(env.info[3][4]['cost2go'])
env.set_feasible_grids(grid_seq, cost2go_seq)
print("cell info next:")
print(env.info[3][4]['cost2go'],env.info[3][5]['cost2go'],env.info[7][7]['cost2go'],env.info[5][4]['cost2go'])
print(env.info[3][4]['previous_visit'],env.info[3][5]['previous_visit'],env.info[7][7]['previous_visit'],env.info[5][4]['previous_visit'])
# print([ele['cost2go'] for ele in env.info_true])

cost_per_episode = []
feas_seq = []
i_epi = 0
while i_epi <= N_episode:
    print("NEW EPISODE")
    env.initialize(np.array([3.5,4.5,0,1]))
    x0 = np.array([3.5,4.5,0,1])
    _, Gx, Gy = env.get_info()
    i_iter = 0
    sum_local_cost = 0
    grid_traj = []
    cost_traj = []


    ################### START ONE ITERATION WITH BANDIT AND CFTOC ##################
    while i_iter <= N_iter and (Gx != Gx_goal or Gy != Gy_goal):
        # BANDIT POLICY

        # check the current grid cell and observe the context (Surrounding occupancy/which cells I can go to)
        cell_info, Gx, Gy = env.get_info()
        grid_traj.append([Gx,Gy])
        occupancy_surr, cango_surr = env.observe() # 3 by 3 grid matrix

        # update the observed grid cell's info
        # env.update_info(Gx, Gy, cell_info)

        # pick out the available next grid cell
        surr = occupancy_surr + cango_surr
        grid_idxs = np.argwhere(surr == 2) # find the empty and safe cell
        
        # if we do not have any possible ways (it means this is the first trial) (if it went to bad direction, the only choice is going back)
        # if len(grid_idxs) == 0:
        #     grid_idxs = np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]])

        # np.set_printoptions(linewidth=np.inf)
        # print(env.get_cost2go_total())
        # print(env.get_previous_visit())
        # print(env.get_occupancy())
        # print(env.get_occupancy_true())
        # print(occupancy_surr, cango_surr, Gx,Gy)
        # estimate the each reward based on (A,B,x0,xgoal) + R_global
        rewards = []
        print(grid_idxs)
        for k in range(len(grid_idxs)):
            grid_idx = grid_idxs[k]
            if grid_idx[0] == 1 and grid_idx[1] == 1:
                # for removing center
                k_rmv = k
            else: 
                # set target state based on the next grid cell
                xr = env.set_target((Gx + grid_idx[1] - 1, Gy + 1 - grid_idx[0])) # x = x_cur + col-1, y = y_cur + 1-row
                r_local = compute_local_reward(cell_info['A'],cell_info['B'],x0,xr)/4
                r_global =  env.info[Gx + grid_idx[1] - 1][Gy + 1 - grid_idx[0]]['cost2go']
                # print("r_global  at {},{} :{}".format(Gx + grid_idx[1] - 1,Gy + 1 - grid_idx[0],r_global))
                rewards.append(r_local+r_global)
        grid_idxs = np.delete(grid_idxs, k_rmv, 0)
        print("grid_idxs,k {}, {}".format(grid_idxs,k_rmv))
        # pick the best grid cell (or exploration) and decide the target (env.set_target)
        p = np.random.random()
        epi_exploration = N_episode*0.4
        if i_epi < epi_exploration:
            eps = 0.05 + (epi_exploration - i_epi)/(epi_exploration)*0.8
        else:
            eps = 0.05
        if p < eps:
            # i_min = random.randint(0,len(rewards)-1)
            a = np.arange(0,len(rewards))
            print("a,rewards,indx {},{},{}".format(a,rewards,rewards.index(min(rewards))))
            np.delete(a,rewards.index(min(rewards)))
            i_min = random.choice(a)
        else:
            i_min = rewards.index(min(rewards))
        xr = env.set_target((Gx + grid_idxs[i_min][1] - 1, Gy + 1 - grid_idxs[i_min][0])) 
        print("xr {}".format(xr))
        print("check grid")
        print(Gx,Gy, grid_idxs[i_min], grid_idxs, rewards, r_global, r_local)
        print("p {}".format(p))
        # grid_traj.append(Gx,Gy)



        # solve CFTOC

        pred_traj = []
        actual_traj = []
        u_traj = []
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
        N = 20


        # Create an OSQP object
        prob = osqp.OSQP()


        # Generate the parameter for osqp
        P, q, A, l, u = generate_osqp_param(Ad,Bd,x0,xr,N)

        # Setup workspace
        prob.setup(P, q, A, l, u, warm_start=True, verbose=False)


        # MPC CONTROLLER
        # Simulate in closed loop
        # nsim = 50
        # for i in range(nsim):
        #     # Solve
        #     res = prob.solve()

        #     # Check solver status
        #     if res.info.status != 'solved':
        #         raise ValueError('OSQP did not solve the problem!')

        #     # Apply first control input to the plant
        #     ctrl = res.x[-N*nu:-(N-1)*nu]
        #     # prediction    
        #     x0_pre = Ad.dot(x0) + Bd.dot(ctrl)
        #     pred_traj.append(x0_pre) 
        #     # actual 
        #     env.step(ctrl)
        #     x0 = env.xcur
        #     # print(x0)

        #     # Update initial state
        #     l[:nx] = -x0
        #     u[:nx] = -x0
        #     prob.update(l=l, u=u)



        # CFTOC CONTROLLER
        # Solve
        res = prob.solve()
        # Check solver status
        if res.info.status != 'solved':            
            raise ValueError('OSQP did not solve the problem!')
        ctrl_cftoc = res.x[-N*nu:]
        x_cftoc = res.x[0:-N*nu]
        # print(x_cftoc)
        nsim = N
        for i in range(nsim):
            # Apply control input to the plant
            ctrl = ctrl_cftoc[i*nu:(i+1)*nu]
            # prediction    
            x0_pre = Ad.dot(x0) + Bd.dot(ctrl)
            pred_traj.append(x0_pre) 
            # actual 
            env.step(ctrl)
            x0 = env.xcur
            actual_traj.append(env.xcur)
            u_traj.append(ctrl)
        # env.visualize(pred_traj)

        # compute the actual cost and store the cost between two grid cells
        actual_cost_local = compute_actual_reward(actual_traj, xr, u_traj)
        cost_traj.append(actual_cost_local)
        sum_local_cost += actual_cost_local
        print(actual_cost_local)
        # estimate A,B from data (regression) and store/update A,B 
        A_reg = env.A_regressor(np.array(actual_traj), np.array(u_traj))
        info_new = {'A':A_reg,'B':Bd,'occupancy': 1} 
        env.update_info_A(Gx, Gy, info_new) 
        # update the 'occupancy', 'previous_visit' of observed grid cells
        for i in range(3):
            for j in range(3):
                if occupancy_surr[i,j] == 1: 
                    # not occupied
                    info_new = {'occupancy': 1} 
                else:
                    # occupied
                    info_new = {'occupancy': 0} 
                env.update_info_occupy(Gx+i-1,Gy+1-j, info_new)
        i_iter += 1
    i_epi += 1
    print("sum_local_cost:{}".format(sum_local_cost))
   
    # print the info
    np.set_printoptions(linewidth=np.inf)
    print(env.get_cost2go_total())
    print(env.get_previous_visit())
    print(env.get_occupancy())
    print(env.get_occupancy_true())
    print(occupancy_surr, cango_surr, Gx,Gy)

    ################### AFTER REACHING THE GOAL#################
    if i_iter > N_iter:
        # failure
        #  no need to update
        cost_per_episode.append(330)
    else:
        # need to squeeze the grid_traj
        cost_per_episode.append(sum_local_cost)
        grid_traj_squeezed = grid_traj
        cost_traj_squeezed = cost_traj
        cost2go_traj = []
        for i in range(len(cost_traj_squeezed)):    
            cost2go_traj.append(sum(cost_traj_squeezed[i:]))
        # update cost2go and visitation of each grid using grid_traj
        for i in range(len(grid_traj_squeezed)):        
            grid_idx = grid_traj_squeezed[i]
            cost2go_cur = env.get_cost2go(grid_idx[0], grid_idx[1])        
            info_new = {'cost2go': min(cost2go_traj[i],cost2go_cur), 'previous_visit': 1} 
            env.update_info_episodic(grid_idx[0], grid_idx[1], info_new)
        print("cost2go traj, grid traj:")
        print(cost2go_traj, grid_traj_squeezed)
        np.set_printoptions(linewidth=np.inf)
        # env.visualize()
        print(env.get_cost2go_total())
print("cost per epi")
print(cost_per_episode)
env.visualize()
plt.figure(1)
plt.plot(range(len(cost_per_episode)),cost_per_episode)
plt.title("learning curve of actual cost")
# plt.legend(['ref_box','actual_box','r1','r2', 'p1','p2'])
plt.ylabel("actual cost", fontsize = 12)
plt.xlabel("episode" , fontsize =12, labelpad = 10)
plt.show()
np.set_printoptions(linewidth=np.inf)
print(env.get_cost2go_total())
print(env.get_previous_visit())
# print(env.xtraj)
# print(pred_traj)




