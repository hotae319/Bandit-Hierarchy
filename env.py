
'''
© 2021 Hotae Lee <hotae.lee@berkeley.edu>
'''

# Import packages.
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import sparse
import sys, os
import casadi as ca
import copy
from math import cos, sin, pi
abspath = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(abspath)
# from utils.math_utils import rot2d, EulerDiscrete

class GridEnv:
    def __init__(self,N1,N2,lx,ly,x0):
        # N1 X N2 grids
        # each grid has a size of lx x ly
        # x0 : agent's initial state, [x,y,xdot,ydot]
        self.N1 = N1
        self.N2 = N2
        self.lx = lx
        self.ly = ly
        self.x0 = x0
        self.xcur = x0         
        self.xtraj = [x0[0]]
        self.ytraj = [x0[1]]

        # Each cell's information
        # previous_visit means this cell is included in the previous feasible trajectory
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
        cell_info_true = {'A':Ad,'B':Bd,'cost2go': 100, 'occupancy': 1, 'previous_visit': 1}
        cell_info_observed = {'A':Ad,'B':Bd,'cost2go': 100, 'occupancy': 1, 'previous_visit': 0} 
        # List including all cell's info
        # self.info[Gx][Gy] returns the each cell's info
        self.info_true = [[{} for i in range(self.N1)] for j in range(self.N2)]
        self.info = [[{} for i in range(self.N1)] for j in range(self.N2)]
        for i in range(self.N1):
            for j in range(self.N2):
                self.info_true[i][j] = copy.deepcopy(cell_info_true)
                self.info[i][j] = copy.deepcopy(cell_info_observed)
    def initialize(self,x0):
        self.xcur = x0         
        self.xtraj = [x0[0]]
        self.ytraj = [x0[1]]

    def set_occupancy(self,occ):
        # occ = [[1,2],[4,5]] (the set of (col,row) of grids)
        return 1
    def set_feasible_grids(self, grid_seq, cost2go_seq):
        for i in range(len(cost2go_seq)):
            grid = grid_seq[i]
            self.info[grid[0]][grid[1]]['cost2go'] = cost2go_seq[i]
            self.info[grid[0]][grid[1]]['previous_visit'] = 1
            # print(self.info[grid[0]][grid[1]]['cost2go'])            
        return 1
    def set_destination(self, Gx_goal, Gy_goal):
        self.destination = [Gx_goal, Gy_goal] 
    def set_target(self, Gtar):
        # Gcur : tuple (a,b)
        # Gtar : tuple (c,d)
        # center pos
        # xcur = self.lx * (Gcur[0]-1) + self.lx/2
        # ycur = self.ly * (Gcur[1]-1) + self.ly/2
        xtar = self.lx * (Gtar[0]) + self.lx/2
        ytar = self.ly * (Gtar[1]) + self.ly/2
        # self.xcur = np.array([xcur, ycur, 0, 0]) # we can consider the previous vel.
        self.xtar = np.array([xtar, ytar, 0, 0])
        return self.xtar
    def step(self, u, x = None):
        if x == None:
            x = self.xcur
        dt = 0.1
        # Decide actual A,B
        Gx = int(self.xcur[0])
        Gy = int(self.xcur[1])
        k = ((Gx+Gy)%3)*0.03
        k = 0.03
        # k = 0
        A = np.array([
            [1,0,dt,0],
            [0,1,0,dt],
            [0,0,1-k,0],
            [0,0,0,1-k]])        
        B = np.array([
            [0,0],
            [0,0],
            [dt,0],
            [0,dt]])
        b = 0.05
        a = -0.05
        w = ((b-a)*np.random.random_sample((4,)) + a)
        x_next = A@x + B@u #+ w
        self.xcur = x_next
        self.xtraj.append(x_next[0])
        self.ytraj.append(x_next[1])
    def deviation_cost(self, Q):
        err = self.xtar - self.xcur
        deviation_cost = 0.5* err.T@Q@err
        return deviation_cost
    def A_regressor(self, x_data, u_data):
        # Solve x_t+1= (Ad+del_A)x_t+Bdu_t for del_A
        # Also using sparsity in uncertainty del_A
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
        xt=x_data.T[:,0:-2]
        ut=u_data.T[:,1:-1]
        xtp1=x_data.T[:,1:-1]
        Adnp=Ad.toarray()
        Bdnp=Bd.toarray()

        print(xt.shape, xtp1.shape, ut.shape, Adnp.shape)
        # print(xt)
        residual=xtp1-np.dot(Adnp,xt)-np.dot(Bdnp,ut)
        residual=xtp1-Adnp@xt-Bdnp@ut
        a = 0
        b = 2
        # xt_compressed_pinv1 = np.linalg.pinv(xt[:2,a:b])
        xt_compressed_pinv = np.linalg.pinv(xt[2:,a:b])
        A_uncertain_compressed_fit=np.dot(residual[2:,a:b],xt_compressed_pinv)
        # print("xt,pinv, resid, A")
        # print(xt[2:,a:b],xt_compressed_pinv, residual[2:,a:b], A_uncertain_compressed_fit)
        del_A=np.block([[np.zeros((2,4))],[np.zeros((2,2)),A_uncertain_compressed_fit]])
        
        return Adnp+del_A
        
        
        
    def check_safegrid(self, Gx, Gy):
        # Whether we can go to the cell or not
        safety = 0
        for i in range(-1,2):
            for j in range(-1,2):
                cell_info = self.info[Gx+i][Gy+j]
                # print("check check_safegrid")
                # print(Gx,Gy,i,j, self.info[Gx+i][Gy+j]['previous_visit'])
                if cell_info['previous_visit'] == 1:
                    safety = 1                    
        return safety     
    def update_info_A(self, Gx, Gy, cell_info):
        self.info[Gx][Gy]['A'] = cell_info['A']
        self.info[Gx][Gy]['B'] = cell_info['B']
    def update_info_occupy(self, Gx, Gy, cell_info):
        self.info[Gx][Gy]['occupancy'] = cell_info['occupancy']
    def update_info_episodic(self, Gx, Gy, cell_info_epi):
        self.info[Gx][Gy]['previous_visit'] = cell_info_epi['previous_visit']
        self.info[Gx][Gy]['cost2go'] = cell_info_epi['cost2go']
    def get_info(self):
        Gx = int(self.xcur[0])
        Gy = int(self.xcur[1])
        info = self.info[Gx][Gy]
        return info, Gx, Gy
    def get_cost2go(self, Gx, Gy):
        return self.info[Gx][Gy]['cost2go']
    def get_cost2go_total(self):
        cost2go = np.zeros((self.N1,self.N2))
        for i in range(self.N2):
            for j in range(self.N1):
                cost2go[i,j] = self.get_cost2go(j,self.N2-i-1)
        return cost2go
    def get_previous_visit(self):
        previous_visit = np.zeros((self.N1,self.N2))
        for i in range(self.N2):
            for j in range(self.N1):                
                previous_visit[i,j] = self.info[j][self.N2-i-1]['previous_visit']
        return previous_visit
    def get_occupancy(self):
        occupancy = np.zeros((self.N1,self.N2))
        for i in range(self.N2):
            for j in range(self.N1):
                occupancy[i,j] = self.info[j][self.N2-i-1]['occupancy']
        return occupancy
    def observe(self):
        Gx = int(self.xcur[0])
        Gy = int(self.xcur[1])
        occupancy_surr = np.zeros((3,3)) # 1 : empty, 0 : occupied
        cango_surr = np.zeros((3,3)) # 1: can go, 0 : cannot go
        for i in range(-1,2):
            for j in range(-1,2):
                cell_info_true = self.info_true[Gx+i][Gy+j]
                occupancy_surr[1-j,i+1] = cell_info_true['occupancy']
                cango_surr[1-j,i+1] = self.check_safegrid(Gx+i,Gy+j)                
        return occupancy_surr, cango_surr 
    def visualize(self, pre_traj = []):
        fig, ax = plt.subplots()
        # draw gridlines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(0, self.lx*self.N1,self.lx));
        ax.set_yticks(np.arange(0, self.ly*self.N2,self.ly));
        ax.set_ylim(0,self.ly*self.N2)
        ax.set_xlim(0,self.lx*self.N1)
        plt.scatter(self.xcur[0], self.xcur[1], color = 'red', s= 50, marker =  'x' )
        plt.scatter(self.xtraj,self.ytraj)
        plt.scatter(self.xtar[0],self.xtar[1], color = 'green', s= 50, marker =  'o')
        plt.scatter(self.destination[0]*self.lx + self.lx/2,self.destination[1]*self.ly + self.ly/2, color = 'hotpink', s= 100, marker =  'p')
        # prediction plot
        pre_xtraj = []
        pre_ytraj = []
        for i in range(len(pre_traj)):
            pre_xtraj.append(pre_traj[i][0])
            pre_ytraj.append(pre_traj[i][1])
        plt.scatter(pre_xtraj,pre_ytraj, alpha = 0.5, marker = '^')
        plt.legend(['current', 'actual_traj','bandit target','goal','predicted_traj'])
        plt.show()

if __name__ == '__main__':
    env = GridEnv(10.,10.,1.,1.,np.array([2.5,3,0,0]))
    env.visualize()
