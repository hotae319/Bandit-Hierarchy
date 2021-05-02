
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
        cell_info = {'A':Ad,'B':Bd,'cost2go': 100, 'occupancy': 1, 'previous_visit': 1} 
        # List including all cell's info
        # self.info[Gx][Gy] returns the each cell's info
        self.info = [[cell_info]*self.N1]*self.N2 
    def set_occupancy(self,occ):
        # occ = [[1,2],[4,5]] (the set of (col,row) of grids)
        return 1
    def set_target(self, Gtar):
        # Gcur : tuple (a,b)
        # Gtar : tuple (c,d)
        # center pos
        # xcur = self.lx * (Gcur[0]-1) + self.lx/2
        # ycur = self.ly * (Gcur[1]-1) + self.ly/2
        xtar = self.lx * (Gtar[0]-1) + self.lx/2
        ytar = self.ly * (Gtar[1]-1) + self.ly/2
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
        xt=x_data[:,0:-1]
        ut=u_data
        xtp1=x_data[:,1:]
        Adnp=Ad.toarray()
        Bdnp=Bd.toarray()
        residual=xtp1-np.dot(Adnp,xt)-np.dot(Bdnp,ut)
        xt_compressed_pinv=np.linalg.pinv(xt[2:,:])
        A_uncertain_compressed_fit=np.dot(residual[2:,:],xt_compressed_pinv)
        del_A=np.block([[np.zeros((2,4))],[np.zeros((2,2)),A_uncertain_compressed_fit]])
        
        return Adnp+del_A
        
        
        
    def check_safegrid(self, Gx, Gy):
        # Whether we can go to the cell or not
        for i in range(-1,2):
            for j in range(-1,2):
                cell_info = self.info[Gx+i][Gx+j]
                if cell_info['previous_visit'] == 1:
                    safety = 1
                    break
                else:
                    safety = 0
        return safety     
    def store_info(self, Gx, Gy):
        # self.info = 0
        a = 1
    def get_info(self):
        Gx = int(self.xcur[0])
        Gy = int(self.xcur[1])
        info = self.info[Gx][Gy]
        return info, Gx, Gy
    def observe(self):
        Gx = int(self.xcur[0])
        Gy = int(self.xcur[1])
        occupancy_surr = np.zeros((3,3)) # 1 : empty, 0 : occupied
        cango_surr = np.zeros((3,3)) # 1: can go, 0 : cannot go
        for i in range(-1,2):
            for j in range(-1,2):
                cell_info = self.info[Gx+i][Gx+j]
                occupancy_surr[i+1,j+1] = cell_info['occupancy']
                cango_surr[i+1,j+1] = self.check_safegrid(Gx+i,Gy+j)
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
        # prediction plot
        pre_xtraj = []
        pre_ytraj = []
        for i in range(len(pre_traj)):
            pre_xtraj.append(pre_traj[i][0])
            pre_ytraj.append(pre_traj[i][1])
        plt.scatter(pre_xtraj,pre_ytraj, alpha = 0.5, marker = '^')
        plt.legend(['current', 'actual_traj','predicted_traj'])
        plt.show()

if __name__ == '__main__':
    env = GridEnv(10.,10.,1.,1.,np.array([2.5,3,0,0]))
    env.visualize()
