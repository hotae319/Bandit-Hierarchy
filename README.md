# Bandit-Hierarchy
05/07/2021

The entire code works. I tried only one feasible trajectory and check if it works well.
Under uncertain obstacles and dynamics, the code works well and the learning curve looks good. 

## Done
* Obtain each cell's information and pick the best cell bsaed on the reward
* Compute the predicted local reward, actual local reward
* Estimate A,B using regression
* Update the occupancy, whether the cell is visited or not
* Update cost2go(global reward) --> It already has the data structure. Just update after comparing with the sequence of cells in current iteration.

## TODO
* Add exploration (it is epsilon greedy for now)

