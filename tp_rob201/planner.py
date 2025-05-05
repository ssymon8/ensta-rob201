"""
Planner class
Implementation of A*
"""

import numpy as np

from occupancy_grid import OccupancyGrid


class Planner:
    """Simple occupancy grid Planner"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid


        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])
    
    def get_neighbors(self, current_cell):
        neighbors= np.array()

        for i in range(-1,2):
            for j in range(-1,2):
                voisin= [current_cell[0] + i, current_cell[0] + j]
                if i==0 and j==0:
                    break
                else:
                    np.append(neighbors, voisin)
        
        return neighbors
    


    def plan(self, start, goal):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates (theta unused)
        goal : [x, y, theta] nparray, goal pose in world coordinates (theta unused)
        """
        # TODO for TP5

        path = [start, goal]  # list of poses
        return path

    def explore_frontiers(self):
        """ Frontier based exploration """
        goal = np.array([0, 0, 0])  # frontier to reach for exploration
        return goal
