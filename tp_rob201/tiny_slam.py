""" A simple robotics navigation code including SLAM, exploration, planning"""

import cv2
import numpy as np
from occupancy_grid import OccupancyGrid

LOC_MAX_ITER=100

class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def _score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        
        map = self.grid.occupancy_map
        score = 0

        ranges= lidar.get_sensor_values()
        angles= lidar.get_ray_angles()

        index= ranges<lidar.max_range

        ranges = ranges[index]
        angles = angles[index]

        x,y,theta= pose[0], pose[1], pose[2]

        x_lidar= x + ranges* np.cos(angles+theta)
        y_lidar= y + ranges* np.sin(angles+theta)

        x_map, y_map = self.grid.conv_world_to_map(x_lidar, y_lidar)

        #for x in x_map:
        score= np.sum(map[x_map,y_map])

        return score

    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        if odom_pose_ref is None:
            odom_pose_ref= self.odom_pose_ref
        
        x_ref, y_ref, theta_ref = odom_pose_ref[0], odom_pose_ref[1], odom_pose_ref[2]
        x0, y0, theta0 = odom_pose[0], odom_pose[1], odom_pose[2]
        alpha0= np.arctan2(y0,x0)

        d0= np.sqrt(x0**2+y0**2)

        x_corrected= x_ref + d0*np.cos(theta_ref+alpha0)
        y_corrected= y_ref + d0*np.sin(theta_ref+ alpha0)
        theta_corrected= theta_ref + theta0

        corrected_pose= [x_corrected, y_corrected, theta_corrected]

        return corrected_pose

    def localise(self, lidar, raw_odom_pose):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """

        best_score = 0
        ref = self.odom_pose_ref

        for i in range(LOC_MAX_ITER):
            offset= np.random.normal(0.0, 2.0, (3,1))
            ref[0]= ref[0] + offset[0]
            ref[1]= ref[1] + offset[1]
            ref[2]= ref[2] + offset[2]

            new_pose= self.get_corrected_pose(raw_odom_pose, ref)
            current_score=self._score(lidar, new_pose)

            if current_score>best_score:
                best_score=current_score
                self.odom_pose_ref= ref

        return best_score

    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        laser_dist = lidar.get_sensor_values()
        ray_angles = lidar.get_ray_angles()

        p_vide=0.01
        p_occupe=0.95

        x, y, theta = pose[0], pose[1], pose[2]

        detect_x= x+ laser_dist*np.cos(ray_angles+theta)
        detect_y= y+ laser_dist*np.sin(ray_angles+theta)

        detect2_x= x+ 0.9*laser_dist*np.cos(ray_angles+theta)
        detect2_y= y+ 0.9*laser_dist*np.sin(ray_angles+theta)

        for i in range(len(detect_x)):
            self.grid.add_value_along_line(x, y, detect2_x[i], detect2_y[i],np.log(p_vide/(1-p_vide)))
            
        
        self.grid.add_map_points(detect_x, detect_y, np.log(p_occupe/(1-p_occupe))-np.log(p_vide/(1-p_vide)))

        self.grid.occupancy_map= np.clip(self.grid.occupancy_map, -40, 40)




    def compute(self):
        """ Useless function, just for the exercise on using the profiler """
        # Remove after TP1

        ranges = np.random.rand(3600)
        ray_angles = np.arange(-np.pi, np.pi, np.pi / 1800)

        # Poor implementation of polar to cartesian conversion
        points = []
        for i in range(3600):
            pt_x = ranges[i] * np.cos(ray_angles[i])
            pt_y = ranges[i] * np.sin(ray_angles[i])
            points.append([pt_x, pt_y])
