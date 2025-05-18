""" A set of robotics control functions """

import random
import numpy as np


def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TODO for TP1

    laser_dist = lidar.get_sensor_values()
    ray_angles = lidar.get_ray_angles()

    cone = laser_dist[150:210]

    new_dist=True

    if min(cone)>=100:
        speed = 1.0
        rotation_speed = 0.0
        new_dist=True
    else:
        if new_dist:
            speed=0.0
            rotation_speed=1/2
            old_rota=rotation_speed
            new_dist=False
        else:
            speed = 0.0
            rotation_speed=old_rota
            
    

    command = {"forward": speed,
               "rotation": rotation_speed}

    return command


def potential_field_control(lidar, current_pose, goal_pose):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose in odom or world frame
    goal_pose : [x, y, theta] nparray, target pose in odom or world frame
    Notes: As lidar and odom are local only data, goal and gradient will be defined either in
    robot (x,y) frame (centered on robot, x forward, y on left) or in odom (centered / aligned
    on initial pose, x forward, y on left)
    """
    q = current_pose
    q_goal = goal_pose
   
    ranges = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()
    
    K_goal = 1
    d_goal = np.linalg.norm(q_goal[:2]-q[:2])
   
    if d_goal >= 100:
        grad_f_goal = [K_goal*(q_goal[0]-q[0])/d_goal, 
            K_goal*(q_goal[1]-q[1])/d_goal]
        
    elif d_goal<100 and d_goal>=20:
        grad_f_goal = [K_goal*(q_goal[0]-q[0])**2/(2*d_goal), 
            K_goal*(q_goal[1]-q[1])**2/(2*d_goal)]
    else:
        grad_f_goal = [0, 0]
   
    # Obstacle avoidance parameters
    K_obs = 1000 # Strength of obstacle avoidance
    d_safe = 250.0  # Safe distance from obstacles
   
    # Initialize obstacle gradient
    grad_f_obs = np.zeros(2)

    d_obs= min(ranges)
    angle= angles[np.argmin(ranges)]
   
    obs_x = d_obs * np.cos(angle)
    obs_y = d_obs * np.sin(angle)
      
    q_obs = [
            q[0] + obs_x * np.cos(q[2]) - obs_y * np.sin(q[2]),
            q[1] + obs_x * np.sin(q[2]) + obs_y * np.cos(q[2])
        ]
           
           
    grad_f_obs = [0,0]

    if d_obs < d_safe:
               
        repulsive_factor = (K_obs / (d_obs**3)) * (1/d_obs - 1/d_safe)

        obstacle_vector=[ q[0] - q_obs[0],
                         q[1] - q_obs[1]]
        
        if d_obs >0:
            obstacle_direction=[
                obstacle_vector[0]/d_obs,
                obstacle_vector[1]/d_obs
            ]

            grad_f_obs = [
                repulsive_factor * obstacle_direction[0],
                repulsive_factor * obstacle_direction[1]
            ]



    grad = [
        grad_f_goal[0]+grad_f_obs[0],
        grad_f_goal[1]+grad_f_obs[1]
    ]

    target_angle = np.arctan2(grad[1], grad[0])
    
    angle_error = target_angle - current_pose[2]
    angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi
    
    # CONTROLEUR PROPORTIONNEL
    K_p = 0.1
    forward = min(K_p * np.linalg.norm(grad), 1) if d_goal >= 40 else 0
    rotation = np.clip(K_p * angle_error, -1, 1)
    
    # COMMANDE
    if d_goal < 40:
        command = {"forward": 0, "rotation": 0}
    else:
        command = {"forward": forward, "rotation": rotation}
    
    return command 
