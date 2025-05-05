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
    K_goal = 0.5
    d_goal = np.sqrt((q[0]-q_goal[0])**2+(q[1]-q_goal[1])**2)
   
    if d_goal >= 40:
        grad_f_goal = (K_goal*(q_goal-q))/d_goal
        desired_angle = np.arctan2(q_goal[1]-q[1], q_goal[0]-q[0])
        angle_diff = desired_angle - q[2]
       
        heading_error = np.arctan2(np.sin(angle_diff), np.cos(angle_diff)) % np.pi
       
        if abs(heading_error) >= np.pi/4:
            grad_f_goal = np.array([0, 0, heading_error])
        else:
            grad_f_goal[2] = heading_error
    elif d_goal<40 and d_goal>=5:
        grad_f_goal = (K_goal*(q_goal-q))
        desired_angle = np.arctan2(q_goal[1]-q[1], q_goal[0]-q[0])
        angle_diff = desired_angle - q[2]
       
        heading_error = np.arctan2(np.sin(angle_diff), np.cos(angle_diff)) % np.pi
       
        if abs(heading_error) >= np.pi/4:
            grad_f_goal = np.array([0, 0, heading_error])    
        else:
            grad_f_goal[2] = heading_error
    else:
        grad_f_goal = np.array([0, 0, q_goal[2] % np.pi])
   
    # Obstacle avoidance parameters
    K_obs = 100000 # Strength of obstacle avoidance
    d_safe = 200.0  # Safe distance from obstacles
   
    # Initialize obstacle gradient
    grad_f_obs = np.zeros(2)

    d_obs= min(ranges)
    angle= angles[np.argmin(ranges)]
   
    obs_x = d_obs * np.cos(angle)
    obs_y = d_obs * np.sin(angle)
      
    q_obs = np.array([
            q[0] + obs_x * np.cos(q[2]) - obs_y * np.sin(q[2]),
            q[1] + obs_x * np.sin(q[2]) + obs_y * np.cos(q[2])
        ])
           
           
    if d_obs < d_safe:
        obs_to_robot = q_obs - q[:2]
               
        repulsive_factor = (K_obs / (d_obs**3)) * (1/d_obs - 1/d_safe)
        grad_f_obs = repulsive_factor * obs_to_robot
    else:
        grad_f_obs = np.zeros(2)
   
    # Calculate obstacle avoidance heading if obstacles are present
    if np.linalg.norm(grad_f_obs) > 0:
        obstacle_angle = np.arctan2(grad_f_obs[1], grad_f_obs[0])
        # Convert to robot frame
        obstacle_angle_local = obstacle_angle - q[2]
        # Normalize angle
        obstacle_angle_local = np.arctan2(np.sin(obstacle_angle_local), np.cos(obstacle_angle_local))
        
        # Blend goal heading with obstacle avoidance heading
        # Higher weight for obstacles when they're closer
        obstacle_weight = min(1.0, np.linalg.norm(grad_f_obs) / (K_obs / (d_safe**3)))
        goal_weight = 1.0 - obstacle_weight
        
        # Combine the headings with weights
        blended_heading = goal_weight * grad_f_goal[2] + obstacle_weight * obstacle_angle_local
    else:
        blended_heading = grad_f_goal[2]
   
    # Combine attractive and repulsive gradients
    grad_f = np.zeros(3)
    grad_f[:2] = grad_f_goal[:2] + grad_f_obs[:2]
    grad_f[2] = blended_heading  # Use the blended heading that includes obstacle influence
    print(grad_f)
   
    forward = round(np.linalg.norm(grad_f[0:2]),2)
    rotation = round(grad_f[2]/(np.pi*2),4)
    print("forward: "+ str(forward) +"\n rotation: "+ str(rotation))
    command = {"forward": forward, "rotation": rotation}
    return command
