from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess

def generate_launch_description():
    # The absolute path to your Computer Vision repository
    
    base_path = '/home/bninaos/UAV-Tracking-Project/ros2_ws'

    '''
        ExecuteProcess(
            cmd=['python3', os.path.join(base_path, 'dummy_60fps.py')],
            cwd=base_path,
            output='screen'
        ),
    '''
    return LaunchDescription([
    ExecuteProcess(
            cmd=['python3', os.path.join(base_path, 'isaac_motor_sim.py')],
            cwd=base_path,
            output='screen'
        ),
        ExecuteProcess(
            cmd=['python3', os.path.join(base_path, 'isaac_track.py'), '--tsize', '640', '--exp_file', os.path.join(base_path, 'AeroTrack', 'exps', 'aerotrack_proposed.py'), "--trt", "--fuse", "--early_exit"],
            cwd=base_path,
            output='screen'
        ),
        ExecuteProcess(
            cmd=['ros2', 'run', 'micro_ros_agent', 'micro_ros_agent', 'serial', '--dev', '/dev/ttyACM0', '-b', '921600'],
            output='screen'
        ),
    ])

