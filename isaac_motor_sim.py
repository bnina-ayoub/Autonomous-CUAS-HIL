#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
import json
import time
import math
from rcl_interfaces.msg import ParameterDescriptor

class IsaacPIDControllerNode(Node):
    def __init__(self):
        super().__init__('isaac_pid_controller')
        
        # --- SÉPARATION DES PARAMÈTRES PAN ET TILT ---
        desc_float = ParameterDescriptor(description="Tunable PID parameter for thesis tuning")

        self.declare_parameter('kp_pan', 0.0075, desc_float)  # 0.0039
        self.declare_parameter('ki_pan', 0.00003, desc_float) # 0.00003
        self.declare_parameter('kd_pan', 0.00085, desc_float) # 0.00005
        self.declare_parameter('max_int_pan', 50.0, desc_float)

        self.declare_parameter('kp_tilt', 0.0028, desc_float) # 0.0002
        self.declare_parameter('ki_tilt', 0.00003, desc_float) # 0.00001
        self.declare_parameter('kd_tilt', 0.0002, desc_float) # 0.00009
        self.declare_parameter('max_int_tilt', 30.0, desc_float)
        self.declare_parameter('warmup_factor', 0.85, desc_float)
        
        self.declare_parameter('alpha', 0.75, desc_float)
        self.joint_pub = self.create_publisher(JointState, 'anti_uav/joint_command', 5)
        self.foxglove_pub = self.create_publisher(JointState, '/foxglove_debug', 5)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        self.track_sub = self.create_subscription(String, 'anti_uav/output_track', self.track_callback, 5)

        self.center_x = 320.0  
        self.center_y = 320.0  
        
        self.ema_x = self.center_x
        self.ema_y = self.center_y
        self.first_frame = True
        self.initial_pos_set = False
        self.first_detect_time = 0.0

        self.prev_err_x = 0.0
        self.prev_err_y = 0.0
        self.filtered_deriv_x = 0.0
        self.filtered_deriv_y = 0.0
        self.int_x = 0.0
        self.int_y = 0.0

        self.current_pan = 0.0
        self.current_tilt = 0.0
        self.sim_tilt_velocity = 0.0

        # State tracking for the timeout logic
        self.last_msg_time = time.time()
        self.target_lost = False

        # Run the control loop strictly at 60Hz (0.016s) independently of callbacks
        self.control_timer = self.create_timer(0.016, self.control_loop)

        self.get_logger().info("EMA PID Controller Node is LIVE! (Pan/Tilt Separated with Foxglove Telemetry)")

    def joint_state_callback(self, msg):
        try:
            tilt_idx = msg.name.index('tilt_joint')
            self.sim_tilt_velocity = msg.velocity[tilt_idx]
        except (ValueError, IndexError):
            pass

    def track_callback(self, msg):
        try:
            data = json.loads(msg.data)
            cx = float(data["cx"])
            cy = float(data["cy"])
        except Exception as e:
            self.get_logger().error(f"Failed to parse tracking JSON: {e}")
            return
        

        # Update last message time for timeout check
        self.last_msg_time = time.time()
        self.target_lost = False

        if self.first_frame:
            self.ema_x = cx
            self.ema_y = cy
            self.first_detect_time = time.time()
            self.first_frame = False
        else:
            live_alpha = self.get_parameter('alpha').get_parameter_value().double_value
            self.ema_x = (live_alpha * cx) + ((1.0 - live_alpha) * self.ema_x)
            self.ema_y = (live_alpha * cy) + ((1.0 - live_alpha) * self.ema_y)

    def control_loop(self):
        current_time = time.time()
        dt = 0.016

        # Variables for Foxglove plotting
        error_x = 0.0
        error_y = 0.0

        # 1. Check for timeout (1.5 seconds)
        if current_time - self.last_msg_time > 0.5 and not self.initial_pos_set:
            if not self.target_lost:
                self.first_frame = True
                self.get_logger().info("Target lost! Homing physical joints to front center (0.0).")
                self.target_lost = True
            
            self.initial_pos_set = True
            # Hard wipe the PID memory so it doesn't jerk when a new target appears
            self.int_x = 0.0
            self.int_y = 0.0
            self.prev_err_x = 0.0
            self.prev_err_y = 0.0
            
            # Reset pixel profilers to center so the S-Curve starts fresh
            self.ema_x = self.center_x
            self.ema_y = self.center_y

        else:
            # Normal Tracking Mode
            active_target_x = self.ema_x
            active_target_y = self.ema_y
            self.initial_pos_set = False

            live_kp_pan = self.get_parameter('kp_pan').get_parameter_value().double_value
            live_ki_pan = self.get_parameter('ki_pan').get_parameter_value().double_value
            live_kd_pan = self.get_parameter('kd_pan').get_parameter_value().double_value
            live_max_int_pan = self.get_parameter('max_int_pan').get_parameter_value().double_value

            live_kp_tilt = self.get_parameter('kp_tilt').get_parameter_value().double_value
            live_ki_tilt = self.get_parameter('ki_tilt').get_parameter_value().double_value
            live_kd_tilt = self.get_parameter('kd_tilt').get_parameter_value().double_value
            live_max_int_tilt = self.get_parameter('max_int_tilt').get_parameter_value().double_value
            live_warmup_factor = self.get_parameter('warmup_factor').get_parameter_value().double_value
            live_alpha = self.get_parameter('alpha').get_parameter_value().double_value

            time_since_detect = current_time - self.first_detect_time
            warmup_duration = 1.0  # Durée de l'échauffement en secondes

            if time_since_detect < warmup_duration:
                
                live_kp_pan *= live_warmup_factor
                live_kp_tilt *= live_warmup_factor
                live_kd_pan *= live_warmup_factor
                live_kd_tilt *= live_warmup_factor
                
                live_ki_pan = 0.0
                live_ki_tilt = 0.0

            
            error_x = self.center_x - active_target_x
            error_y = active_target_y - self.center_y

            
            self.int_x += error_x * dt
            self.int_x = max(-live_max_int_pan, min(live_max_int_pan, self.int_x))
            raw_deriv_x = (error_x - self.prev_err_x) / dt

            self.filtered_deriv_x = (0.3 * raw_deriv_x) + (0.7 * self.filtered_deriv_x)


            pan_speed_command = (live_kp_pan * error_x) + (live_ki_pan * self.int_x) + (live_kd_pan * self.filtered_deriv_x)
            self.prev_err_x = error_x

            # --- CALCUL PID TILT (Y) - MODE VELOCITY ---
            self.int_y += error_y * dt
            self.int_y = max(-live_max_int_tilt, min(live_max_int_tilt, self.int_y))
            raw_deriv_y = (error_y - self.prev_err_y) / dt
            
            self.filtered_deriv_y = (0.3 * raw_deriv_y) + (0.7 * self.filtered_deriv_y)

            tilt_speed_command = (live_kp_tilt * error_y) + (live_ki_tilt * self.int_y) + (live_kd_tilt * self.filtered_deriv_y)
            self.prev_err_y = error_y

            # Le Pan reçoit directement sa commande de vitesse (pas d'accumulation d'angle)
            self.current_pan = pan_speed_command  
            
            # The tilt receives the PID output directly as a velocity command.
            self.current_tilt = tilt_speed_command

        # --- 3. Normalization & Publish JointState ---
        
        max_pan_speed = 5.0
        max_tilt_speed = 5.0
        max_error = 320.0
        
        norm_pan_vel = self.current_pan / max_pan_speed
        norm_tilt_vel = self.current_tilt / max_tilt_speed
        norm_err_x = error_x / max_error
        norm_err_y = error_y / max_error

        # A. ORIGINAL MESSAGE: Real physical values for Isaac Sim control
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = ['pan_joint', 'tilt_joint']
        joint_msg.velocity = [float(self.current_pan), float(self.current_tilt)]
        joint_msg.position = [0.0, 0.0]  # Kept clean for the physics engine
        joint_msg.effort = [0.0, 0.0]  # Kept clean for the physics engine

        self.joint_pub.publish(joint_msg)

'''
        # B. FOXGLOVE MESSAGE: Strictly normalized values for visual graphing
        foxglove_msg = JointState()
        foxglove_msg.header.stamp = joint_msg.header.stamp
        foxglove_msg.name = ['pan_joint', 'tilt_joint']
        
        # Velocity field holds normalized velocity
        foxglove_msg.velocity = [float(norm_pan_vel), float(norm_tilt_vel)]
                
        # Effort field holds normalized pixel error
        foxglove_msg.effort = [float(norm_err_x), float(norm_err_y)]

        self.foxglove_pub.publish(foxglove_msg)

'''

def main(args=None):
    rclpy.init(args=args)
    node = IsaacPIDControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()