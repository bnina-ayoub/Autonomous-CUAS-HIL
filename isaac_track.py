#!/usr/bin/env python3
import argparse
import json
import os
import sys
import cv2
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from loguru import logger
import math
import torch.nn.functional as F
# -----------------------------------------------------------------------------
# 1. Environment & Path Configuration
# -----------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)  # Priority to custom aerotrack package

BYTETRACK_SUBMODULE = os.path.join(ROOT, "third_party", "ByteTrack")
if os.path.exists(BYTETRACK_SUBMODULE) and BYTETRACK_SUBMODULE not in sys.path:
    sys.path.append(BYTETRACK_SUBMODULE)

sys.path.insert(0, "/home/bninaos/UAV-Tracking-Project/ros2_ws/AeroTrack")
# -----------------------------------------------------------------------------
# 2. AeroTrack Imports
# -----------------------------------------------------------------------------
from aerotrack.data.data_augment import preproc
from aerotrack.exp import get_exp
from aerotrack.utils import fuse_model, postprocess
from aerotrack.utils.visualize import plot_tracking
from aerotrack.tracker.byte_tracker import BYTETracker


def make_parser():
    parser = argparse.ArgumentParser("AeroTrack Isaac Sim Dual-TRT Node")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=os.path.join(ROOT, "AeroTrack", "exps", "aerotrack_proposed.py"),
        type=str,
        help="experiment description file",
    )
    parser.add_argument(
        "-c",
        "--ckpt",
        default=os.path.join(ROOT, "AeroTrack", "weights", "baseline_model.pth"),
        type=str,
        help="checkpoint for PyTorch fallback/head weights",
    )
    parser.add_argument("-n", "--name", default=None, type=str, help="model name")
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model: cpu or gpu")
    parser.add_argument("--conf", default=0.1, type=float, help="test conf threshold")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=640, type=int, help="test img size")
    parser.add_argument("--fps", default=60, type=int, help="frame rate (fps)")
    parser.add_argument("--fp16", dest="fp16", default=True, action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for PyTorch testing.")

    # TensorRT Configuration
    parser.add_argument("--trt", dest="trt", default=True, action="store_true", help="Enable Dual TensorRT Engines.")
    parser.add_argument("--trt_early", default="early_stage_fp16.trt", type=str, help="Path to early exit TRT engine")
    parser.add_argument("--trt_deep", default="deep_stage_fp16.trt", type=str, help="Path to deep stage TRT engine")

    # Tracking args
    parser.add_argument("--track_thresh", type=float, default=0.2, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=50, help="the frames for keeping lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.97, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for unusual aspect ratios")
    parser.add_argument('--min_box_area', type=int, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    
    # AeroTrack Dynamic Routing & Distance Metric
    parser.add_argument("--distance", type=str, default="nwd", choices=["nwd", "iou"], help="distance metric for tracking")
    parser.add_argument("--early_exit", dest="early_exit", default=False, action="store_true", help="Enable dynamic early exit routing.")
    return parser


def ros_image_to_bgr8(msg):
    """Convert a ROS Image message to an OpenCV BGR image without cv_bridge."""
    if msg.encoding not in {"bgr8", "rgb8", "mono8"}:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    data = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding == "mono8":
        image = data.reshape((msg.height, msg.step))[:, : msg.width]
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    channels = 3
    row_stride = msg.step
    image = data.reshape((msg.height, row_stride))[:, : msg.width * channels]
    image = image.reshape((msg.height, msg.width, channels))

    if msg.encoding == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image


class IsaacTrackerNode(Node):
    def __init__(self, args, exp):
        super().__init__('isaac_tracker_node')
        
        self.args = args
        self.exp = exp
        self.exp.early_exit_enabled = args.early_exit

        if args.device == "gpu":
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # -------------------------------------------------------------
        # Model Initialization: Dual TRT vs PyTorch Fallback
        # -------------------------------------------------------------
        if args.trt:
            logger.info("⚡ Initializing Dual-Engine Hardware Accelerated TensorRT Pipeline...")
            from tools.dual_model_loader import DualTRTModel
            
            original_model = exp.get_model()
            if os.path.exists(args.ckpt):
                logger.info(f"Loading gating/head weights from checkpoint: {args.ckpt}")
                ckpt = torch.load(args.ckpt, map_location="cpu")
                state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

                # --- RESIZE SWIN TRANSFORMER BIAS TABLES ---
                model_state = original_model.state_dict()
                for k in list(state_dict.keys()):
                    if 'relative_position_bias_table' in k and k in model_state:
                        ckpt_shape = state_dict[k].shape
                        model_shape = model_state[k].shape
                        
                        if ckpt_shape != model_shape:
                            num_heads = ckpt_shape[1]
                            S_ckpt = int(math.sqrt(ckpt_shape[0]))
                            S_model = int(math.sqrt(model_shape[0]))
                            
                            table = state_dict[k].view(S_ckpt, S_ckpt, num_heads).permute(2, 0, 1).unsqueeze(0)
                            table = F.interpolate(table, size=(S_model, S_model), mode='bicubic', align_corners=False)
                            state_dict[k] = table.squeeze(0).permute(1, 2, 0).view(-1, num_heads)
                # -------------------------------------------

                original_model.load_state_dict(state_dict, strict=False)               

            self.model = DualTRTModel(
                args.trt_early,
                args.trt_deep,
                num_classes=exp.num_classes,
                img_hw=exp.test_size,
                gate=original_model.decision_gate,
                early_exit_enabled=exp.early_exit_enabled
            )
            self.model.head = original_model.head
            self.model.head.decode_in_inference = False
            self.model.to(self.device)

            # Pre-warm both early and deep paths to prevent the initial cold-start latency spike
            logger.info("⚡ Pre-warming Dual TRT engines...")
            dummy_input = torch.zeros(1, 3, exp.test_size[0], exp.test_size[1], dtype=torch.half if args.fp16 else torch.float).to(self.device)
            with torch.no_grad():
                original_flag = self.model.early_exit_enabled
                self.model.early_exit_enabled = False  # Forces deep branch execution
                for _ in range(3):
                    _ = self.model(dummy_input)
                self.model.early_exit_enabled = original_flag
                torch.cuda.synchronize()
            logger.info("✅ TensorRT engines initialized and warmed up.")
        else:
            logger.info("Loading PyTorch Native Model...")
            self.model = exp.get_model()
            self.model.to(self.device)
            self.model.eval()

            if os.path.exists(args.ckpt):
                logger.info(f"Loading checkpoint from: {args.ckpt}")
                ckpt = torch.load(args.ckpt, map_location="cpu")
                state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                self.model.load_state_dict(state_dict, strict=False)

            if args.fuse:
                logger.info("Fusing model layers...")
                self.model = fuse_model(self.model)

            if args.fp16:
                self.model.half()

        # -------------------------------------------------------------
        # Tracker Setup
        # -------------------------------------------------------------
        self.tracker = BYTETracker(args, frame_rate=args.fps)
        self.frame_id = 0

        self.publisher_ = self.create_publisher(String, 'anti_uav/output_track', 10)
        self.subscription = self.create_subscription(
            Image,
            '/anti_uav/input_track',  
            self.image_callback,
            10
        )

        self.latest_target = {"id": 0, "cx": 0, "cy": 0, "count": 0}
        self.latest_frame_id = 0
        self.latest_time_sec = 0
        self.latest_time_nanosec = 0
        self.data_ready = False
        
        # Pacing timer: 60 Hz output loop
        timer_period = 1.0 / float(args.fps)
        self.timer = self.create_timer(timer_period, self.timer_callback)
        logger.info("ROS 2 Dual-TRT Tracking Node initialized. Subscribed to /anti_uav/input_track")

    def timer_callback(self):
        """Strict timer loop for sending centroid telemetry."""
        if self.data_ready: 
            msg = String()
            msg.data = json.dumps({
                "frame_id": int(self.latest_frame_id),
                "t_sec": int(self.latest_time_sec),
                "t_ns": int(self.latest_time_nanosec),
                "cx": self.latest_target["cx"],
                "cy": self.latest_target["cy"],
                "count": self.latest_target["count"]
            }, separators=(',', ':'))
            
            self.publisher_.publish(msg)
            self.data_ready = False

    def image_callback(self, msg):
        self.frame_id += 1
        try:
            img = ros_image_to_bgr8(msg)
        except Exception as e:
            logger.error(f"Image conversion failed: {str(e)}")
            return

        img_info = {"id": self.frame_id, "height": img.shape[0], "width": img.shape[1]}

        # Preprocessing (Resizing, padding, channel transpose)
        res_img, ratio = preproc(img, self.exp.test_size)
        res_img = torch.from_numpy(res_img).unsqueeze(0).to(self.device)
        if self.args.fp16:
            res_img = res_img.half()
        else:
            res_img = res_img.float()

        # Neural Forward & Post-Processing
        with torch.no_grad():
            outputs = self.model(res_img)
            outputs = postprocess(outputs, self.exp.num_classes, self.args.conf, self.args.nms)

        online_tlwhs = []
        online_ids = []

        if outputs is not None and len(outputs) > 0 and outputs[0] is not None:
            online_targets = self.tracker.update(outputs[0], [img_info['height'], img_info['width']], self.exp.test_size)
            
            sum_cx = 0.0
            sum_cy = 0.0
            valid_uav_count = 0

            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                
                # Filter noise and ensure the track is active
                if tlwh[2] * tlwh[3] > self.args.min_box_area and t.is_activated:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    
                    # Compute center using tlwh
                    drone_center_x = tlwh[0] + (tlwh[2] / 2.0)
                    drone_center_y = tlwh[1] + (tlwh[3] / 2.0)
                    
                    sum_cx += drone_center_x
                    sum_cy += drone_center_y
                    valid_uav_count += 1
            
            if valid_uav_count > 0:
                raw_cx = sum_cx / valid_uav_count
                raw_cy = sum_cy / valid_uav_count
                
                if self.latest_target["count"] > 0:
                    prev_cx = self.latest_target["cx"]
                    prev_cy = self.latest_target["cy"]
                    
                    raw_delta_x = raw_cx - prev_cx
                    raw_delta_y = raw_cy - prev_cy
                    
                        
                    # 2. SLEW-RATE LIMITER: Clamp massive teleportation spikes
                    max_jump = 35.0
                    delta_x = max(-max_jump, min(max_jump, raw_delta_x))
                    delta_y = max(-max_jump, min(max_jump, raw_delta_y))
                    
                    swarm_cx = prev_cx + delta_x
                    swarm_cy = prev_cy + delta_y
                else:
                    swarm_cx = raw_cx
                    swarm_cy = raw_cy
                
                self.latest_target = {
                    "id": 999,
                    "cx": float(swarm_cx),
                    "cy": float(swarm_cy),
                    "count": valid_uav_count
                }
                
                self.latest_frame_id = self.frame_id
                now = self.get_clock().now().seconds_nanoseconds()
                self.latest_time_sec = now[0]
                self.latest_time_nanosec = now[1]
                self.data_ready = True

            online_im = plot_tracking(
                img,
                online_tlwhs,
                online_ids,
                frame_id=self.frame_id,
                fps=float(self.args.fps),
            )
        else:
            online_im = img

        cv2.imshow("Isaac Sim Live Tracking", online_im)
        cv2.waitKey(1)


def main():
    args = make_parser().parse_args()
    if not os.path.isfile(args.exp_file):
        raise FileNotFoundError(f"Experiment file not found: {args.exp_file}")

    exp = get_exp(args.exp_file, args.name)
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    print("DEBUG: Starting Dual-TRT Isaac Sim Node Initialization...")
    try:
        rclpy.init(args=None)
        node = IsaacTrackerNode(args, exp)
        print("DEBUG: Node running with TensorRT acceleration, spinning...")
        rclpy.spin(node)
    except Exception as e:
        print(f"DEBUG: CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("DEBUG: Shutting down ROS 2 node.")
        rclpy.shutdown()


if __name__ == "__main__":
    main()