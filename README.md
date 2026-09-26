

# Autonomous C-UAS Hardware-in-the-Loop (HIL) Simulation

<img width="2540" height="1440" alt="frame_000039" src="https://github.com/user-attachments/assets/ed0c6273-1881-48b2-9d87-7b29abdd126c" />
A ROS 2 and NVIDIA Isaac Sim digital twin engineered for closed-loop visual servoing and real-time Counter-Unmanned Aerial Systems (C-UAS) swarm tracking.

This repository bridges deterministic kinematic actuation with the **AeroTrack** deep learning perception model. By deploying a photorealistic, physically accurate simulation environment, the system validates sub-10ms tracking permanence and non-linear target acquisition prior to physical hardware deployment.

---

## System Performance

- **Throughput & Latency** — Dual-engine TensorRT (FP16) deployment achieves **119.8 FPS** with an end-to-end pipeline latency of **8.35 ms**.
- **Association Stability** — Replaces discrete IoU with Normalized Wasserstein Distance (NWD) to maintain target identity during extreme multi-agent swarm intersections, yielding an **85.1% IDF1** score.
- **Kinematic Actuation** — Implements an asymmetrical Proportional-Integral-Derivative (PID) control loop, utilizing velocity control for continuous pan tracking and position control for bounded tilt elevation.
- **Signal Conditioning** — Neutralizes bounding box micro-jitter and prevents actuator resonance using an Exponential Moving Average (EMA) filter (α = 0.60).

---

## Repository Structure

The project files are mapped directly in the root directory to facilitate immediate ROS 2 integration and Isaac Sim execution.

### Submodules & Workspaces

| Path | Description |
|---|---|
| `AeroTrack/` | Git submodule containing the core P3 early-exit tracking architecture, NWD association logic, and TensorRT export utilities. |
| `src/` | ROS 2 workspace directory containing the custom nodes and messaging interfaces required for inter-process communication. |

### Core Simulation Scripts

| Script | Description |
|---|---|
| `isaac_track.py` | The primary visual servoing bridge. Captures synthetic camera feeds from Isaac Sim, executes AeroTrack inference, calculates the collective swarm centroid, and publishes spatial coordinates to the ROS 2 network at 60 Hz. |
| `isaac_motor_sim.py` | The actuation node. Subscribes to the perception centroid, processes the spatial error through the decoupled PID control laws, and applies continuous joint velocity/position commands to the URDF Pan-Tilt model. |
| `swarm_diamond_bullseye_formation.py` | Parametric swarm generator. Spawns microscopic kamikaze UAVs executing dynamic, multi-agent orbital and diamond formations to stress-test the NWD association loop. |

---

## Build & Installation

### 1. Clone the repository with submodules

```bash
git clone --recursive https://github.com/<YourUsername>/Autonomous-CUAS-HIL.git
cd Autonomous-CUAS-HIL
```

### 2. Build the ROS 2 Workspace

```bash
colcon build --symlink-install
source install/setup.bash
```

### 3. TensorRT Dual-Engine Setup

To achieve zero-copy shared memory execution and dynamic early-exit routing, the baseline model is split into two distinct ONNX graphs and compiled into FP16 TensorRT engines:

- `early_stage_fp16.trt` — Executes the ECA-backbone and P3 extraction.
- `deep_stage_fp16.trt` — Executes the deeper PAFPN layers when triggered.

TensorRT engines are strictly bound to the GPU architecture they were compiled on. If you encounter a hardware mismatch error during execution, you must rebuild the engines locally from the ONNX graphs using the provided export tools:

```bash
# Example rebuild command for local GPU architectures
trtexec --onnx=early_stage.onnx --saveEngine=early_stage_fp16.trt --fp16
trtexec --onnx=deep_stage.onnx --saveEngine=deep_stage_fp16.trt --fp16
```
---

## Running the Simulation

Initialize the complete Hardware-in-the-Loop simulation pipeline using the ROS 2 launch system, followed by the Isaac Sim environment.

### 1. Launch the ROS 2 Tracking & Control Nodes

Start the perception, PID control, and visualization nodes via the bringup launch file:

```bash
ros2 launch antiuav_bringup hil_tracking.launch.py
```

### 2. Start the Isaac Sim Digital Twin

1. Open NVIDIA Isaac Sim and load your counter-UAV scenario.
2. Press **Play** to start the simulation.

> **Note:** The ROS 2 nodes will wait in a standby state until the simulation starts. Once Play is pressed, the synthetic camera begins publishing frames to the network, triggering the AeroTrack inference and closed-loop visual servoing pipeline.
