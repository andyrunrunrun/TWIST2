# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TWIST2 is a teleoperated humanoid robot control and motion data collection system built on NVIDIA Isaac Gym. It enables real-time teleoperation using PICO VR headsets with motion retargeting capabilities, supporting both simulation (Isaac Gym/MuJoCo) and real robot deployment on Unitree G1.

**Architecture**: Two-tier hierarchical control system
- **High-level**: Motion retargeting (GMR) and teleop interface (PICO VR)
- **Low-level**: RL policy for joint-level control

**Training Framework**: Teacher-Student knowledge distillation
- **Teacher** (`g1_priv_mimic`): Uses privileged motion information
- **Student** (`g1_stu_future`): Deployable policy with limited observation space

## Conda Environments

This project uses **two** conda environments:

| Environment | Python Version | Purpose |
|-------------|---------------|---------|
| `twist2` | 3.8 | Main: training, deployment, teleop data collection |
| `gmr` | 3.10 | Motion retargeting (GMR), PICO teleop |

The split exists because Isaac Gym requires Python 3.8, while newer MuJoCo requires Python 3.10+.

## Essential Commands

### Training

**Single GPU:**
```bash
# Student training (main task)
bash train.sh <exptid> cuda:0
# Example: bash train.sh 0205_twist2 cuda:0
```

**Multi-GPU DDP:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  legged_gym/legged_gym/scripts/train.py \
  --task g1_stu_future --proj_name g1_stu_future --exptid <exptid>
```

**Teacher Training (privileged):**
```bash
python legged_gym/legged_gym/scripts/train.py \
  --task g1_priv_mimic --proj_name g1_priv_mimic --exptid <exptid> \
  --device cuda:0 --num_envs 4096
```

**Student with Distillation:**
```bash
python legged_gym/legged_gym/scripts/train.py \
  --task g1_stu_future --proj_name g1_stu_future --exptid <exptid> \
  --teacher_exptid <teacher_exptid> --teacher_checkpoint -1
```

### Memory Optimization

For large datasets or limited GPU memory:
```bash
# Memory optimization for large motion datasets
--motion.storage_dtype float16 --gpu_cache 8.0
--lazy_load --cpu_cache 100.0  # for very large datasets
```

### Export and Deployment

**Export policy to ONNX:**
```bash
bash to_onnx.sh legged_gym/logs/g1_stu_future/<exptid>/model_<iter>.pt
```

**Simulation verification (sim2sim):**
```bash
# Terminal 1: High-level motion server (offline motion)
bash run_motion_server.sh

# Terminal 2: Low-level controller
bash sim2sim.sh
```

**Real robot deployment (sim2real):**
```bash
# Terminal 1: Low-level controller
bash sim2real.sh  # Edit net=... in script first

# Terminal 2: High-level (choose one)
bash run_motion_server.sh  # OR
bash teleop.sh             # PICO VR teleop (activate gmr env)
```

**GUI interface:**
```bash
bash gui.sh
```

### Evaluation

```bash
cd legged_gym/legged_gym/scripts
python play.py --task g1_stu_future --proj_name g1_stu_future --exptid <exptid> \
  --device cuda:0 --num_envs 1 --headless --record_video \
  --motion.motion_file <path_to_yaml>
```

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `legged_gym/envs/` | Environment definitions (`base/` for base classes, `g1/` for G1-specific) |
| `legged_gym/scripts/` | Train (`train.py`), eval (`play.py`), export (`save_onnx.py`) |
| `legged_gym/motion_data_configs/` | YAML configs for motion datasets |
| `rsl_rl/` | PPO algorithm, Actor-Critic modules, motion loaders |
| `pose/` | Motion retargeting (GMR implementation) |
| `deploy_real/` | Real robot deployment scripts (sim2real, teleop) |
| `assets/ckpts/` | Pre-trained ONNX models |
| `assets/example_motions/` | Sample motion data for testing |

## Configuration System

**Environment configs**: `legged_gym/envs/g1/`
- `g1_mimic_priv_config.py` - Teacher (privileged) config
- `g1_mimic_future_config.py` - Student config with future motion prediction

**Motion configs**: `legged_gym/motion_data_configs/*.yaml`
- Define dataset paths, weights, and sampling parameters
- Use `generate_yaml.py` to create new configs

**Key config parameters:**
- `motion.storage_dtype`: `float32` or `float16` (memory savings)
- `num_envs`: Parallel environment count (typically 4096)
- `history_len`: Observation history length (default 10)

## Redis Communication

The system uses Redis for high-level <-> low-level communication:
- High-level (motion server/teleop) publishes target poses
- Low-level (controller) subscribes and executes policy

Setup (one-time):
```bash
sudo apt install redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server
# Edit /etc/redis/redis.conf: bind 0.0.0.0, protected-mode no
sudo systemctl restart redis-server
```

## Checkpoint Structure

Checkpoints are saved to:
```
legged_gym/logs/<task_name>/<exptid>/model_<iteration>.pt
```

For distillation, teacher checkpoints are loaded from:
```
legged_gym/logs/g1_priv_mimic/<teacher_exptid>/model_<checkpoint>.pt
```

## VR Teleop Controls (PICO)

| Controller | Button | Action |
|------------|--------|--------|
| Right | A | Start/pause teleop |
| Left | X | Exit teleop (default pose) |
| Right | Index grip | Close right hand |
| Right | Grip | Open right hand |
| Left | Index grip | Close left hand |
| Left | Grip | Open left hand |
| Left | Axis click | Emergency stop |

## Common Issues

- **Isaac Gym import error**: Ensure conda env `twist2` is active and Isaac Gym is installed
- **Redis connection**: Check `redis-server` is running and config allows connections
- **GPU memory**: Reduce `--num_envs` or use `--motion.storage_dtype float16`
- **Large dataset OOM**: Add `--motion.storage_dtype float16 --lazy_load`
- **Unitree connection**: Verify network interface (edit `net=` in `sim2real.sh`), ping `192.168.123.164`
