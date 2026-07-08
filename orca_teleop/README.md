# Orca hand teleop

Retarget SAM3D hand keypoints onto the Orca robotic hand with an ACADOS MPC.

```
cosmik_hand_demo.py  --emit-hand-port 8092         (perception, in repo root)
    │  TCP: [>I frame][21×3 float32]  = 256 B, wrist-relative right-hand keypoints
    ▼
retarget_mpc.py / hand_teleop_node.py
    palm alignment + human→Orca scale  →  fingertip MPC (acados)  →  q command
```

## Files

| File | What it does |
|---|---|
| `retarget_mpc.py` | The retargeting core + a **viser** UI: red = target tips, green = URDF tips (FK), per-finger sliders to calibrate the tip offsets live. `--replay goliath70_3d.npy` or `--listen host:port`. |
| `hand_teleop_node.py` | **ROS 2 node** for the real hand. Reuses `retarget_mpc`'s MPC; adds safety layers (startup ramp, velocity clamp, stale→hold, release→neutral, solver-fail→hold). `--no-ros` for a dry run. |
| `orcahand/` | Vendored URDF + STL meshes (right hand, 17 revolute joints; the `to_TopTower` wrist joint is locked → nq=16). |

## Run

Viser calibration / replay (needs the `acados` env):

```bash
export ACADOS_SOURCE_DIR=~/code/comfi-examples-hands/acados
export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
export ACADOS_EXT_FUN_COMPILE_FLAGS=-O1
python retarget_mpc.py --listen localhost:8092          # live from cosmik_hand_demo
python retarget_mpc.py --replay path/to/goliath70_3d.npy
```

ROS 2 node (source ROS 2 first so `rclpy` is importable):

```bash
python hand_teleop_node.py --listen localhost:8092                 # publishes JointState
python hand_teleop_node.py --listen localhost:8092 --no-ros        # dry run, no rclpy
```

## Key parameters

- Cost weights (fingertips are near end-effectors): `--w-tip 50 --w-mid 2 --w-mcp 0.1`.
- Regularization: `--w-dq 1e-3 --w-u 1e-4`; horizon `--N 8 --dt 0.04`.
- Tip offsets default to `TIP_OFFSETS_CALIB` (slider-calibrated, local distal frame, metres).
- Node safety: `--vmax` (rad/s, also the startup ramp), `--stale` (hold), `--release` (drift to neutral).

## Driving the real hand

`hand_teleop_node.py` publishes a `sensor_msgs/JointState` on `--topic` (default
`/orca/joint_states_target`) using the **URDF joint names**. Map them to the driver's
names with `--joint-map map.yaml` (`{urdf_joint_name: driver_joint_name}`). The exact
sink (an `orca_core` Python bridge vs. an existing ROS driver topic) is still TBD.
