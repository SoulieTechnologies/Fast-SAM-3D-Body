# Orca hand teleop

Retarget SAM3D hand keypoints onto the Orca robotic hand with an ACADOS MPC.

```
cosmik_hand_demo.py  --emit-hand-port 8092         (perception, in repo root)
    │  TCP: [>I frame][21×3 float32]  = 256 B, wrist-relative right-hand keypoints
    ▼
retarget_mpc.py / hand_teleop_node.py              (acados env)
    palm alignment + human→Orca scale  →  fingertip MPC  →  JointState (rad, URDF names)
    │  ROS 2: /orca/joint_states_target
    ▼
orca_hand_driver_node.py                           (orca_core env)
    joint map (URDF→orca names, rad→deg, sign/offset)  →  vel clamp  →  OrcaHand.set_joint_pos
```

## Files

| File | What it does |
|---|---|
| `retarget_mpc.py` | The retargeting core + a **viser** UI: red = target tips, green = URDF tips (FK), per-finger sliders to calibrate the tip offsets live. `--replay goliath70_3d.npy` or `--listen host:port`. |
| `hand_teleop_node.py` | **ROS 2 node** for the real hand. Reuses `retarget_mpc`'s MPC; adds safety layers (startup ramp, velocity clamp, stale→hold, release→neutral, solver-fail→hold). `--no-ros` for a dry run. |
| `orca_hand_driver_node.py` | **ROS 2 driver** for the physical hand: subscribes the JointState, maps URDF→`orca_core` joints via the yaml, velocity-clamps, writes `OrcaHand.set_joint_pos()`. Needs only `rclpy` + `orca_core` (no acados). `--dry` for hardware-less testing. |
| `joint_map_v1_right.yaml` | URDF CAD-hash names → `orca_core` names, with per-joint `sign` / `offset_deg` (⚠ abduction signs unverified — see bring-up). |
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
`/orca/joint_states_target`) in **radians with the URDF joint names** — leave its
`--joint-map` unset; the unit/name/sign conversion lives entirely in the driver
(`joint_map_v1_right.yaml`, built by matching URDF limits against `orca_core`
`joint_roms`). Driver safety: first command ramps from the **measured** pose,
per-tick `--vmax-deg` clamp, ROM clip, NaN/unmapped skip (wrist never commanded),
targets stale > `--idle` → ramp to config neutral, shutdown → torque off.

### Bring-up procedure (first time on the physical hand)

1. **Calibrate once** (orca_core env): `python orca_core/scripts/calibrate.py <model_path>`
   — writes `calibration.yaml`; then sanity-check with `scripts/neutral.py`.
2. **Verify joint directions**: the `*_abd` signs in `joint_map_v1_right.yaml` cannot be
   inferred offline (symmetric ROMs). Move each joint with `orca_core/scripts/slider_joint.py`
   and compare with the URDF direction in the viser view (`retarget_mpc.py --replay ...`).
   Flip `sign:` / tweak `offset_deg:` in the yaml as needed.
3. **Dry-run the driver** (no hardware): `python orca_hand_driver_node.py --dry --no-ros`
   feeds a synthetic flexion wave through the full mapping+clamp path and prints the
   deg commands. With ROS: `--dry` alone subscribes the real topic but only prints.
4. **Full chain**, 3 terminals:
   ```bash
   # ① perception (GPU env, repo root)
   python cosmik_hand_demo.py --emit-hand-port 8092 ...
   # ② MPC teleop node (acados env + ROS 2 sourced)
   python hand_teleop_node.py --listen localhost:8092
   # ③ hand driver (orca_core env + ROS 2 sourced) — start with a LOW vmax
   python orca_hand_driver_node.py --model <path/to/orcahand_v1_right> --vmax-deg 60
   ```
   Keep a hand on the e-stop / USB cable for the first run; raise `--vmax-deg`
   (default 200) once the motion looks right.
