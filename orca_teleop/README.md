# Robot-hand teleop (Orca / Sharpa Wave)

Retarget SAM3D hand keypoints onto a robot hand with an ACADOS MPC. The hand
is selected with `--hand {orca,sharpa}` (see `HANDS` in `retarget_mpc.py` —
URDF, frame names, tip offsets, driver topic and joint order per hand).

```
cosmik_hand_demo.py  --emit-hand-port 8092         (perception, in repo root)
    │  TCP: [>I frame][21×3 float32]  = 256 B, wrist-relative right-hand keypoints
    ▼
retarget_mpc.py / hand_teleop_node.py              (acados env)
    palm alignment + human→robot scale  →  fingertip MPC  →  JointState
    │
    ├─ --hand orca  → /orca/joint_states_target (rad, URDF names)
    │       ▼
    │  orca_hand_driver_node.py                    (orca_core env)
    │      joint map yaml (URDF→orca names, rad→deg, sign/offset)
    │      → vel clamp → OrcaHand.set_joint_pos
    │
    └─ --hand sharpa → wave/right/joint_commands (rad, SDK 22-joint order)
            ▼
       Sharpa SDK's own ROS 2 bridge (/opt/sharpa-wave-sdk/sample/ROS/
       wave_ros_server.py) — no custom driver needed
```

## Files

| File | What it does |
|---|---|
| `retarget_mpc.py` | The retargeting core + a **viser** UI: red = target tips, green = URDF tips (FK), per-finger sliders to calibrate the tip offsets live. `--replay goliath70_3d.npy` or `--listen host:port`. |
| `hand_teleop_node.py` | **ROS 2 node** for the real hand. Reuses `retarget_mpc`'s MPC; adds safety layers (startup ramp, velocity clamp, stale→hold, release→neutral, solver-fail→hold). `--no-ros` for a dry run. |
| `orca_hand_driver_node.py` | **ROS 2 driver** for the physical hand: subscribes the JointState, maps URDF→`orca_core` joints via the yaml, velocity-clamps, writes `OrcaHand.set_joint_pos()`. Needs only `rclpy` + `orca_core` (no acados). `--dry` for hardware-less testing. |
| `joint_map_v1_right.yaml` | URDF CAD-hash names → `orca_core` names, with per-joint `sign` / `offset_deg` (⚠ abduction signs unverified — see bring-up). |
| `orcahand/` | Vendored URDF + STL meshes (right hand, 17 revolute joints; the `to_TopTower` wrist joint is locked → nq=16). |
| `sharpawave/` | Vendored Sharpa Wave right-hand URDF + meshes (from sharpa-robotics/sharpa-urdf-usd-xml, `package://` paths rewritten relative; 22 revolute joints, real `*_fingertip` frames → tip offsets default to 0). |

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

## Driving the Sharpa Wave

The Wave SDK (install `sharpa-wave-sdk_<v>_amd64.deb` from
github.com/sharpa-robotics/sharpa-wave-sdk/releases → `/opt/sharpa-wave-sdk/`)
ships its own ROS 2 bridge, so there is no custom driver:

```bash
# ① its bridge (system python 3.10 + ROS 2 sourced; needs cv_bridge):
python3 /opt/sharpa-wave-sdk/sample/ROS/wave_ros_server.py
# ② our teleop node (acados env + ROS 2 sourced):
python hand_teleop_node.py --listen localhost:8092 --hand sharpa
```

Notes:
- The bridge indexes `msg.position` BY POSITION (ignores names) — the node
  reorders q into the SDK's 22-joint order via `publish_order` in the config.
- No calibration step: the SDK works in radians with the URDF conventions,
  and `pin.neutral` = all-zeros matches the SDK sample's init pose.
- The bridge does not set speed/current coefficients; run the SDK python
  sample once (or Sharpa Pilot) if you want `set_speed_coeff`/`set_current_coeff`
  limits configured, and start the first session with a LOW `--vmax` (e.g. 1.0).
- First tip-offset pass should be unnecessary (real fingertip frames), but the
  viser sliders (`retarget_mpc.py --hand sharpa --replay ...`) still work if
  the tips need a tweak.
