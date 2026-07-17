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
    │                 or --emit-q 8093 (TCP: rad, joint_map key order)
    │       ▼
    │  orca_hand_driver_node.py                    (orca_core env/venv)
    │      ROS topic OR --listen-q host:8093 (no ROS anywhere)
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
| `orca_hand_driver_node.py` | **Driver** for the physical hand: consumes the JointState topic (ROS mode) or the `--emit-q` TCP stream (`--listen-q host:port`, no ROS — the mode to use when the hand's machine has no rclpy, e.g. the Mac), maps URDF→`orca_core` joints via the yaml, velocity-clamps, writes `OrcaHand.set_joint_pos()`. Needs only `orca_core` (+ `rclpy` for ROS mode). `--dry` = no orca_core at all; `--mock` = orca_core's simulated motors with the REAL config+calibration (full rehearsal). |
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

### Self-collision (sharpa)

The MPC carries slacked sphere constraints between phalanges (`collision_spheres`
/`collision_pairs` in the hand config; orca has none yet). Spheres sit at the
midpoint of each proximal/middle phalanx, radii from the collision-STL
half-widths minus ~1 mm — adjacent MCPs are only 20.5 mm apart, so full radii
would bind at rest. **Fingertip contact stays free by construction**: the four
fingers' distal segments carry no sphere (pinching, fingers held together),
and no thumb-vs-distal pair exists (tip-to-tip opposition); with abduction at
the MCP, constraining PP/MP still blocks actual finger crossing (verified:
scissor pose at AA limits violates by 12 mm, tightest pinch keeps +4 mm on
every constrained pair). The constraints are soft (L1 1e2 / L2 1e5 on m²
violations) so the solver can never go infeasible. Flags on both
`retarget_mpc.py` and `hand_teleop_node.py`: `--no-self-collision`,
`--col-margin <m>` (extra clearance, default 0 = allow light touch, block
interpenetration); viser can draw the spheres with `--show-collision`.

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
3. **Rehearse without the hand**: `python orca_hand_driver_node.py --mock --no-ros`
   feeds a synthetic flexion wave through the full mapping+clamp path AND the real
   config/calibration (simulated motors); `--dry --no-ros` does the same without
   orca_core installed. With ROS: `--dry` alone subscribes the real topic but only prints.
4. **Full chain (TCP mode — hand plugged into a machine without ROS, e.g. the Mac)**:
   ```bash
   # ① perception (crslab, GPU env, repo root)
   python cosmik_hand_demo.py --emit-hand-port 8092 ...
   # ② MPC teleop node (crslab, acados env) — no ROS, mirror q over TCP
   python hand_teleop_node.py --listen localhost:8092 --no-ros --emit-q 8093
   # ③ hand driver (hand's machine, orca_core venv) — start with a LOW vmax
   python orca_hand_driver_node.py --listen-q crslab:8093 --vmax-deg 60
   ```
   The TCP stream carries positions only — index alignment with the driver is
   guaranteed because both ends read `joint_map_v1_right.yaml`
   (`_orca_publish_order()` in `retarget_mpc.py` = the yaml's key order).
   ROS variant: source ROS 2 in ② and ③, drop `--no-ros`/`--emit-q`/`--listen-q`.
   Keep a hand on the e-stop / USB cable for the first run; raise `--vmax-deg`
   (default 200) once the motion looks right.

## Driving the Sharpa Wave

The Wave SDK (install `sharpa-wave-sdk_<v>_amd64.deb` from
github.com/sharpa-robotics/sharpa-wave-sdk/releases → `/opt/sharpa-wave-sdk/`)
ships its own ROS 2 bridge, so there is no custom driver:

```bash
# ① the joints-only bridge (system python 3.10 + ROS 2 sourced) — trimmed
#    from the SDK sample: its wave_ros_server.py needs cv_bridge (numpy-1-built
#    on Humble, breaks with system numpy 2.x) for tactile topics we don't use:
python3 sharpa_ros_bridge.py
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
