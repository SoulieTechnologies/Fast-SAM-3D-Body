# ═══ RERUN DEMO — camera + skeleton + ACADOS retargeting (new) ═══

One command records the camera, shows a **Rerun UI** (video with skeleton overlay +
3D skeleton alone + latency plots + retargeted skeleton) and runs the **ACADOS IK**
retargeting (meshcat on localhost:7000 for the full URDF human).

One-time setup on the machine:
```bash
conda activate <sam3d-env>  && pip install "rerun-sdk>=0.28"
conda activate <acados-env> && pip install "rerun-sdk>=0.28"
cp run_ik_live_rerun.py <comfi-examples_new>/scripts/   # ships in this repo
```

Run everything (both processes, Ctrl+C stops both):
```bash
cd <this-repo>
SAM3D_ENV=<sam3d-env> ACADOS_ENV=<acados-env> COMFI_DIR=~/code/comfi-examples_new \
SOURCE=0 GPU=0 FX=900 bash run_rerun_demo.sh
```
Then open (add `ssh -L <port>:localhost:<port>` for each if remote):
- **Rerun UI**: http://localhost:9090
- **meshcat** (URDF human): http://localhost:7000/static/

Recording lands in `output_rerun_demo/<timestamp>/`: `raw.mp4`, `overlay.mp4`,
`joints_2d.npy`, `joints_3d.npy`, `joints_3d_world.npy`, `timestamps.npy`.
IK process log: `output_rerun_demo/ik.log`.

On a machine with a screen, use `RERUN_MODE=native` for the local Rerun window.
Each process also runs standalone — see the docstrings of `rerun_demo.py` and
`scripts/run_ik_live_rerun.py`.

# ═══ LIVE SAM3D SKELETON DEMO (primary) ═══

Real-time 3D skeleton (body + hands) streamed to your laptop browser. Two ways:

## A. Server processes a recorded video (safest — looks live)
Server (SSH):
```bash
cd ~/code/Fast-SAM-3D-Body && conda activate <sam3d-env>
python stream_demo.py --source test_input/take_01/cam0.mp4 \
  --intrinsics /home/users/theo/code/test_input/cam_params/cam0_intrinsics.npz \
  --gpu 7 --port 8094
```
Laptop browser: **http://clear-antares.tailb614a0.ts.net:8094**  (Tailscale — no tunnel)
or `ssh -L 8094:localhost:8094 …` then http://localhost:8094

## B. LIVE camera on the MacBook → server → browser  (tested, ~16 fps)
Server (SSH):
```bash
cd ~/code/Fast-SAM-3D-Body && conda activate <sam3d-env>
python stream_demo.py --recv-port 8091 --fx 900 --cx 640 --cy 360 --gpu 7 --port 8094
```
MacBook (once: `pip install opencv-python`):
```bash
python stream_client.py --host clear-antares.tailb614a0.ts.net --port 8091 --source 0
```
Laptop browser: **http://clear-antares.tailb614a0.ts.net:8094**

Notes:
- Inference ~16 fps < 30 fps capture → server always processes the LATEST frame (drops
  the backlog) → low latency, ~16 fps effective. This is correct, not a bug.
- Only the MAIN subject (largest bbox + tracking) is drawn — passers-by are ignored.
- Webcam has no calibration → `--fx 900 --cx 640 --cy 360` is a fine 720p approximation;
  tune `--fx` if the skeleton looks slightly off. (cam0 footage → use its real npz.)
- FPS counter is burned into the stream.

---

# Two-Camera → Metric 3D → ACADOS IK — Demo Runbook

Full pipeline: **2 calibrated cameras → SAM3D per view → stereo triangulation (metric
Goliath-70) → gravity alignment → ACADOS MPC IK on a MANO body → meshcat**.

Everything below is **already installed and tested** on `clear-antares`. Follow the
steps in order.

---

## 0. One-time prerequisites (ALREADY DONE — for reference)
- conda env **`acados`** (pinocchio+casadi+acados_template+comfi_examples+meshcat).
- Linux `t_renderer` (tera 0.0.34, glibc-compatible).
- Fixes baked into `comfi-examples_new/scripts/run_ik_acados_mpc_sam3d.py`:
  NaN-marker skipping, `-O1` compile flag, COSMIK head-marker scaling.
- Gravity alignment baked into `Fast-SAM-3D-Body/extract_dualgpu.py`
  (outputs `joints_3d_world.npy`).

---

## 1. Local machine — open the meshcat tunnel
In a terminal **on your laptop**:
```bash
ssh -L 7000:localhost:7000 theo@clear-antares.tailb614a0.ts.net
```
Keep this session open — the IK runs inside it. Later, open
**http://localhost:7000/static/** in your local browser.

## 2. Server — extract + triangulate + align (≈ 3 min)
In the SSH session:
```bash
cd ~/code/Fast-SAM-3D-Body
./demo_extract.sh                       # default = take_01 test pair
#  or:  ./demo_extract.sh /path/cam0.mp4 /path/cam1.mp4
```
Produces `output_dualgpu/joints_3d_world.npy` and stages it for the IK.
(Sanity numbers printed: ~13 fps pair, kept ~2155/2156, reproj ~2.3 px.)

## 3. Server — run the IK + meshcat
```bash
cd ~/code/comfi-examples_new
./run_ik.sh --id demo --task take --keypoints-file joints_world.npy --no-cv-to-ros \
  --start-sample 1 --display --N 10 --w-dq 1e-3 --w-u 1e-4 \
  --urdf-file urdf/human_with_mano.urdf --subject-height 1.75 --subject-weight 70
```
- First launch **compiles the solver ≈ 1 min** (talk over it). Then it prints
  `Meshcat ready -- press Enter to start IK`.
- Open **http://localhost:7000/static/** locally, then press **Enter** in the SSH
  session → the model animates (~25 fps).
- Set `--subject-height`/`--subject-weight` to the demo subject for a better fit.

---

## Knobs (if asked / to tune live)
- **Smoothness ↔ reactivity**: `--N` (horizon), `--w-dq` (velocity reg),
  `--w-u` (accel reg). Smoother: bigger N / w-dq / w-u.
- **Hands more reactive**: add `--w-dq-hand 1e-5`.
- Faster solve (less smooth): `--N 5` or `--N 3`.

## Talking points
- Two cameras remove the **monocular scale ambiguity** → metric skeleton by
  triangulation (not a learned average-human prior).
- **Goliath-70** incl. articulated hands; dual-GPU ~14 fps.
- Model **scaled to the subject** from COSMIK markers; MPC IK retargets to a
  MANO-hand human model, solving joint accelerations in real time.

## Troubleshooting
- `libhpipm.so: cannot open shared object file` → you didn't use `run_ik.sh`
  (it sets `LD_LIBRARY_PATH`). Always launch via `./run_ik.sh`.
- meshcat page blank → tunnel not up, or open the exact URL `http://localhost:7000/static/`.
- Skeleton leaning/sideways → you fed `joints_3d_tri.npy` (cam0 frame). Use
  `joints_world.npy` **with `--no-cv-to-ros`**.
- All-NaN / crash on frame 0 → keep `--start-sample 1` (first frame is often undetected).
- Compile stuck minutes → ensure `ACADOS_EXT_FUN_COMPILE_FLAGS=-O1` (run_ik.sh sets it).
