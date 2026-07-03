# Setup on a fresh GPU machine — Rerun demo + ACADOS retargeting

Exhaustive command list, from bare machine to the full live demo
(`rerun_demo.py` + `run_ik_live_rerun.py`). Tested target: Linux x86_64 + NVIDIA GPU.

## 0. Prerequisites

```bash
# NVIDIA driver must be working:
nvidia-smi                                  # must show the GPU

# Build tools (for detectron2, chumpy, acados):
sudo apt-get update && sudo apt-get install -y git build-essential cmake wget

# Miniconda (skip if conda already installed):
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/etc/profile.d/conda.sh
conda init bash    # then reopen the shell
```

## 1. Clone both repos

```bash
cd ~ && mkdir -p code && cd code
git clone https://github.com/SoulieTechnologies/Fast-SAM-3D-Body.git
git clone https://github.com/SoulieTechnologies/comfi-examples-hands.git
```

## 2. Env A — SAM3D (`fast_sam_3d_body`)

```bash
cd ~/code/Fast-SAM-3D-Body
bash setup_env.sh                           # creates the env, installs torch cu124,
                                            # detectron2, ultralytics, MoGe, TensorRT... (~20 min)
conda activate fast_sam_3d_body
pip install "rerun-sdk>=0.28"
```

## 3. Checkpoints (repo A)

```bash
conda activate fast_sam_3d_body
cd ~/code/Fast-SAM-3D-Body

# 3a. SAM-3D-Body (gated on HuggingFace):
#     1) visit https://huggingface.co/facebook/sam-3d-body-dinov3 and accept the license
#     2) login and download EXPLICITLY (no auto-download when --checkpoint_dir is a
#        local path — setup_sam_3d_body only pulls from HF when given no local path):
huggingface-cli login                       # paste a HF token (read)
huggingface-cli download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3
ls checkpoints/sam-3d-body-dinov3/model.ckpt checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt  # must exist

# 3b. YOLO11-pose weights:
mkdir -p checkpoints/yolo
wget -P checkpoints/yolo \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt
```

## 4. First smoke test — WITHOUT TensorRT (slow but zero build)

```bash
conda activate fast_sam_3d_body
cd ~/code/Fast-SAM-3D-Body
USE_TRT_BACKBONE=0 python rerun_demo.py --source 0 --gpu 0 \
    --checkpoint_dir ./checkpoints/sam-3d-body-dinov3 --emit-port 0
# → first run downloads the HF checkpoint (~5 GB), then serves the Rerun UI.
# open http://localhost:9090  (ssh -L 9090:localhost:9090 <machine> if remote)
# Expect only ~2-4 FPS here — TRT comes next. Ctrl+C to stop.
# No webcam? use a video file:  --source path/to/video.mp4
```

## 5. Build the TensorRT engines (~15-20 min, once)

```bash
conda activate fast_sam_3d_body
cd ~/code/Fast-SAM-3D-Body
python convert_backbone_tensorrt.py --all        # DINOv3 backbone → FP16 engine
python convert_yolo_pose_trt.py --model checkpoints/yolo/yolo11m-pose.pt --imgsz 640 --half
# engines land in checkpoints/sam-3d-body-dinov3/backbone_trt/ and checkpoints/yolo/
```

## 6. Env B — ACADOS (`acados`)

```bash
conda create -n acados python=3.11 -y
conda activate acados
cd ~/code/comfi-examples-hands
pip install --upgrade pip
pip install -e .                            # pinocchio(pin)+casadi, meshcat, onnxruntime...
pip install "rerun-sdk>=0.28"

# Build acados INSIDE the repo (scripts expect ACADOS_SOURCE_DIR=$PWD/acados):
git clone https://github.com/acados/acados.git --recursive
cd acados && mkdir -p build && cd build
cmake -DACADOS_WITH_QPOASES=ON ..
make install -j$(nproc)
cd .. && pip install -e interfaces/acados_template
cd ..

# Tera renderer (codegen templating) — install it NOW. The first solver build
# happens inside the BACKGROUNDED IK process (no tty), so the interactive
# auto-download prompt of older acados_template versions would hang there:
wget https://github.com/acados/tera_renderer/releases/download/v0.2.0/t_renderer-v0.2.0-linux \
  -O acados/bin/t_renderer && chmod +x acados/bin/t_renderer
```

## 7. Smoke test — IK process alone

```bash
conda activate acados
cd ~/code/comfi-examples-hands
export ACADOS_SOURCE_DIR=$PWD/acados
export LD_LIBRARY_PATH=$PWD/acados/lib:${LD_LIBRARY_PATH:-}
export ACADOS_EXT_FUN_COMPILE_FLAGS=-O1     # gcc -O2 takes MINUTES on the generated FK
python scripts/run_ik_live_rerun.py --emit-port 8090 --rerun-url ""
# should print "waiting for extractor on localhost:8090 ..." → setup OK, Ctrl+C.
```

## 8. Full demo (both processes)

```bash
cd ~/code/Fast-SAM-3D-Body
SAM3D_ENV=fast_sam_3d_body ACADOS_ENV=acados \
COMFI_DIR=~/code/comfi-examples-hands \
SOURCE=0 GPU=0 FX=900 \
bash run_rerun_demo.sh
```
- `FX` = camera focal in px (0 → MoGe2 auto-estimates once). `--fx` wrong ⇒ possible NaNs.
- `SOURCE=path/to/video.mp4` to demo on a file. `RERUN_MODE=native` on a machine with a screen.
- `USE_TRT=0` to run before the engines are built (slow).
- IK log: `output_rerun_demo/ik.log`. Recording: `output_rerun_demo/<timestamp>/`.

If remote, tunnel ALL THREE ports (the Rerun web page connects back to the gRPC
port 9876 from your browser — 9090 alone shows an empty viewer):

```bash
ssh -L 9090:localhost:9090 -L 9876:localhost:9876 -L 7000:localhost:7000 <machine>
```

| URL | What |
|---|---|
| http://localhost:9090 | **Rerun UI** — camera+overlay, 3D skeleton, retarget, latency |
| http://localhost:7000/static/ | **meshcat** — full URDF human driven by the IK |

First full run: acados generates + compiles the solver (~1-2 min), IPOPT warm-start
(~10 s), gravity alignment locks after ~30 frames — then everything goes live.

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `AttributeError: ... BuilderFlag has no attribute 'FP16'` (or EXPLICIT_BATCH) | TensorRT 11 installed → `pip install "tensorrt-cu12<11"` |
| `RuntimeError: operator torchvision::nms does not exist` | torch got bumped (often by ultralytics auto-update) → `pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-deps` |
| `TRT engine not found` | Step 5 not done, or `export TRT_BACKBONE_PATH=<repo>/checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_fp16.engine` |
| HF 401/403 on checkpoint download | Accept the license on the model page + `huggingface-cli login` |
| acados: `t_renderer not found` | Manual install (end of step 6) |
| acados solver compile takes minutes | `export ACADOS_EXT_FUN_COMPILE_FLAGS=-O1` |
| `pinocchio.casadi` import error | `pip install "pin>=3.7.0"` (NOT conda pinocchio without casadi) |
| Webcam not found | `ls /dev/video*`; try `SOURCE=1`; check user is in the `video` group |
| Skeleton NaN / diverges | Wrong `FX` — measure the camera focal or set `FX=0` (MoGe2) |
| Rerun page blank | rerun-sdk version <0.28 → `pip install -U rerun-sdk` in BOTH envs |
| IK never starts | Check `output_rerun_demo/ik.log`; proc B retries 120 s max — restart it after A is live |
