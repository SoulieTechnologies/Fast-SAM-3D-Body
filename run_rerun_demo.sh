#!/usr/bin/env bash
# Launch the full live demo: SAM3D extractor (Rerun UI + recording) + ACADOS IK
# retargeting (meshcat + Rerun mirror). Ctrl+C stops both.
#
#   bash run_rerun_demo.sh                        # webcam 0
#   SOURCE=path/to/video.mp4 bash run_rerun_demo.sh
#   GPU=7 FX=900 bash run_rerun_demo.sh
#
# Then open (over ssh -L if remote):
#   Rerun UI : http://localhost:9090   (camera + 3D skeleton + retarget + latency)
#   meshcat  : http://localhost:7000/static/   (full URDF human)
set -uo pipefail

# ── Config (override via env vars) ───────────────────────────────────────────
SAM3D_ENV="${SAM3D_ENV:-fast_sam_3d_body}"       # conda env with torch/TRT/SAM3D
ACADOS_ENV="${ACADOS_ENV:-acados}"               # conda env with pinocchio/acados
COMFI_DIR="${COMFI_DIR:-$HOME/code/comfi-examples_new}"
SOURCE="${SOURCE:-0}"                            # webcam index or video path
GPU="${GPU:-0}"
FX="${FX:-0}"                                    # camera focal (0 = MoGe2 auto)
EMIT_PORT="${EMIT_PORT:-8090}"
RERUN_GRPC="${RERUN_GRPC:-9876}"
RERUN_WEB="${RERUN_WEB:-9090}"
RERUN_MODE="${RERUN_MODE:-web}"                  # web | native | save
SUBJECT_HEIGHT="${SUBJECT_HEIGHT:-1.75}"
SUBJECT_WEIGHT="${SUBJECT_WEIGHT:-70}"
IK_N="${IK_N:-10}"
EXTRA_A="${EXTRA_A:-}"                           # extra flags for rerun_demo.py
EXTRA_B="${EXTRA_B:-}"                           # extra flags for run_ik_live_rerun.py

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$REPO_DIR/output_rerun_demo"
mkdir -p "$LOG_DIR"

CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
[ -f "$CONDA_SH" ] || { echo "conda not found"; exit 1; }
source "$CONDA_SH"

PIDS=()
cleanup() {
    echo; echo "Stopping..."
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null; done
    wait 2>/dev/null
}
trap cleanup INT TERM EXIT

# ── Proc B: ACADOS IK (starts first; retries until the extractor is up) ──────
echo "[B] Starting ACADOS IK retargeting (env: $ACADOS_ENV)..."
(
    conda activate "$ACADOS_ENV"
    cd "$COMFI_DIR"
    export ACADOS_SOURCE_DIR="$PWD/acados"
    export LD_LIBRARY_PATH="$PWD/acados/lib:${LD_LIBRARY_PATH:-}"
    export ACADOS_EXT_FUN_COMPILE_FLAGS=-O1
    exec python scripts/run_ik_live_rerun.py \
        --host localhost --emit-port "$EMIT_PORT" \
        --rerun-url "rerun+http://127.0.0.1:${RERUN_GRPC}/proxy" \
        --N "$IK_N" --subject-height "$SUBJECT_HEIGHT" \
        --subject-weight "$SUBJECT_WEIGHT" $EXTRA_B
) > "$LOG_DIR/ik.log" 2>&1 &
PIDS+=($!)
echo "    log: $LOG_DIR/ik.log"

# ── Proc A: SAM3D extractor + Rerun UI + recording (foreground) ──────────────
echo "[A] Starting SAM3D extractor (env: $SAM3D_ENV)..."
conda activate "$SAM3D_ENV"
cd "$REPO_DIR"
FX_FLAG=""
[ "$FX" != "0" ] && FX_FLAG="--fx $FX"
python rerun_demo.py \
    --source "$SOURCE" --gpu "$GPU" $FX_FLAG \
    --emit-port "$EMIT_PORT" \
    --rerun-mode "$RERUN_MODE" \
    --rerun-grpc-port "$RERUN_GRPC" --rerun-web-port "$RERUN_WEB" \
    $EXTRA_A
