#!/usr/bin/env bash
# Two-camera demo: extract → triangulate → gravity-align → stage for the ACADOS IK.
# Usage:  ./demo_extract.sh [cam0.mp4] [cam1.mp4]   (defaults = take_01 test pair)
set -e

FASTSAM=/home/users/theo/code/Fast-SAM-3D-Body
TI=/home/users/theo/code/test_input
COMFI=/home/users/theo/code/comfi-examples_new
STEREO=$TI/cam_params/stereo_params.npz

CAM0="${1:-$TI/take_01/cam0.mp4}"
CAM1="${2:-$TI/take_01/cam1.mp4}"

cd "$FASTSAM"
echo "==> Extracting (dual-GPU, gpu0=7 gpu1=1) — cam0=$CAM0  cam1=$CAM1"
python extract_dualgpu.py \
  --cam0 "$CAM0" --cam1 "$CAM1" --stereo "$STEREO" \
  --output_dir ./output_dualgpu --gpu0 7 --gpu1 1

echo "==> Staging gravity-aligned keypoints for the IK"
mkdir -p "$COMFI/output/res_hpe/demo/take"
cp output_dualgpu/joints_3d_world.npy "$COMFI/output/res_hpe/demo/take/joints_world.npy"

echo ""
echo "DONE. Now run the IK + meshcat:"
echo "  cd $COMFI"
echo "  ./run_ik.sh --id demo --task take --keypoints-file joints_world.npy --no-cv-to-ros \\"
echo "    --start-sample 1 --display --N 10 --w-dq 1e-3 --w-u 1e-4 \\"
echo "    --urdf-file urdf/human_with_mano.urdf --subject-height 1.75 --subject-weight 70"
