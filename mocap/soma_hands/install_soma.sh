#!/usr/bin/env bash
# Install SOMA + MoSh++ + psbody-mesh on crslab (Linux + CUDA GPU).
# SOMA targets python 3.7 / torch 1.8.2+cu102 / Ubuntu 20.04. Known-painful bits
# are flagged. Do NOT run on the Mac (no CUDA).
set -euo pipefail

ENV=${1:-soma}
echo "== conda env: $ENV =="
conda create -y -n "$ENV" python=3.7
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

# ezc3d must come from conda-forge (no working pip wheel for py3.7):
conda install -y -c conda-forge ezc3d

# torch 1.8.2 LTS + cu102 (match crslab's CUDA; adjust cuXXX if newer driver):
pip install torch==1.8.2+cu102 torchvision==0.9.2+cu102 torchaudio==0.8.2 \
    -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

# --- psbody.mesh (MPI-IS/mesh): the usual failure point ---------------------
# needs libboost-dev + a C++ toolchain. If `make all` fails on Boost, install
# libboost-dev and set BOOST_INCLUDE_DIRS.
git clone https://github.com/MPI-IS/mesh.git
( cd mesh && sudo apt-get install -y libboost-dev && BOOST_INCLUDE_DIRS=/usr/include/boost make all )

# --- MoSh++ ------------------------------------------------------------------
git clone https://github.com/nghorbani/moshpp.git
( cd moshpp && pip install -r requirements.txt && python setup.py develop )
# chumpy / smpl-fast-derivatives: if MoSh complains about missing derivatives,
# extract smpl-fast-derivatives into
#   $(python -c 'import psbody,os;print(os.path.dirname(psbody.__file__))')/smpl

# --- human_body_prior + smplx ------------------------------------------------
pip install git+https://github.com/nghorbani/human_body_prior.git
pip install smplx

# --- SOMA --------------------------------------------------------------------
git clone https://github.com/nghorbani/soma.git
( cd soma && pip install -r requirements.txt && python setup.py develop )

echo
echo "== done. Now: =="
echo "  python check_assets.py --support-base \$SUPPORT   # verify SMPL-X/AMASS/GRAB"
echo "  (see RUNBOOK.md for superset -> data-gen -> train -> label)"
