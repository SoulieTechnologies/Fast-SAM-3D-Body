# SOMA hand auto-labelling — runbook

Goal: use **SOMA** (ICCV'21, MPI) to auto-label our raw 21-marker hand mocap.
Define the marker layout once on the SMPL-X mesh → SOMA trains a per-layout model
on synthetic bodies → it labels raw Motive point clouds (handling occlusion,
ghosts, unstable IDs — exactly our failure modes). This package is prepared on
the Mac; **everything here runs on crslab (Linux + CUDA GPU)**.

## ⚠️ Read first — the finger-animation caveat (key finding, 2026-07-22)

Open-source SOMA **cannot animate fingers** during synthetic-data generation:
`animate_hand=true` hard-raises `NotImplementedError`, and the whole hand-pose
sampler (`sample_hand_sequences.py`) is stubbed. This is *the* reason "SOMA is
body-oriented". Two strategies:

- **Strategy A (default, runs on stock SOMA now).** Train with `animate_hand=false`:
  markers sit on the hand vertices of otherwise flat-hand AMASS bodies. Marker-
  layout augmentation + AMASS noise give variation. Finger flexion at capture is
  out-of-distribution, but the wrist→MCP frame is rigid so labels are usually
  still separable. **Try this first — it needs no code changes.**
- **Strategy C (quality upgrade, needs our patch).** Implement the hand sampler
  from GRAB so `animate_hand=true` yields real finger poses. See
  `patches/enable_hand_animation.md`. Real dev + GPU validation.

**Decision needed from Théophile/boss:** start with A and only escalate to C if
A's finger labels aren't good enough? (Recommended.)

## Assets to obtain (licensed — need personal MPI accounts)

Put everything under a `$SUPPORT` base dir on crslab. `python check_assets.py
--support-base $SUPPORT` reports what's missing.

| asset | where | into |
|---|---|---|
| SMPL-X model (locked head for SOMA) | smpl-x.is.tue.mpg.de | `$SUPPORT/smplx/{neutral,male,female}/model.npz` |
| extra SMPL-X data | download.is.tue.mpg.de (soma domain) | `$SUPPORT/smplx/` |
| AMASS SMPL-X **neutral** params (CMU, PosePrior, Transitions, HumanEva, ACCAD, TotalCapture) | amass.is.tue.mpg.de | `$SUPPORT/smplx/amass_neutral/<DS>/` |
| **GRAB** SMPL-X params (finger articulation — Strategy C) | grab.is.tue.mpg.de | `$SUPPORT/smplx/amass_neutral/GRAB/` |
| AMASS marker-noise model (or original AMASS markers) | soma.is.tue.mpg.de | `$SUPPORT/marker_noise/` |
| SOMA folder template | download.is.tue.mpg.de/soma/tutorials/SOMA_FOLDER_TEMPLATE.tar.bz2 | `$WORK/` |

fastsam3d already uses SMPL-X + AMASS (`mhr2smpl/smooth/preprocess_amass.py`) —
check those paths first, they may already hold most of this.

## Steps (on crslab)

```bash
# 0. install (see notes in the script; psbody-mesh + moshpp are the hard parts)
bash install_soma.sh soma
conda activate soma
export SUPPORT=/path/to/support_files   WORK=/path/to/soma_work

# 1. assets present?
python check_assets.py --support-base $SUPPORT

# 2. define the marker layout on the SMPL-X mesh (the boss's "markers on a mesh")
python build_hand_superset.py --model-path $SUPPORT/smplx/neutral/model.npz \
    --hand left --out $WORK/data/clear_hands/superset.json
#   -> VERIFY the 21 vertices in a mesh viewer (dorsal side!) before training.

# 3. place our raw takes into the SOMA folder structure
python prepare_soma_data.py --c3d ../data/take_gabin_1.c3d --c3d ../data/take_gabin_0.c3d \
    --ds-name clear_hands --subject gabin --gender male --work-dir $WORK/mocap
#   -> note the reported units; set mocap.unit accordingly (our Motive = m).

# 4. generate synthetic data + train (Strategy A). Smoke first, then the real run:
python train_hands.py --work-base $WORK --support-base $SUPPORT \
    --layout $WORK/data/clear_hands/superset.json \
    --expr-id clear_hands_A_v1 --num-gpus 1 --num-cpus 4 --smoke   # validate wiring
python train_hands.py --work-base $WORK --support-base $SUPPORT \
    --layout $WORK/data/clear_hands/superset.json \
    --expr-id clear_hands_A_v1 --num-gpus 1 --num-cpus 4           # ~hours on 1 GPU
#   --layout can instead be a LABELLED c3d frame (MoSh++ extracts the placement).
#   The AMASS marker-noise model is disabled (we don't have it) inside the driver.

# 5. label our raw take with the trained model
python label_hands.py --work-base $WORK --support-base $SUPPORT \
    --expr-id clear_hands_A_v1 --ds-name clear_hands \
    --occ 5 --ghost 3 --unit m --tracklet
#   -> LABELLED c3d + .pkl under training_experiments/<expr>/<data_id>/evaluations/

# 6. bridge back to our angles + score it
python soma_labeled_to_angles.py $WORK/.../take_gabin_1_labeled.c3d --hand left
#    -> reuses utils.hand_angles, and can be scored with the same bone-health /
#       clean-% metric as track_hand_hybrid for a direct A/B vs our tracker.
```

## Why this can beat our geometric tracker

`track_hand_hybrid` is a hand-crafted registration (96%/93% clean frames but
still ~1 red bone/frame and a per-frame confidence gap). SOMA is a learned
labeller trained explicitly against occlusion + ghost points — our two failure
modes (constant dropouts, bracelet reflections). If Strategy A already labels
cleanly it replaces the whole seed+track pipeline; if not, Strategy C (GRAB
fingers) is the escalation. Either way we report SOMA-vs-hybrid on the same
metric so the boss sees the trade honestly.

## Verified on the Mac (2026-07-22, no GPU needed)

Components that don't need SMPL-X/AMASS/GPU were tested against the real takes:

- `prepare_soma_data.py` reads the real c3d. **take_gabin_1: 1631 point columns,
  28899 frames @120fps, UNITS='m'; take_gabin_0: 1634 cols, 14222 frames, 'm'.**
  → the raw Motive c3d is ~1631 ID-unstable sparse trajectories (mostly NaN per
  frame, ~21 active) — the point cloud SOMA is meant to ingest; confirm SOMA's
  `MocapSession` accepts this width. **Set `mocap.unit=m`** (SOMA default mm =
  1000× wrong). NB take_gabin_0.c3d has 14222 frames vs 26338 in its csv — the
  c3d is a shorter export; pick the intended one.
- `soma_labeled_to_angles.py` validated end-to-end: fabricated a labelled c3d
  with our 21 SOMA marker names, wrote+read it, and got the 20 angles out
  (fixed a real dict-vs-array bug in the process). The bridge reproduces the
  exact `hand_angle_series` our tracker uses, so SOMA-vs-hybrid is apples-to-apples.
- name↔slot bijection (21↔21) and `palm_basis` left/right handedness confirmed.
- For reference, our current hybrid angles are sparse (real take_gabin_1:
  mcp_flex 46% / mcp_abd 33% / pip 56% / dip 13% finite) — the coverage SOMA is
  meant to lift; measure SOMA's coverage on the same columns.

## File map

- `build_hand_superset.py` — 21 hand landmarks → SMPL-X vertex ids → layout JSON
- `prepare_soma_data.py` — raw c3d → SOMA `ds/subject/seq.c3d` + settings.json
- `check_assets.py` — verify SMPL-X / AMASS / GRAB / deps
- `install_soma.sh` — conda env + psbody-mesh + moshpp + soma
- `train_hands.py` — driver: synth data + train (wraps train_multiple_soma)
- `label_hands.py` — driver: auto-label our c3d (wraps run_soma_on_multiple_settings)
- `configs/soma_train_hands.yaml` — hand training overrides, documented (+ the caveat)
- `configs/soma_run_hands.yaml` — labelling overrides, documented
- `soma_labeled_to_angles.py` — SOMA labelled c3d → our 20 angles (the bridge)
- `compare_angle_coverage.py` — **the boss-facing result**: SOMA vs hybrid per-angle coverage + agreement RMSE + bar chart
- `patches/` — Strategy-C: GRAB hand-pose sampler + the 5 edits to enable it
