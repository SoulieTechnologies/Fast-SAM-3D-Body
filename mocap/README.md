# MOCAP — hand mocap vs fastsam3d validation

Offline pipeline to validate the fastsam3d+cosmik markerless hand joint angles
against marker-based mocap (OptiTrack Motive), comparing **angles** (frame-
independent, robust to the marker↔joint-centre offset). Pure numpy/matplotlib,
no GPU. Hands only (body markers handled separately).

## Layout

```
utils/          library (importable as `from utils import ...`)
  hand_kinematics.py   angle maths + STA handling (Points 1-3), with proofs
  io_motive.py         Motive CSV -> point cloud; ID-free tracking; labelled read
  hand_angles.py       labelled 21-marker seq -> all 20 articulation angles
  compare.py           mocap-vs-SAM3D comparison over the repeated iterations
scripts/        runnable entry points
  process_hand.py      MAIN: labelled take -> angles CSV + graph (the reference)
  generate_collection_xlsx.py   builds the data-collection workbook
  pilot_index_wrist.py / pilot_index_3.py   single-finger pilots (validated)
  test_hand_kinematics.py / test_hand_angles.py   tests (run them, no GPU)
data/           takes, exports, generated CSV/PNG, the workbook
```

## Raw (unlabelled) Motive take → angles — the working path

Motive exports raw, ID-unstable point clouds (a marker gets a new id on every
reappearance). Workflow:

```
# 1. label the 21 markers once on the flat calibration frame (interactive)
python scripts/label_seed.py data/<take>.csv            # -> <take>_seed.npz
# 2. track from the seed (per-frame INDEPENDENT registration, no drift) + angles
python scripts/process_hand.py data/<take>.csv --seed data/<take>_seed.npz --hand left
# optional: see where tracking is weak
python scripts/track_review.py data/<take>.csv data/<take>_seed.npz
```

`track_hand_independent` registers a DISTINCTIVE palm anchor (wrist + thumb +
finger MCPs) to each frame on its own — no temporal propagation, so a bad frame
never corrupts later ones — then a per-finger chain search (fixed seed bone
lengths, "try combinations, keep the best skeleton fit") places PIP/DIP/tip.
On the gabin left-hand take this gives ~84% correct bones (vs 26% for
frame-to-frame trackers) and yields **clean, readable flexion segments** during
the finger-test windows.

Known limit: ~50% per-frame coverage, and a minority of segments have
systematic mislabels (a wrong finger chain with correct bone lengths → a
smooth-but-wrong angle that no bone/position/temporal filter catches). Use the
clean segments for the SAM3D comparison; re-label a bad window with
`label_seed.py --frame N --append <take>_seed.npz` (multi-seed) or, best, feed
Motive's own labelled-marker tracking. Cleaner capture (mask reflections, kill
the bracelet ghosts, better fixation) lifts every method.

## The reference pipeline

```
python scripts/process_hand.py data/<take>.csv --list-labels   # check naming
python scripts/process_hand.py data/<take>.csv                 # -> angles
```
Reads a **labelled** 21-marker Motive/SOMA export (or a pre-labelled
`T x 21 x 3 .npy`), computes every articulation angle, and writes
`<take>_angles.csv` (frame, time, 20 angle columns) + `<take>_angles.png`
(one panel per finger). **This CSV is what the SAM3D angles are compared to.**

## 21 markers per hand -> 20 articulation angles

Landmarks (fastsam3d order): per finger `[tip, DIP, PIP, MCP]` + wrist.
Per finger we log 4 angles — `mcp_flex`, `mcp_abd` (palm-frame decomposition),
`pip`, `dip` (inter-segment); thumb columns are CMC flex/abd, MCP, IP.

Labelling: `io_motive.read_labeled` maps named columns to slots via a robust
`default_resolver` (handles `RIndexPIP`, `index_pip`, `Idx3`, `wrist`, ...). If
markers arrive **unlabelled** (raw Motive), SOMA (colleague's tool) labels them
first; `io_motive.track_chain` also re-tracks by position (IDs are useless —
Motive re-IDs a marker on every reappearance) and re-IDs a returning marker by
its bone length to tracked neighbours.

## STA handling (see hand_kinematics + its tests)

Angles depend only on segment directions, so axial marker slip is harmless;
perpendicular slip biases the angle by `dphi ~ s/L` and changes bone length only
at O(s^2), so it is **measured, not corrected**. `hand_angles` cleans each
joint: reject frames whose adjacent bone length deviates >30% (swap), drop
anatomically impossible values, median-smooth spikes, interpolate only short
gaps (real dropouts stay NaN).

## Comparing to SAM3D over the iterations (boss: >=2 takes x 10 reps)

mocap and SAM3D are **co-captured** -> every rep is a paired sample. `compare.py`:
1. **Per-rep, per-joint RMSE** of the time-aligned waveform -> distribution over
   ~20 reps -> mean +/- 95% CI (the headline accuracy).
2. **Bland-Altman** pooled per joint -> bias + limits of agreement.
3. **Repeatability floor** — the 10 reps give the subject's own rep-to-rep
   spread; SAM3D is "as good as ground truth is stable" when its error <= that.
   The 2 takes give test-retest / a held-out set.
4. **Feature agreement** — ROM, peak flexion, timing per rep: do the SAM3D
   distributions match mocap's, not just the instantaneous value.
5. **Robustness vs occlusion** — tasks are graded None->High; report accuracy vs
   occlusion, excluding reps with mocap dropout while keeping N large.

## Data collection

`data/hand_data_collection.xlsx` (regenerate with
`scripts/generate_collection_xlsx.py`): Protocol, Tasks, Calendar (15 subjects,
5/day, Wed 23 -> Fri 25 Jul 2026), Subjects, Data log (pre-filled 165 rows =
15 x [calib + 5 tasks x 2 takes]), Schedule. Tasks: hand flexion + per-finger
pinch, pick-and-place, screwdriver, open a plastic bottle, hammer (slowly).

## Next (needs real data)

Wire `compare.py` end-to-end once we have (a) a labelled 21-marker take (from
SOMA) to confirm the naming resolver, and (b) a simultaneous SAM3D recording
(`goliath70_3d.npy`) + the clap for `find_offset`. The SAM3D side reuses the
SAME `hand_angles` so the conventions match by construction.
