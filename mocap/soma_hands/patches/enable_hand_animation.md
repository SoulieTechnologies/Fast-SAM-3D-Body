# Strategy C — make `animate_hand=true` actually work (GRAB fingers)

Stock SOMA cannot articulate fingers during synthetic-data generation: the code
hard-raises `NotImplementedError` the moment `animate_hand=true`. This patch
fills the missing pieces so training bodies have **real GRAB finger poses**,
which is what lifts finger-marker labels from "OK" (Strategy A, flat hands) to
"good".

Do this only if Strategy A's finger accuracy is insufficient. It is real dev +
GPU validation, not a config flip.

## The 5 edits (paths relative to the `soma` repo)

1. **`src/soma/data/sample_hand_sequences.py`** — replace the whole stub file
   with `sample_hand_sequences_grab.py` from this folder (implements
   `hand_populate_source` and `hand_pose_sequence_generator`).

2. **`src/soma/data/synthetic_body_dataset.py`** (~line 141) — under the
   `smplx` branch, instead of `raise NotImplementedError`, declare the pytables
   column for the hand pose:
   ```python
   if animate_hand:
       pose_hand = pytables.Float32Col(num_timeseq_frames * 2 * 15 * 3)  # L|R
   ```

3. **`src/soma/data/synthetic_body_dataset.py`** (~line 156) — replace
   `hand_frames = hand_populate_source() ...` with a GRAB-backed source:
   ```python
   from soma.data.sample_hand_sequences import hand_populate_source
   hand_frames = (hand_populate_source(cfg_grab_dir) if animate_hand
                  else {k: None for k in body_npz_fnames.keys()})
   ```
   and thread `cfg_grab_dir` (a new `data_parms.body_dataset.grab_dir`) through
   `prepare_marker_dataset` / the data module.

4. **`src/soma/data/body_synthesizer.py`** (~line 144) — where it does
   `if hand_frames is not None: body_parms_np.update(hand_sampler(...))`,
   implement `hand_sampler` to call
   `hand_pose_sequence_generator(T, num_hand_var_perseq, grab_source)` and write
   the result into `body_parms_np['pose_hand']` (SMPL-X expects left then right,
   45+45). Make sure the SMPL-X forward call receives `left_hand_pose` /
   `right_hand_pose` (full, `use_pca=False`).

5. **`configs/soma_train_hands.yaml`** — set `animate_hand: true`,
   `enable_rnd_vid_on_face_hands: true`, add `GRAB` to `amass_splits.train`, and
   add `data_parms.body_dataset.grab_dir: <amass>/GRAB`.

## Validation checklist (on crslab)

- [ ] Render 5 synthetic frames (Blender/`parameters_to_mesh.py`) and confirm
      fingers actually flex and the 21 hand markers ride the moving fingers.
- [ ] Confirm marker count per body == our layout (21/hand) with no NaNs.
- [ ] Overfit a tiny run (100 seqs, 5 epochs) → labeling loss drops.
- [ ] Compare Strategy-A vs Strategy-C labels on `take_gabin_1.c3d` with
      `eval_labeling` (or our bone-health metric via the reverse bridge).
