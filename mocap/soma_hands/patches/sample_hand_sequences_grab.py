"""Strategy-C drop-in: a GRAB-backed hand-pose sampler to replace SOMA's stubbed
`src/soma/data/sample_hand_sequences.py` so `animate_hand=true` produces bodies
with REAL articulated fingers during synthetic-data generation.

Stock SOMA raises NotImplementedError in every function here; this module fills
`hand_pose_sequence_generator` and a `hand_populate_source` that draws MANO
hand-pose windows from GRAB (SMPL-X) npz sequences. Wire it per
patches/enable_hand_animation.md.

STATUS: UNVALIDATED — written offline (no GPU/GRAB on the Mac). Shapes follow
SMPL-X (left/right hand = 15 axis-angle joints = 45 dims each). Verify the GRAB
npz key names ('pose_hand' vs 'lhand_pose'/'rhand_pose') on crslab and the PCA-
vs-full convention before trusting it.
"""
import glob
import os

import numpy as np


def hand_populate_source(grab_dir, splits=None, min_frames=30):
    """Return {seq_key: npz_path} of GRAB sequences to sample hand poses from.
    Mirrors SOMA's body_populate_source contract (a dict keyed like the body
    npzs so the two align in the synthesizer loop)."""
    paths = sorted(glob.glob(os.path.join(grab_dir, "**", "*.npz"), recursive=True))
    out = {}
    for p in paths:
        key = os.path.relpath(p, grab_dir).replace(os.sep, "_")[:-4]
        out[key] = p
    if not out:
        raise FileNotFoundError(f"no GRAB npz under {grab_dir}")
    return out


def _load_hand_pose(npz_path):
    """Return (T, 45), (T, 45) left/right full axis-angle hand pose from a GRAB
    (SMPL-X) npz. GRAB stores full 15-joint MANO pose; handle both key styles."""
    d = np.load(npz_path, allow_pickle=True)
    keys = set(d.keys())
    if {"lhand_pose", "rhand_pose"} <= keys:
        return d["lhand_pose"].reshape(-1, 45), d["rhand_pose"].reshape(-1, 45)
    if "pose_hand" in keys:               # concatenated [L(45) | R(45)]
        ph = d["pose_hand"].reshape(-1, 90)
        return ph[:, :45], ph[:, 45:]
    if "fullpose" in keys:                # SMPL-X fullpose: hands are the tail
        fp = d["fullpose"].reshape(-1, 165)
        return fp[:, 75:120], fp[:, 120:165]
    raise KeyError(f"no recognised hand-pose key in {npz_path}: {sorted(keys)}")


def hand_pose_sequence_generator(T, num_hand_var_perseq, grab_source,
                                 rng=None):
    """Yield num_hand_var_perseq windows of length T of (pose_hand_L, pose_hand_R)
    sampled from random GRAB sequences. Returns an array (num_var, T, 90) that
    the body synthesizer concatenates onto pose_hand (SMPL-X order L|R)."""
    rng = rng or np.random.default_rng(0)
    seq_paths = list(grab_source.values())
    out = np.zeros((num_hand_var_perseq, T, 90), dtype=np.float32)
    for i in range(num_hand_var_perseq):
        lp, rp = _load_hand_pose(rng.choice(seq_paths))
        n = min(len(lp), len(rp))
        if n < T:                          # short clip -> tile with reflection
            reps = int(np.ceil(T / max(n, 1)))
            lp = np.concatenate([lp, lp[::-1]] * reps)[:T]
            rp = np.concatenate([rp, rp[::-1]] * reps)[:T]
        else:
            s = int(rng.integers(0, n - T + 1))
            lp, rp = lp[s:s + T], rp[s:s + T]
        out[i] = np.concatenate([lp, rp], axis=1)
    return out
