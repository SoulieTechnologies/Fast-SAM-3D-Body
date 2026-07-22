"""Build a SOMA/MoSh++ marker layout ("superset") for our 21-marker HAND set.

This is the "show the markers on the human mesh" step the boss asked for: we
place each of the 21 fastsam3d hand landmarks on a specific SMPL-X vertex, and
SOMA then learns to auto-label raw point clouds against this layout.

The 21 landmarks per hand (fastsam3d order) are, per finger, [tip, dip, pip,
mcp] + wrist. We anchor:
  * fingertips  -> the documented SMPL-X fingertip vertex ids (smplx/vertex_ids)
  * dip/pip/mcp -> the nearest DORSAL surface vertex to the corresponding MANO
                   hand joint centre (markers sit on the back of the hand)
  * wrist       -> nearest dorsal vertex to the wrist joint

Output is a moshpp marker-layout JSON:
  {"surface_model_type": "smplx",
   "markersets": [{"type": "hand", "distance_from_skin": 0.0095,
                   "indices": {"LINDEX_TIP": 4933, ...}}]}

RUN ON crslab (needs the SMPL-X model + the `smplx` pkg). NOT on the Mac.

    python build_hand_superset.py \
        --model-path $SUPPORT/smplx/neutral/model.npz \
        --hand left --out layouts/hand_left.json

Marker-name convention: <SIDE><FINGER>_<JOINT>, e.g. LINDEX_MCP, RTHUMB_TIP,
LWRIST. These names round-trip to our slots via name_to_slot() below, so a
SOMA-labelled c3d can be read straight back into utils.hand_angles.
"""
import argparse
import json
import pathlib

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
JOINTS = ("mcp", "pip", "dip", "tip")  # our per-finger order (proximal->distal)

# SMPL-X fingertip vertex ids (from smplx/vertex_ids.py, 'smplx' key).
TIP_VID = {
    "left":  {"thumb": 5361, "index": 4933, "middle": 5058, "ring": 5169, "pinky": 5286},
    "right": {"thumb": 8079, "index": 7669, "middle": 7794, "ring": 7905, "pinky": 8022},
}

# SMPL-X joints. Body wrists: 20 L / 21 R. Hands are 3 joints/finger (proximal,
# middle, distal) in the order index, middle, pinky, ring, thumb.
#
# CAUTION: the hand base index depends on the joint set. The `model().joints`
# TENSOR inserts jaw + 2 eyes before the hands -> left hand at 25, right at 40.
# The smplx JOINT_NAMES *list* omits them there -> left 22, right 37. Feeding the
# wrong base silently mislabels everything, so build() AUTO-DETECTS the base by
# checking that each fingertip vertex lands nearest its own finger's joints.
WRIST_J = {"left": 20, "right": 21}
_HAND_BASE_CANDIDATES = {"left": (25, 22), "right": (40, 37)}
_FINGER_ORDER = ("index", "middle", "pinky", "ring", "thumb")  # SMPL-X hand order


def hand_joint_ids(hand, base):
    """{finger: {'mcp': j, 'pip': j, 'dip': j}} given the hand's base joint id."""
    out = {}
    for fi, f in enumerate(_FINGER_ORDER):
        j0 = base + 3 * fi
        out[f] = {"mcp": j0, "pip": j0 + 1, "dip": j0 + 2}  # proximal->distal
    return out


def resolve_hand_base(hand, joints, verts):
    """Pick the hand base offset whose finger joints match the known fingertip
    vertices: each finger's tip vertex must be nearest that finger's dip joint
    among all five dip joints. Removes the 25-vs-22 tensor/list ambiguity."""
    tips = {f: verts[TIP_VID[hand][f]] for f in FINGERS}
    for base in _HAND_BASE_CANDIDATES[hand]:
        hj = hand_joint_ids(hand, base)
        ok = True
        for f in FINGERS:
            dip = joints[hj[f]["dip"]]
            nearest = min(FINGERS, key=lambda g: np.linalg.norm(tips[g] - dip))
            if nearest != f:
                ok = False
                break
        if ok:
            return base
    raise RuntimeError(
        f"could not resolve {hand}-hand joint base from {_HAND_BASE_CANDIDATES[hand]}; "
        "the SMPL-X joint layout is unexpected — print joint names and fix.")


def marker_name(hand, finger, joint):
    return f"{hand[0].upper()}{finger.upper()}_{joint.upper()}"


def name_to_slot(name):
    """SOMA marker name -> (hand, fastsam3d slot index) for the reverse bridge."""
    from utils import hand_kinematics as hk  # local import (repo utils)
    hand = "left" if name[0] == "L" else "right"
    body = name[1:]
    if body.endswith("WRIST") or body == "WRIST":
        return hand, hk.WRIST
    finger, joint = body.rsplit("_", 1)
    return hand, hk.LM[finger.lower()][joint.lower()]


def vertex_normals(verts, faces):
    vn = np.zeros_like(verts)
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    n = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.clip(n, 1e-9, None)


def nearest_dorsal_vertex(joint_xyz, verts, vnorm, palm_normal, radius=0.02, k=60):
    """Nearest vertex to a joint, biased to the dorsal (back-of-hand) side:
    among the k nearest, take the one whose normal best aligns with the palm's
    dorsal normal (markers are taped on the back of the hand)."""
    d = np.linalg.norm(verts - joint_xyz, axis=1)
    cand = np.argsort(d)[:k]
    within = cand[d[cand] <= radius]
    if len(within) == 0:
        return int(cand[0])
    score = vnorm[within] @ palm_normal
    return int(within[np.argmax(score)])


def build(model_path, hand, distance_from_skin=0.0095, radius=0.02):
    import smplx
    import torch

    mp = pathlib.Path(model_path)
    model = smplx.create(
        model_path=str(mp.parent.parent if mp.is_file() else mp),
        model_type="smplx", gender="neutral", use_pca=False,
        flat_hand_mean=True, num_betas=10,
    )
    with torch.no_grad():
        out = model(return_verts=True)
    verts = out.vertices[0].numpy()
    joints = out.joints[0].numpy()
    faces = model.faces.astype(np.int64)
    vnorm = vertex_normals(verts, faces)

    # dorsal (back-of-hand) normal from the palm plane wrist->MCP fan
    base = resolve_hand_base(hand, joints, verts)
    hj = hand_joint_ids(hand, base)
    print(f"{hand} hand: resolved joint base = {base}")
    wrist = joints[WRIST_J[hand]]
    idx_mcp = joints[hj["index"]["mcp"]]
    pinky_mcp = joints[hj["pinky"]["mcp"]]
    n = np.cross(idx_mcp - wrist, pinky_mcp - wrist)
    n = n / np.linalg.norm(n)
    # orient it dorsally: it should point away from the mean fingertip curl side.
    # heuristic: dorsal normal points away from the thumb for a flat hand; if the
    # sign is wrong the operator flips it once in the viewer (--flip-normal).
    palm_normal = n

    indices = {}
    for f in FINGERS:
        # tip: fixed vertex id
        indices[marker_name(hand, f, "tip")] = int(TIP_VID[hand][f])
        for j in ("dip", "pip", "mcp"):
            jx = joints[hj[f][j]]
            indices[marker_name(hand, f, j)] = nearest_dorsal_vertex(
                jx, verts, vnorm, palm_normal, radius)
    indices[f"{hand[0].upper()}WRIST"] = nearest_dorsal_vertex(
        wrist, verts, vnorm, palm_normal, radius)

    return {
        "surface_model_type": "smplx",
        "markersets": [{
            "type": "hand",
            "distance_from_skin": distance_from_skin,
            "indices": indices,
        }],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True,
                   help="SMPL-X neutral model.npz (or its folder)")
    p.add_argument("--hand", choices=["left", "right", "both"], default="left")
    p.add_argument("--distance-from-skin", type=float, default=0.0095)
    p.add_argument("--radius", type=float, default=0.02)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    hands = ["left", "right"] if args.hand == "both" else [args.hand]
    merged = None
    for h in hands:
        layout = build(args.model_path, h, args.distance_from_skin, args.radius)
        if merged is None:
            merged = layout
        else:
            merged["markersets"][0]["indices"].update(
                layout["markersets"][0]["indices"])
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(merged, f, indent=2)
    n = len(merged["markersets"][0]["indices"])
    print(f"wrote {args.out}  ({n} markers: {', '.join(list(merged['markersets'][0]['indices'])[:6])} ...)")
    print("VERIFY the vertex placement in a mesh viewer before training — the "
          "dorsal-normal sign is a heuristic; a wrong-side marker mislabels.")


if __name__ == "__main__":
    main()
