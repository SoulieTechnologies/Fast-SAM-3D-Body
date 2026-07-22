"""Per-hand best-view selection for the multi-camera hand decoder.

With 3-4 cameras, running the SAM hand decoder on every view doubles the
batch for little gain: a view where the hand is small, edge-on (fingers
toward the camera / hand seen from the side) or body-occluded contributes a
hallucinated 2D that the epipolar check then has to throw away. Instead,
each hand picks its own top-K views every frame and only those crops enter
the decoder batch.

Per (hand, view) the score multiplies four independent [0,1]-ish factors:

  size      metric crop side in pixels (fx * hand_size / depth), normalized
            by the best view — "biggest crop" == closest/highest-res view
  vis       |cos| between the palm-plane normal and the camera->wrist ray.
            The palm plane comes from the THREE NLF hand markers cosmik
            already triangulates (wrist centre, thumb, pinky): its normal is
            their cross product — equivalent to the smallest-eigenvector of
            the marker covariance but closed-form and sign-free. vis ~ 1
            when the palm/back faces the camera, ~ 0 both when the hand is
            edge-on AND when the fingers point at the camera (in both cases
            the line of sight lies IN the palm plane).
  conf      mean NLF score of that view's 2D hand markers — NLF uncertainty
            is per view, so a body-occluded hand (side camera looking
            through the torso) scores low here even when its crop is big
  in_frame  area fraction of the crop inside the image (cropped-off hands
            near the frame edge feed the decoder black borders)

vis/conf are floored (a 45-degree hand is still perfectly decodable) and
missing components are neutral, so the ranking degrades gracefully to
"biggest crop wins" when the 3D markers are not available. A small
multiplicative bonus for the previously selected views adds hysteresis:
without it two near-equal views flap every frame and the per-view decoder
noise shows up as hand jitter.

Pure numpy — no torch/cv2 — so the geometry is unit-testable anywhere
(tools/test_view_selection.py).
"""

import numpy as np

# floors keep one bad-but-recoverable factor from vetoing a view outright
VIS_FLOOR = 0.25
CONF_FLOOR = 0.25
# ray-diversity strength: after the best view is picked, a candidate whose
# camera->wrist ray is PARALLEL to an already-selected one is scaled by
# (1 - DIVERSITY); an orthogonal ray keeps its full score. Two near-parallel
# rays (e.g. the two front cameras) triangulate depth poorly — DLT
# conditioning goes with sin(angle) — so near-ties break toward a wide pair,
# while a genuinely better view (flat vs edge-on) still wins.
DIVERSITY = 0.35


def palm_normal(wrist, thumb, pinky):
    """Unit normal of the palm plane from the 3 metric-3D hand markers (m).

    Returns None when any marker is missing or the three are near-collinear
    (cross-product area < ~1 cm^2 — orientation is then unknowable).
    """
    pts = np.asarray([wrist, thumb, pinky], np.float64)
    if not np.isfinite(pts).all():
        return None
    n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    nn = np.linalg.norm(n)
    if nn < 1e-4:
        return None
    return n / nn


def view_visibility(normal, wrist_w, R, T):
    """|cos(palm normal, camera line of sight)| for one view, in [0,1].

    normal/wrist_w in the world (cam0) frame; R,T world->cam. 1 = palm or
    back of hand square to the camera; 0 = the line of sight lies in the
    palm plane (edge-on hand OR fingers pointing at the camera).
    Returns None (neutral) when the normal is unknown or the wrist is
    missing/at the camera centre.
    """
    if (
        normal is None
        or wrist_w is None
        or not np.isfinite(np.asarray(wrist_w)).all()
    ):
        return None
    ray = np.asarray(R) @ np.asarray(wrist_w) + np.asarray(T).reshape(3)
    d = np.linalg.norm(ray)
    if d < 1e-6:
        return None
    return float(abs((np.asarray(R) @ normal) @ (ray / d)))


def in_frame_fraction(center_xy, side_px, width, height):
    """Area fraction of a side_px box centred at center_xy inside the image."""
    if (
        center_xy is None
        or side_px is None
        or not np.isfinite(side_px)
        or side_px <= 0
        or not np.isfinite(np.asarray(center_xy)).all()
    ):
        return None
    cx, cy = center_xy
    h = side_px / 2.0
    w_in = max(0.0, min(cx + h, width) - max(cx - h, 0.0))
    h_in = max(0.0, min(cy + h, height) - max(cy - h, 0.0))
    return (w_in * h_in) / (side_px * side_px)


def rank_views(cands, prev=(), switch_bonus=1.15):
    """Rank candidate views for ONE hand, best first.

    cands: {view: {"size": crop side px | None, "vis": [0,1] | None,
                   "conf": [0,1] | None, "in_frame": [0,1] | None}}
    prev: view ids selected last frame (hysteresis bonus).
    Returns (ordered view list, {view: score}).
    """
    sizes = [c.get("size") for c in cands.values()]
    sizes = [s for s in sizes if s is not None and np.isfinite(s) and s > 0]
    smax = max(sizes) if sizes else None
    scores = {}
    for v, c in cands.items():
        s = c.get("size")
        size_n = (
            (s / smax)
            if (smax and s is not None and np.isfinite(s) and s > 0)
            else 0.5
        )
        vis = c.get("vis")
        vis_f = (
            1.0
            if vis is None
            else VIS_FLOOR + (1 - VIS_FLOOR) * min(max(vis, 0.0), 1.0)
        )
        conf = c.get("conf")
        conf_f = (
            1.0
            if conf is None
            else CONF_FLOOR + (1 - CONF_FLOOR) * min(max(conf, 0.0), 1.0)
        )
        inf = c.get("in_frame")
        inf_f = 1.0 if inf is None else min(max(inf, 0.0), 1.0)
        sc = size_n * vis_f * conf_f * inf_f
        if v in prev:
            sc *= switch_bonus
        scores[v] = sc
    order = sorted(scores, key=scores.get, reverse=True)
    return order, scores


def _sin_angle(a, b):
    """sin of the angle between two 3D rays (1 = orthogonal, 0 = parallel)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    if d < 1e-9:
        return 1.0
    return float(min(np.linalg.norm(np.cross(a, b)) / d, 1.0))


def select_views(cands, k, prev=(), switch_bonus=1.15, rays=None):
    """Top-k views for one hand, plus the full ranking for fallbacks.

    rays: optional {view: camera->wrist direction, any common frame} — enables
    the greedy ray-diversity pass (see DIVERSITY): the best view is taken
    outright, then each further pick discounts candidates by how parallel
    their ray is to the closest already-selected one.
    """
    order, scores = rank_views(cands, prev, switch_bonus)
    k = max(k, 0)
    if rays is None or k <= 1 or len(order) <= k:
        return order[:k], order, scores
    sel, rest = [order[0]], list(order[1:])
    while rest and len(sel) < k:
        best_v, best_e = None, -1.0
        for v in rest:
            e = scores[v]
            if v in rays:
                sels = [s for s in sel if s in rays]
                if sels:
                    minsin = min(_sin_angle(rays[v], rays[s]) for s in sels)
                    e *= 1.0 - DIVERSITY * (1.0 - minsin)
            if e > best_e:
                best_v, best_e = v, e
        sel.append(best_v)
        rest.remove(best_v)
    return sel, order, scores
