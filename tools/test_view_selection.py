#!/usr/bin/env python3
"""Geometry tests for hand_view_select on a synthetic 4-camera ceiling rig.

Rig mirrors the real room: cams 0,1 in FRONT of the subject, cams 2,3 on the
SIDES, all looking at the origin where the hand is. Pure numpy — runs on any
machine (python3 tools/test_view_selection.py).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hand_view_select import (in_frame_fraction, palm_normal, rank_views,
                              select_views, view_visibility)


def look_at(cam_pos, target=(0, 0, 0)):
    """World->cam R,T with +z toward target (y roughly down)."""
    z = np.asarray(target, float) - np.asarray(cam_pos, float)
    z /= np.linalg.norm(z)
    up = np.array([0.0, -1.0, 0.0])                 # world y-up
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:                    # looking straight down
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])                         # rows = cam axes in world
    T = -R @ np.asarray(cam_pos, float)
    return R, T


# subject at origin; world: x right, y up, z toward the front cams
CAMS = {
    0: look_at([-0.6, 1.6, 2.5]),                   # front-left  (ceiling)
    1: look_at([+0.6, 1.6, 2.5]),                   # front-right (ceiling)
    2: look_at([-2.5, 1.6, 0.0]),                   # left side
    3: look_at([+2.5, 1.6, 0.0]),                   # right side
}
WRIST = np.array([0.15, 0.0, 0.3])                  # right hand, slightly front


def hand_markers(normal_dir):
    """Wrist/thumb/pinky ~9 cm apart spanning a palm plane with that normal."""
    n = np.asarray(normal_dir, float)
    n /= np.linalg.norm(n)
    a = np.cross(n, [0.0, 1.0, 0.0])
    if np.linalg.norm(a) < 1e-6:
        a = np.cross(n, [1.0, 0.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    return WRIST, WRIST + 0.09 * a, WRIST + 0.09 * b


def vis_all(normal_dir):
    w, t, p = hand_markers(normal_dir)
    n = palm_normal(w, t, p)
    assert n is not None
    return {v: view_visibility(n, w, R, T) for v, (R, T) in CAMS.items()}


def test_palm_normal():
    w, t, p = hand_markers([0, 0, 1])
    n = palm_normal(w, t, p)
    assert abs(abs(n @ [0, 0, 1]) - 1) < 1e-9, "normal should be +-z"
    assert palm_normal(w, t, np.full(3, np.nan)) is None, "missing marker"
    assert palm_normal(w, w + [0.09, 0, 0], w + [0.18, 0, 0]) is None, \
        "collinear markers must be degenerate"
    print("  palm_normal ok")


def test_visibility_front_vs_side():
    # palm faces the front cams -> front vis ~1, side cams see it edge-on
    vis = vis_all([0, 0, 1])
    # ceiling cams look ~30 deg down at the origin, so ~0.78 not ~1
    assert vis[0] > 0.7 and vis[1] > 0.7, f"front cams should see flat: {vis}"
    assert vis[2] < 0.35 and vis[3] < 0.35, f"side cams edge-on: {vis}"
    # palm rotated to face the LEFT side cam -> ranking flips
    vis = vis_all([-1, 0, 0])
    assert vis[2] > max(vis[0], vis[1]), f"left cam should win: {vis}"
    # fingers pointing at cam0 = line of sight IN the palm plane (palm down)
    w = WRIST
    ray = (np.asarray([-0.6, 1.6, 2.5]) - w)
    a = ray / np.linalg.norm(ray)                    # fingers along the ray
    b = np.cross([0, 1, 0], a)
    b /= np.linalg.norm(b)                           # in-plane, horizontal-ish
    n = palm_normal(w, w + 0.09 * a, w + 0.09 * b)
    v0 = view_visibility(n, w, *CAMS[0])
    assert v0 < 0.1, f"fingers-at-camera must be near edge-on: {v0}"
    print("  visibility front/side/fingers-at-cam ok")


def test_in_frame_fraction():
    assert in_frame_fraction((640, 360), 200, 1280, 720) == 1.0
    assert abs(in_frame_fraction((0, 360), 200, 1280, 720) - 0.5) < 1e-9
    assert in_frame_fraction((-200, 360), 200, 1280, 720) == 0.0
    assert in_frame_fraction((np.nan, 360), 200, 1280, 720) is None
    print("  in_frame_fraction ok")


def test_ranking_and_hysteresis():
    # equal size: visibility decides
    vis = vis_all([0, 0, 1])
    cands = {v: {"size": 200.0, "vis": vis[v], "conf": 1.0, "in_frame": 1.0}
             for v in CAMS}
    sel, order, sc = select_views(cands, 2)
    assert set(sel) == {0, 1}, f"front cams should be picked: {order} {sc}"
    # big-but-edge-on loses to slightly smaller flat view
    cands = {0: {"size": 150.0, "vis": 0.95, "conf": 1.0, "in_frame": 1.0},
             2: {"size": 200.0, "vis": 0.05, "conf": 1.0, "in_frame": 1.0}}
    sel, _, _ = select_views(cands, 1)
    assert sel == [0], "flat 150px crop must beat edge-on 200px crop"
    # occluded view (low NLF conf) demoted despite big crop
    cands = {0: {"size": 160.0, "vis": 0.8, "conf": 0.9, "in_frame": 1.0},
             3: {"size": 200.0, "vis": 0.8, "conf": 0.05, "in_frame": 1.0}}
    sel, _, _ = select_views(cands, 1)
    assert sel == [0], "body-occluded view must lose"
    # hysteresis: a marginally better newcomer does NOT evict
    cands = {0: {"size": 190.0, "vis": 0.8, "conf": 1.0, "in_frame": 1.0},
             1: {"size": 200.0, "vis": 0.8, "conf": 1.0, "in_frame": 1.0}}
    sel, _, _ = select_views(cands, 1, prev={0}, switch_bonus=1.15)
    assert sel == [0], "hysteresis should hold the current view"
    sel, _, _ = select_views(cands, 1, prev={0}, switch_bonus=1.0)
    assert sel == [1], "without bonus the bigger crop wins"
    # missing components are neutral -> falls back to biggest-crop
    cands = {v: {"size": s, "vis": None, "conf": None, "in_frame": None}
             for v, s in ((0, 120.0), (1, 90.0), (2, 200.0), (3, 150.0))}
    sel, _, _ = select_views(cands, 2)
    assert set(sel) == {2, 3}, "no 3D -> biggest crops win"
    print("  ranking + hysteresis ok")


def test_ray_dependence():
    # same normal, but a wrist far off-axis changes each cam's ray: visibility
    # must use the actual camera->wrist ray, not the optical axis
    w = np.array([0.0, 0.0, 2.4])                    # right next to front cams
    n = np.array([0.0, 0.0, 1.0])
    v_near = view_visibility(n, w, *CAMS[0])
    v_far = view_visibility(n, WRIST, *CAMS[0])
    assert v_near < v_far, "oblique ray must reduce visibility"
    print("  ray dependence ok")


if __name__ == "__main__":
    test_palm_normal()
    test_visibility_front_vs_side()
    test_in_frame_fraction()
    test_ranking_and_hysteresis()
    test_ray_dependence()
    print("all view-selection tests passed")
