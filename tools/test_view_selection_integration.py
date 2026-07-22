#!/usr/bin/env python3
"""Integration test of cosmik_hand_demo's per-hand view selection WITHOUT GPU.

Stubs torch + the heavy model modules (same approach as test_stereo_hands.py)
and verifies the risky bookkeeping of the new flat (view, hand) decoder batch:
  - want routing: only selected (view, hand) crops become batch entries
  - output de-interleave: entry i of the fake forward lands back on the right
    (view, hand) slot in `per`, left hand un-flipped, crops slot correct
  - HandWorker._select picks the expected cameras on a synthetic 4-cam rig
"""

import contextlib
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ───────────────────────── fake torch (numpy-backed) ─────────────────────────
class FT:  # FakeTensor
    def __init__(self, a):
        self.a = np.asarray(a)

    shape = property(lambda s: s.a.shape)

    def __getitem__(self, i):
        return FT(self.a[i])

    def __mul__(self, k):
        return FT(self.a * k)

    def permute(self, *dims):
        return FT(np.transpose(self.a, dims))

    def byte(self):
        return FT(self.a.astype(np.uint8))

    def float(self):
        return FT(self.a.astype(np.float32))

    detach = cpu = lambda s: s

    def numpy(self):
        return self.a


torch = types.ModuleType("torch")
torch.no_grad = contextlib.nullcontext
torch.cat = lambda ts, dim=0: FT(np.concatenate([t.a for t in ts], axis=dim))
torch.tensor = lambda a, dtype=None: FT(np.asarray(a, np.float32))
torch.float32 = "float32"
torch.cuda = types.SimpleNamespace(synchronize=lambda: None)
torch.backends = types.SimpleNamespace(
    cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
    cudnn=types.SimpleNamespace(allow_tf32=False),
)
sys.modules["torch"] = torch

# ─────────────────── stub the heavy imports of cosmik_hand_demo ───────────────
sd = types.ModuleType("stream_demo")
sd._emit_server, sd._EMIT = (lambda port: None), {}
sys.modules["stream_demo"] = sd

nb = types.ModuleType("notebook")
nbu = types.ModuleType("notebook.utils")
nbu.setup_sam_3d_body = lambda **kw: None
nb.utils = nbu
sys.modules["notebook"], sys.modules["notebook.utils"] = nb, nbu

rd = types.ModuleType("rerun_demo")
rd._H_WRIST = 20
rd.HAND_SRC = list(range(21))  # identity keypoint mapping
rd.L_ELBOW, rd.L_WRIST, rd.R_ELBOW, rd.R_WRIST = 7, 9, 8, 10
rd._quiet = contextlib.nullcontext
rd._hand_box_v2 = lambda k17, w, e, args: None  # replaced via _hand_box_view
sys.modules["rerun_demo"] = rd

s3 = types.ModuleType("sam_3d_body")
s3m = types.ModuleType("sam_3d_body.models")
s3ma = types.ModuleType("sam_3d_body.models.meta_arch")
s3mb = types.ModuleType("sam_3d_body.models.meta_arch.sam3d_body")


def fake_prep(
    img, lbox, rbox, cam_int, output_size=(64, 64), padding=0.9, device="cuda"
):
    """Batches shaped like the real _prepare_hand_batches_gpu: (1, N=1, ...);
    the img content encodes (box_x1) so tiles are traceable to their box."""
    oh, ow = output_size

    def batch(box):
        return {
            "img": FT(np.full((1, 1, 3, oh, ow), box[0, 0] / 255.0)),
            "img_size": FT(np.zeros((1, 1, 2))),
            "ori_img_size": FT(np.zeros((1, 1, 2))),
            "bbox_center": FT(np.zeros((1, 1, 2))),
            "bbox_scale": FT(np.zeros((1, 1, 2))),
            "bbox": FT(box[None]),
            "affine_trans": FT(np.zeros((1, 1, 2, 3))),
            "mask": FT(np.zeros((1, 1, 1, oh, ow))),
            "mask_score": FT(np.zeros((1, 1))),
            "person_valid": FT(np.ones((1, 1))),
            "cam_int": FT(cam_int.a if isinstance(cam_int, FT) else cam_int),
        }

    return batch(lbox), batch(rbox), lbox


s3mb._prepare_hand_batches_gpu = fake_prep
s3.models, s3m.meta_arch, s3ma.sam3d_body = s3m, s3ma, s3mb
for n, m in (
    ("sam_3d_body", s3),
    ("sam_3d_body.models", s3m),
    ("sam_3d_body.models.meta_arch", s3ma),
    ("sam_3d_body.models.meta_arch.sam3d_body", s3mb),
):
    sys.modules[n] = m

import cosmik_hand_demo as chd


class FakeModel:
    """forward_step: entry i predicts all-21 keypoints at (100*i+7, 100*i+9),
    3D at z = i — so routing errors are unmissable."""

    cfg = types.SimpleNamespace(
        MODEL=types.SimpleNamespace(IMAGE_SIZE=(64, 64))
    )

    def _initialize_batch(self, bh):
        self.B = bh["img"].shape[0]
        assert bh["cam_int"].shape == (self.B, 3, 3), bh["cam_int"].shape

    def forward_step(self, bh, decoder_type):
        assert decoder_type == "hand"
        B = self.B
        p2 = np.zeros((B, 21, 2), np.float32)
        p3 = np.zeros((B, 21, 3), np.float32)
        for i in range(B):
            p2[i] = [100 * i + 7, 100 * i + 9]
            p3[i, :, 2] = i
        return {
            "mhr_hand": {
                "pred_keypoints_2d": FT(p2),
                "pred_keypoints_3d": FT(p3),
            }
        }

    class _D(dict):
        pass


# make the boxes deterministic: right box x1 = 10+view, left box x1 = 50+view
def fake_box_view(k17, wrist_i, elbow_i, args, side_px=None):
    base = 10.0 if wrist_i == rd.R_WRIST else 50.0
    v = k17[0, 0]  # view id smuggled in k17[0]
    return np.array([base + v, 0, base + v + 20, 20], np.float32)


chd._hand_box_view = fake_box_view

ARGS = types.SimpleNamespace(hand_res=64)


def run_step(views, want):
    W = 640
    frames = {v: np.zeros((480, W, 3), np.uint8) for v in views}
    k17s = {v: np.full((17, 2), float(v), np.float32) for v in views}
    cam_ints = {v: FT(np.eye(3)[None] * (v + 1)) for v in views}
    return (
        chd._hand_decoder_step_views(
            FakeModel(), frames, k17s, cam_ints, ARGS, sides=None, want=want
        ),
        W,
    )


def test_routing():
    views = [0, 1, 2]
    want = {0: (True, False), 1: (True, True), 2: (False, True)}
    (per, tms, crops), W = run_step(views, want)
    # entry order: v0-r, v1-r, v1-l, v2-l  → predictions 0,1,2,3
    kp_r0, kp_l0 = per[0][0], per[0][1]
    assert kp_l0 is None and per[0][5] is None, "L not selected in view 0"
    assert kp_r0[0, 0] == 7, f"v0 R should be entry 0: {kp_r0[0]}"
    assert per[1][0][0, 0] == 107, "v1 R should be entry 1"
    # left = entry 2 → x un-flipped: W - 207 - 1
    assert per[1][1][0, 0] == W - 207 - 1, f"v1 L un-flip: {per[1][1][0]}"
    assert per[2][0] is None and per[2][4] is None, "R not selected in view 2"
    assert per[2][1][0, 0] == W - 307 - 1, "v2 L should be entry 3"
    # 3D: z encodes the entry, left x sign flipped (x was 0 → -0 ok, check z)
    assert per[1][2][0, 2] == 1 and per[1][3][0, 2] == 2, "3D routing"
    # boxes land on the right slots
    assert per[0][4][0] == 10 and per[1][5][0] == 51 and per[2][5][0] == 52
    # crops: view 0 only R slot, view 2 only L slot; content encodes box x1
    assert crops[0][1] is None and crops[2][0] is None
    assert crops[0][0][0, 0, 0] == 10 and crops[2][1][0, 0, 0] == 52
    print("  flat-batch routing + un-flip + crops ok")


def test_no_want_all_views():
    views = [0, 1]
    (per, _, crops), W = run_step(views, None)
    for v in views:
        assert per[v][0] is not None and per[v][1] is not None
        assert crops[v][0] is not None and crops[v][1] is not None
    # entry order v0-r, v0-l, v1-r, v1-l
    assert per[0][0][0, 0] == 7 and per[1][0][0, 0] == 207
    print("  want=None decodes both hands everywhere ok")


def test_handworker_select():
    # 4-cam rig: front cams 0,1 see the palm flat; side cams 2,3 edge-on
    import tools.test_view_selection as rig

    Rs = [rig.CAMS[i][0] for i in range(4)]
    Ts = [rig.CAMS[i][1] for i in range(4)]
    Ks = [np.eye(3)] * 4
    Ds = [np.zeros(5)] * 4
    wri, thu, pin = rig.hand_markers([0, 0, 1])
    kp3d = np.full((chd.NMK, 3), np.nan, np.float32)
    for pair, val in ((chd.R_WRIST_PAIR, wri),):
        kp3d[pair[0]] = kp3d[pair[1]] = val
    kp3d[chd.R_PALM[0]], kp3d[chd.R_PALM[1]] = thu, pin
    # left hand markers missing → neutral vis; give cam 2,3 bigger conf/size? no:
    res = {
        "kp3d": kp3d,
        "kp2d": np.full((4, chd.NMK, 2), 640.0, np.float32),
        "scores": np.ones((4, chd.NMK), np.float32),
    }
    hw = chd.HandWorker.__new__(chd.HandWorker)  # no thread init needed
    hw.calib = (Ks, Ds, Rs, Ts)
    hw.args = types.SimpleNamespace(
        cap_width=1280, cap_height=720, hand_switch_bonus=1.15
    )
    hw.topk = 2
    hw._sel_prev = {"r": set(), "l": set()}
    sides = {v: (150.0, 150.0) for v in range(4)}  # equal size everywhere
    want, order = hw._select(res, [0, 1, 2, 3], sides)
    r_sel = {v for v, (r, _) in want.items() if r}
    assert r_sel == {
        0,
        1,
    }, f"front cams must win for the flat right hand: {order}"
    assert hw._sel_prev["r"] == {0, 1}
    # palm rotated toward the LEFT side cam → cam 2 must enter the selection
    wri2, thu2, pin2 = rig.hand_markers([-1, 0, 0])
    kp3d[chd.R_WRIST_PAIR[0]] = kp3d[chd.R_WRIST_PAIR[1]] = wri2
    kp3d[chd.R_PALM[0]], kp3d[chd.R_PALM[1]] = thu2, pin2
    want, order = hw._select(res, [0, 1, 2, 3], sides)
    assert order["r"][0] == 2, f"left side cam should now rank first: {order}"
    # no 3D and no metric sides -> the 2D projected-forearm fallback must
    # still rank "biggest crop" first (view 2 gets a 3x longer forearm)
    kp2d = np.full((4, chd.NMK, 2), np.nan, np.float32)
    for v in range(4):
        L = 150.0 if v == 2 else 50.0
        for m in chd.R_WRIST_PAIR:
            kp2d[v, m] = (320.0, 240.0)
        for m in chd.R_ELBOW_PAIR:
            kp2d[v, m] = (320.0 + L, 240.0)
    res2 = {
        "kp3d": np.full((chd.NMK, 3), np.nan, np.float32),
        "kp2d": kp2d,
        "scores": np.ones((4, chd.NMK), np.float32),
    }
    hw._sel_prev = {"r": set(), "l": set()}
    want, order = hw._select(res2, [0, 1, 2, 3], None)
    assert order["r"][0] == 2, f"2D forearm fallback must rank size: {order}"
    print("  HandWorker._select on the 4-cam rig ok")


def test_handworker_run_once():
    """One full HandWorker.run() iteration end-to-end (fake cams/body/model):
    selection → flat decoder batch → per-hand sources → result dict."""
    import tools.test_view_selection as rig

    chd._STOP.clear()
    Rs = [rig.CAMS[i][0] for i in range(4)]
    Ts = [rig.CAMS[i][1] for i in range(4)]
    calib = ([np.eye(3)] * 4, [np.zeros(5)] * 4, Rs, Ts)
    wri, thu, pin = rig.hand_markers([0, 0, 1])
    kp3d = np.full((chd.NMK, 3), np.nan, np.float32)
    kp3d[chd.R_WRIST_PAIR[0]] = kp3d[chd.R_WRIST_PAIR[1]] = wri
    kp3d[chd.R_PALM[0]], kp3d[chd.R_PALM[1]] = thu, pin
    res = {
        "kp3d": kp3d,
        "kp2d": np.full((4, chd.NMK, 2), 320.0, np.float32),
        "scores": np.ones((4, chd.NMK), np.float32),
    }

    class FakeCam:
        def latest(self):
            return np.zeros((480, 640, 3), np.uint8), 0.0, 1

    class FakeBody:
        calls = 0

        def latest(self):
            FakeBody.calls += 1
            if FakeBody.calls >= 2:
                chd._STOP.set()
            return res, 1

    args = types.SimpleNamespace(
        hand_topk=-1,
        hand_switch_bonus=1.15,
        cap_width=640,
        cap_height=480,
        hand_size_m=0,
        hand_size_frac=0,
        det_thr=0.3,
        hand_reproj_thr=15.0,
        hand_res=64,
    )
    hw = chd.HandWorker(
        [FakeCam()] * 4,
        FakeBody(),
        FakeModel(),
        calib,
        args,
        [0, 1, 2, 3],
        hand_cam=0,
    )
    assert hw.topk == 2, "auto topk on 4 views"
    hw.run()  # exits via _STOP
    chd._STOP.clear()
    r = hw.result
    assert r is not None and hw.n == 1
    assert len(r["sel"]["r"]) == 2 and len(r["sel"]["l"]) == 2
    assert set(r["sel"]["r"]) == {
        0,
        1,
    }, f"flat right hand → front cams: {r['sel']}"
    assert r["src_r"] in r["sel"]["r"] and r["src_l"] in r["sel"]["l"]
    assert r["kp_r"] is not None and r["kp_l"] is not None
    # kp2d_views populated exactly on the selected (view, hand) slots
    got_r = set(
        np.flatnonzero(np.isfinite(r["kp2d_views"][:, :21]).all((1, 2)))
    )
    got_l = set(
        np.flatnonzero(np.isfinite(r["kp2d_views"][:, 21:]).all((1, 2)))
    )
    assert got_r == set(r["sel"]["r"]) and got_l == set(r["sel"]["l"])
    assert r["X_r"].shape == (21, 3) and "tri_ms" in r["tms"]
    print("  HandWorker.run() one iteration ok")


def test_selection_dynamics():
    """Palm rotates from facing the front cams to facing the LEFT side cam
    over 120 simulated body frames: the right hand's selection must migrate
    to include cam 2, with few switches (hysteresis, no flapping)."""
    import tools.test_view_selection as rig

    chd._STOP.clear()
    Rs = [rig.CAMS[i][0] for i in range(4)]
    Ts = [rig.CAMS[i][1] for i in range(4)]
    calib = ([np.eye(3)] * 4, [np.zeros(5)] * 4, Rs, Ts)
    N = 120
    seq = []
    for i in range(N):
        # hold front for 40 frames, then rotate 90 deg toward -x over 40
        ang = min(1.0, max(0.0, (i - 40) / 40.0)) * (np.pi / 2)
        wri, thu, pin = rig.hand_markers([-np.sin(ang), 0.0, np.cos(ang)])
        kp3d = np.full((chd.NMK, 3), np.nan, np.float32)
        kp3d[chd.R_WRIST_PAIR[0]] = kp3d[chd.R_WRIST_PAIR[1]] = wri
        kp3d[chd.R_PALM[0]], kp3d[chd.R_PALM[1]] = thu, pin
        seq.append(
            {
                "kp3d": kp3d,
                "kp2d": np.full((4, chd.NMK, 2), 320.0, np.float32),
                "scores": np.ones((4, chd.NMK), np.float32),
            }
        )

    class FakeCam:
        def latest(self):
            return np.zeros((480, 640, 3), np.uint8), 0.0, 1

    class FakeBody:
        n = 0

        def latest(self):
            if FakeBody.n >= N:
                chd._STOP.set()
                return seq[-1], N
            FakeBody.n += 1
            return seq[FakeBody.n - 1], FakeBody.n

    sels = []
    orig = chd.HandWorker._select

    def rec(self, res, views, sides):
        want, order = orig(self, res, views, sides)
        sels.append(tuple(sorted(v for v, (r, _) in want.items() if r)))
        return want, order

    chd.HandWorker._select = rec
    try:
        args = types.SimpleNamespace(
            hand_topk=-1,
            hand_switch_bonus=1.15,
            cap_width=640,
            cap_height=480,
            hand_size_m=0,
            hand_size_frac=0,
            det_thr=0.3,
            hand_reproj_thr=15.0,
            hand_res=64,
        )
        hw = chd.HandWorker(
            [FakeCam()] * 4,
            FakeBody(),
            FakeModel(),
            calib,
            args,
            [0, 1, 2, 3],
            hand_cam=0,
        )
        hw.run()
    finally:
        chd.HandWorker._select = orig
        chd._STOP.clear()
    assert len(sels) == N
    assert sels[0] == (0, 1), f"front phase must use the front cams: {sels[0]}"
    assert (
        2 in sels[-1]
    ), f"left side cam must join after the rotation: {sels[-1]}"
    switches = sum(a != b for a, b in zip(sels, sels[1:]))
    assert 1 <= switches <= 4, f"selection flaps: {switches} switches"
    print(f"  selection dynamics ok (final {sels[-1]}, {switches} switch(es))")


if __name__ == "__main__":
    test_routing()
    test_no_want_all_views()
    test_handworker_select()
    test_handworker_run_once()
    test_selection_dynamics()
    print("all integration tests passed")
