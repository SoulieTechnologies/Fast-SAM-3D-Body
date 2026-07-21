"""Compare mocap reference angles to the SAM3D pipeline, exploiting the
repeated iterations (boss's plan: ≥2 takes per task, 10 reps each = ≥20 reps
per task per subject).

METHODOLOGY — how the iterations are used most effectively
----------------------------------------------------------
mocap and SAM3D are CO-CAPTURED (same motion, same instant) → every rep is a
PAIRED sample. That pairing is what gives statistical power:

1. Primary metric — per-rep, per-joint RMSE of the time-aligned angle waveform.
   20 reps → a DISTRIBUTION of RMSE per joint → report mean ± 95% CI. Far more
   trustworthy than one number.
2. Bland–Altman, pooled over all frames of all reps per joint → systematic bias
   + limits of agreement (not just spread).
3. Repeatability floor — the 10 reps give the subject's own rep-to-rep spread
   (mocap std across reps). SAM3D is "as good as ground truth is stable" when
   its error ≤ that floor. The 2 takes give a session test–retest / held-out set
   and guard against a take where a marker shifted mid-session.
4. Feature agreement per rep — ROM (max−min), peak flexion, movement time per
   joint → compare the DISTRIBUTIONS (mocap vs SAM3D), i.e. does SAM3D preserve
   the range/timing, not only the instantaneous value.
5. Robustness vs occlusion — tasks are graded None→High; report accuracy as a
   function of occlusion. Reps with a mocap dropout are excluded yet N stays large.

Pipeline per (subject, task): segment each take into reps → for each rep
time-align SAM3D to mocap (clap offset + resample) → per-joint RMSE → aggregate.
"""
import numpy as np


def find_offset(y_ref, y_test, fps, max_lag_s=2.0):
    """Time offset (s) that best aligns y_test to y_ref by cross-correlation
    (use a shared joint's angle, or the clap impulse). Positive → y_test lags."""
    a = np.nan_to_num(np.asarray(y_ref, float) - np.nanmean(y_ref))
    b = np.nan_to_num(np.asarray(y_test, float) - np.nanmean(y_test))
    n = int(max_lag_s * fps)
    lags = range(-n, n + 1)
    best, blag = -np.inf, 0
    for L in lags:
        if L >= 0:
            c = np.dot(a[L:len(b)], b[:len(a) - L]) if len(a) - L > 0 else 0
        else:
            c = np.dot(a[:len(b) + L], b[-L:len(a)]) if len(b) + L > 0 else 0
        if c > best:
            best, blag = c, L
    return blag / fps


def resample(t_src, y_src, t_dst):
    """Linear resample y_src(t_src) onto t_dst; NaN outside coverage/gaps."""
    t_src = np.asarray(t_src, float)
    y = np.asarray(y_src, float)
    good = np.isfinite(y)
    if good.sum() < 2:
        return np.full(len(t_dst), np.nan)
    out = np.interp(t_dst, t_src[good], y[good], left=np.nan, right=np.nan)
    return out


def segment_reps(y, fps, n_reps, ref=None, min_sep_s=0.5):
    """Split a take into n_reps by the peaks of a driving signal `ref`
    (default: `y` itself). Returns a list of (start_idx, end_idx) covering one
    rep each (peak-to-peak midpoints). Falls back to equal division."""
    s = np.asarray(ref if ref is not None else y, float)
    s = np.nan_to_num(s, nan=np.nanmin(s))
    T = len(s)
    sep = int(min_sep_s * fps)
    # simple peak pick: local maxima above the mean, min-separated
    peaks = []
    thr = s.mean()
    i = 1
    while i < T - 1:
        if s[i] > thr and s[i] >= s[i - 1] and s[i] > s[i + 1]:
            if not peaks or i - peaks[-1] >= sep:
                peaks.append(i)
        i += 1
    if len(peaks) < n_reps:                       # fallback: equal slices
        b = np.linspace(0, T, n_reps + 1).astype(int)
        return list(zip(b[:-1], b[1:]))
    peaks = peaks[:n_reps]
    bounds = [0] + [(peaks[k] + peaks[k + 1]) // 2
                    for k in range(len(peaks) - 1)] + [T]
    return list(zip(bounds[:-1], bounds[1:]))


def rep_rmse(t_moc, a_moc, t_sam, a_sam, offset=0.0):
    """RMSE (deg) between mocap and SAM3D for one angle over one rep, after
    shifting SAM3D by `offset` and resampling onto the mocap timebase. Uses only
    frames where both are finite; returns (rmse, n_used)."""
    a_on = resample(np.asarray(t_sam) + offset, a_sam, t_moc)
    m = np.asarray(a_moc, float)
    both = np.isfinite(m) & np.isfinite(a_on)
    if both.sum() == 0:
        return np.nan, 0
    d = m[both] - a_on[both]
    return float(np.sqrt(np.mean(d ** 2))), int(both.sum())


def aggregate(rmses):
    """mean, std, 95% CI half-width of a list of per-rep RMSEs (NaN-safe)."""
    x = np.asarray([r for r in rmses if np.isfinite(r)], float)
    if x.size == 0:
        return dict(mean=np.nan, std=np.nan, ci=np.nan, n=0)
    ci = 1.96 * x.std(ddof=1) / np.sqrt(x.size) if x.size > 1 else np.nan
    return dict(mean=float(x.mean()), std=float(x.std(ddof=1) if x.size > 1
                else 0.0), ci=float(ci), n=int(x.size))


def bland_altman(a, b):
    """Pooled bias and 95% limits of agreement between two aligned series."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    bias = float(d.mean())
    sd = float(d.std(ddof=1)) if d.size > 1 else 0.0
    return dict(bias=bias, loa_low=bias - 1.96 * sd, loa_high=bias + 1.96 * sd,
                n=int(d.size))


def rom(y):
    """Range of motion (max−min) of a rep, NaN-safe."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    return float(y.max() - y.min()) if y.size else np.nan


if __name__ == "__main__":               # sanity: recover a known offset + RMSE
    fps = 120.0
    t = np.arange(0, 5, 1 / fps)
    ref = 40 * (1 - np.cos(2 * np.pi * t))
    test = np.interp(t - 0.1, t, ref) + np.random.default_rng(0).normal(0, 2, len(t))
    off = find_offset(ref, test, fps)
    r, n = rep_rmse(t, ref, t, test, offset=off)
    print(f"recovered offset {off*1000:.0f} ms (true 100 ms); RMSE {r:.1f}° over {n} fr")
