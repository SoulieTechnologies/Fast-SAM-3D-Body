"""Interactive labeller for the 21 hand markers on a flat calibration frame.

Pick a clean flat-hand frame from a raw Motive take, then GUIDED-CLICK each of
the 21 fastsam3d slots in order (wrist, then thumb->pinky, each MCP->PIP->DIP->
nail). Saves <take>_seed.npz (the labelled 21x3 seed + its frame index), which
process_hand.py then tracks through the whole take to produce the angles.

    python scripts/label_seed.py data/take_gabin_0.csv
    python scripts/label_seed.py data/take_gabin_0.csv --frame 324

Controls:
    left click  assign the nearest point to the current slot (title)
    backspace   undo the last assignment
    f / g       flip the view horizontally / vertically (orient like your photo)
    s           save (once all 21 are assigned)   q  quit without saving
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils import hand_kinematics as hk  # noqa: E402
from utils import io_motive as io  # noqa: E402

SLOTS = ["wrist"] + [f"{f}_{j}" for f in hk.FINGERS
                     for j in ("mcp", "pip", "dip", "tip")]


def slot_index(name):
    if name == "wrist":
        return hk.WRIST
    f, j = name.rsplit("_", 1)
    return hk.LM[f][j]


def find_seed_frame(frames, fps, want=None):
    if want is not None:
        return want, frames[want]

    def diam(P):
        return max(np.linalg.norm(a - b) for a in P for b in P) if len(P) > 1 else 0
    for i in range(min(len(frames), int(10 * fps))):
        P = frames[i]
        if 21 <= len(P) <= 23:
            c = P.mean(0)
            keep = P[np.linalg.norm(P - c, axis=1) < 0.20]
            if len(keep) == 21 and diam(keep) < 0.30:
                return i, keep
    raise SystemExit("no clean 21-point flat frame found in the first 10 s; "
                     "pass --frame N")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--frame", type=int, default=None)
    p.add_argument("--append", default="",
                   help="add this seed to an existing (multi-)seed npz "
                        "(re-labelling a break frame from track_review)")
    p.add_argument("--out", default="")
    args = p.parse_args()

    meta, _, frames = io.read_motive_csv(args.path)
    fps = meta.get("fps") or 120.0
    fidx, P = find_seed_frame(frames, fps, args.frame)
    # crop to the hand cluster (drop stray reflections)
    c = P.mean(0)
    P = P[np.linalg.norm(P - c, axis=1) < 0.22]
    print(f"seed frame {fidx} ({fidx/fps:.2f} s), {len(P)} points to label")

    import matplotlib.pyplot as plt        # interactive backend (not Agg)
    c = P.mean(0)
    Vt = np.linalg.svd(P - c)[2]
    proj = (P - c) @ Vt.T                   # 2D hand plane = cols 0,1
    xy = proj[:, :2].copy()
    flip = [1.0, 1.0]

    assign = {}                             # slot_name -> point row in P
    cur = [0]

    fig, ax = plt.subplots(figsize=(9, 9))

    def redraw():
        ax.clear()
        x, y = xy[:, 0] * flip[0], xy[:, 1] * flip[1]
        done_pts = set(assign.values())
        ax.scatter(x, y, s=260, c="lightgray", edgecolors="k", zorder=1)
        for name, pt in assign.items():
            ax.scatter(x[pt], y[pt], s=260, c="tab:green", zorder=2)
            ax.annotate(name, (x[pt], y[pt]), fontsize=7, ha="center",
                        va="center", zorder=3)
        for k in range(len(P)):
            if k not in done_pts:
                ax.annotate(str(k), (x[k], y[k]), fontsize=8, ha="center",
                            va="center", color="gray", zorder=3)
        slot = SLOTS[cur[0]] if cur[0] < len(SLOTS) else "ALL DONE - press s"
        ax.set_title(f"[{cur[0]}/21]  CLICK: {slot}\n"
                     "(backspace=undo  f/g=flip  s=save  q=quit)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        fig.canvas.draw_idle()

    def on_click(ev):
        if ev.inaxes != ax or cur[0] >= len(SLOTS):
            return
        x, y = xy[:, 0] * flip[0], xy[:, 1] * flip[1]
        k = int(np.argmin((x - ev.xdata) ** 2 + (y - ev.ydata) ** 2))
        assign[SLOTS[cur[0]]] = k
        cur[0] += 1
        redraw()

    def on_key(ev):
        if ev.key == "backspace" and cur[0] > 0:
            cur[0] -= 1
            assign.pop(SLOTS[cur[0]], None)
            redraw()
        elif ev.key == "f":
            flip[0] *= -1
            redraw()
        elif ev.key == "g":
            flip[1] *= -1
            redraw()
        elif ev.key == "s":
            if len(assign) < 21:
                print(f"  only {len(assign)}/21 assigned — finish first")
                return
            seed = np.full((hk.N_LM, 3), np.nan)
            for name, k in assign.items():
                seed[slot_index(name)] = P[k]
            # merge with an existing (multi-)seed file if appending
            seeds, framesL = [seed], [fidx]
            src = args.append or args.out or (
                args.path.rsplit(".", 1)[0] + "_seed.npz")
            if args.append and pathlib.Path(args.append).exists():
                z = np.load(args.append)
                if "seeds" in z:
                    seeds = list(z["seeds"]) + [seed]
                    framesL = list(z["frames"].astype(int)) + [fidx]
                else:
                    seeds = [z["seed"], seed]
                    framesL = [int(z["frame"]), fidx]
            o = np.argsort(framesL)
            seeds = np.array(seeds)[o]
            framesL = np.array(framesL)[o]
            out = args.out or src
            np.savez(out, seeds=seeds, frames=framesL,
                     seed=seeds[0], frame=int(framesL[0]))
            print(f"saved {out}  ({len(seeds)} seed(s) at frames "
                  f"{list(framesL)})")
            # quick bone-length sanity
            from utils.label_hand import bone_sanity
            print("bone lengths mm [wrist-MCP, MCP-PIP, PIP-DIP, DIP-tip]:")
            for f, b in bone_sanity(seed).items():
                print(f"  {f:7s}: {[round(v, 1) for v in b]}")
            plt.close(fig)
        elif ev.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    print("Label the 21 slots by clicking; press s when done.")
    plt.show()


if __name__ == "__main__":
    main()
