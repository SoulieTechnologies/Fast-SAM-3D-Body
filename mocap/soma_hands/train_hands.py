"""Train a SOMA model for our 21-marker HAND layout (Strategy A).

Thin, version-matched wrapper over soma.train.train_soma_multiple.train_multiple_soma
— mirrors src/tutorials/run_soma_on_soma_dataset.ipynb with our values. The
hand-specific overrides live in the soma_train_cfg dict below; configs/
soma_train_hands.yaml documents the same choices.

    conda activate soma
    python train_hands.py \
        --work-base $WORK --support-base $SUPPORT \
        --layout $WORK/data/clear_hands/superset.json \
        --expr-id clear_hands_A_v1 --num-gpus 1 --num-cpus 4 --smoke

--layout may be our JSON superset OR a labelled c3d frame (SOMA runs MoSh++ to
extract the placement from a c3d). --smoke does one fast_dev_run iteration to
validate wiring before the multi-hour real run.

RUN ON crslab (GPU). NOT on the Mac.

CAVEAT: stock SOMA can't animate fingers (animate_hand hard-raises
NotImplementedError). This trains on flat-hand bodies (Strategy A). For real
finger poses see patches/enable_hand_animation.md (Strategy C).
"""
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work-base", required=True)
    p.add_argument("--support-base", required=True)
    p.add_argument("--layout", required=True,
                   help="hand marker layout: our superset.json or a labelled c3d")
    p.add_argument("--expr-id", default="clear_hands_A_v1")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--num-cpus", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--occ", type=int, default=5, help="max synthetic occlusions")
    p.add_argument("--ghost", type=int, default=3, help="max synthetic ghosts")
    p.add_argument("--strategy", choices=["A", "C"], default="A",
                   help="A: flat-hand (runs on stock SOMA, PLUMBING ONLY — flexed "
                        "finger angles are unreliable). C: GRAB fingers (real "
                        "finger angles) — REQUIRES the patch in "
                        "patches/enable_hand_animation.md applied first.")
    p.add_argument("--grab-dir", default=None,
                   help="GRAB npz dir (Strategy C only); run inspect_grab.py first")
    p.add_argument("--smoke", action="store_true",
                   help="one fast_dev_run iteration to check wiring, no real train")
    args = p.parse_args()

    if args.strategy == "C" and not args.grab_dir:
        p.error("--strategy C needs --grab-dir (and the hand-animation patch applied)")

    from soma.train.train_soma_multiple import train_multiple_soma

    # (occ, ghost, real_fraction, synt_fraction) — 0% real, 100% synthetic
    soma_data_settings = [(args.occ, args.ghost, 0.0, 1.0)]

    soma_train_cfg = {
        "soma.expr_id": args.expr_id,
        "dirs.support_base_dir": args.support_base,
        "dirs.work_base_dir": args.work_base,

        # our hand layout (JSON superset or labelled c3d):
        "data_parms.mocap_dataset.marker_layout_fnames": [args.layout],
        # we don't have the AMASS marker-noise model -> disable it:
        "data_parms.mocap_dataset.amass_marker_noise_model.enable": False,

        # dense small markerset -> more layout variants generalise better:
        "data_parms.marker_dataset.num_marker_layout_augmentation": 8,
        "data_parms.marker_dataset.num_random_vid_ring": 3,

        "moshpp_cfg_override.moshpp.verbosity": 1,
        "moshpp_cfg_override.dirs.support_base_dir": args.support_base,

        "train_parms.batch_size": args.batch_size,
        "trainer.num_gpus": args.num_gpus,
        "train_parms.num_workers": args.num_cpus,
        "trainer.fast_dev_run": bool(args.smoke),
    }

    if args.strategy == "A":
        # PLUMBING ONLY: flat hands, GRAB unused. Flexed-finger angles unreliable.
        soma_train_cfg["data_parms.body_dataset.animate_hand"] = False
        soma_train_cfg["data_parms.body_dataset.animate_face"] = False
    else:
        # Strategy C: real GRAB finger poses (needs the hand-animation patch).
        soma_train_cfg["data_parms.body_dataset.animate_hand"] = True
        soma_train_cfg["data_parms.body_dataset.grab_dir"] = args.grab_dir
        soma_train_cfg["data_parms.marker_dataset.enable_rnd_vid_on_face_hands"] = True
        print("[Strategy C] animate_hand=True — ensure patches/"
              "enable_hand_animation.md is APPLIED, else SOMA raises "
              "NotImplementedError. Run inspect_grab.py first.")

    print(f"[{'SMOKE' if args.smoke else 'TRAIN'}/{args.strategy}] expr={args.expr_id} "
          f"layout={args.layout} occ={args.occ} ghost={args.ghost}")
    train_multiple_soma(
        soma_data_settings=soma_data_settings,
        soma_train_cfg=soma_train_cfg,
    )
    print("done. data_id = OC_%02d_G_%02d_real_000_synt_100 (used by label_hands.py)"
          % (args.occ, args.ghost))


if __name__ == "__main__":
    main()
