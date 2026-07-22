"""Auto-label our raw hand takes with a trained SOMA model.

Thin wrapper over soma.tools.run_soma_multiple.run_soma_on_multiple_settings
(run_tasks=['soma']), mirroring the tutorial with our values. Writes a LABELLED
c3d + .pkl under the SOMA work dir; feed the c3d to soma_labeled_to_angles.py.

    conda activate soma
    python label_hands.py \
        --work-base $WORK --support-base $SUPPORT \
        --expr-id clear_hands_A_v1 --ds-name clear_hands \
        --occ 5 --ghost 3 --unit m --tracklet

--ds-name / --unit must match prepare_soma_data.py (our Motive export is METRES).
--occ/--ghost must match the trained model's data_id.

RUN ON crslab (GPU). NOT on the Mac.
"""
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work-base", required=True)
    p.add_argument("--support-base", required=True)
    p.add_argument("--expr-id", required=True)
    p.add_argument("--ds-name", default="clear_hands",
                   help="mocap dataset dir under <work>/mocap/ (from prepare_soma_data.py)")
    p.add_argument("--mocap-base-dir", default=None,
                   help="defaults to <work>/mocap")
    p.add_argument("--occ", type=int, default=5)
    p.add_argument("--ghost", type=int, default=3)
    p.add_argument("--unit", choices=["m", "mm"], default="m")
    p.add_argument("--tracklet", action="store_true",
                   help="tracklet labelling (stabler; Motive gives tracklets)")
    p.add_argument("--max-jobs", type=int, default=0,
                   help="limit parallel jobs (0 = all)")
    args = p.parse_args()

    import os.path as osp
    from soma.tools.run_soma_multiple import run_soma_on_multiple_settings
    from soma.train.soma_trainer import create_soma_data_id

    data_id = create_soma_data_id(args.occ, args.ghost, 0.0, 1.0)
    mocap_base_dir = args.mocap_base_dir or osp.join(args.work_base, "mocap")

    soma_cfg = {
        "soma.batch_size": 512,
        "dirs.support_base_dir": args.support_base,
        "mocap.unit": args.unit,                 # our Motive export = 'm'
        "soma.tracklet_labeling.enable": bool(args.tracklet),
        "save_c3d": True,
        "keep_nan_points": True,
        "remove_zero_trajectories": False,
    }
    parallel_cfg = {"randomly_run_jobs": True}
    if args.max_jobs:
        parallel_cfg["max_num_jobs"] = args.max_jobs

    print(f"[LABEL] expr={args.expr_id} data_id={data_id} ds={args.ds_name} unit={args.unit}")
    run_soma_on_multiple_settings(
        soma_expr_ids=[args.expr_id],
        soma_mocap_target_ds_names=[args.ds_name],
        soma_data_ids=[data_id],
        soma_cfg=soma_cfg,
        mocap_base_dir=mocap_base_dir,
        run_tasks=["soma"],
        mocap_ext=".c3d",
        soma_work_base_dir=args.work_base,
        parallel_cfg=parallel_cfg,
    )
    print("done. Labelled c3d under "
          f"{args.work_base}/training_experiments/{args.expr_id}/{data_id}/evaluations/...")
    print("Next: python soma_labeled_to_angles.py <labelled.c3d> --hand left")


if __name__ == "__main__":
    main()
