import cv2
import numpy as np
import torch
import os
import sys

sys.path.insert(0, "/home/users/theo/code/Fast-SAM-3D-Body")

from notebook.utils import setup_sam_3d_body

CHECKPOINT_DIR = "/home/users/theo/code/checkpoints/sam-3d-body-dinov3"
MHR_PATH       = os.path.join(CHECKPOINT_DIR, "assets", "mhr_model.pt")

CAM_INTRINSICS = torch.tensor([[
    [2726.9,    0.0, 1080.0],
    [   0.0, 2726.9, 1920.0],
    [   0.0,    0.0,    1.0],
]], dtype=torch.float32)


class FastSAM3DExtractor:
    """Batch 3D body keypoint extractor using SAM-3D-Body."""

    def __init__(self, gpu_id: int = 1):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SAM-3D-Body on GPU {gpu_id} ({self.device})...")

        self.estimator = setup_sam_3d_body(
            local_checkpoint_path=CHECKPOINT_DIR,
            local_mhr_path=MHR_PATH,
            detector_name="yolo11",
            detector_model="yolo11m-pose.pt",
            fov_name="moge2",
            device=self.device,
        )
        print("Model loaded.")

    def process_video(self, video_path: str, output_npy_path: str = "joints_fastsam3d.npy"):
        """Extract 3D keypoints for every frame of a video.

        Args:
            video_path:      Path to input video.
            output_npy_path: Destination for the (T, 70, 3) joints array.

        Returns:
            np.ndarray of shape (T, 70, 3) — 3D joint positions per frame.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS)
        print(f"Processing: {video_path}  ({total_frames} frames @ {fps:.1f} FPS)")

        sequence_joints    = []
        sequence_joints_2d = []
        skipped_frames     = []

        with torch.no_grad():
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                outputs = self.estimator.process_one_image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )

                if not outputs:
                    print(f"  Warning: no person at frame {frame_idx}, skipping.")
                    skipped_frames.append(frame_idx)
                    continue

                sequence_joints.append(outputs[0]["pred_keypoints_3d"])
                sequence_joints_2d.append(outputs[0]["pred_keypoints_2d"])

                if frame_idx % 10 == 0:
                    print(f"  Frame {frame_idx}/{total_frames}")

        cap.release()

        if not sequence_joints:
            raise RuntimeError("No joints extracted — check video or detection threshold.")

        kinematics_matrix = np.stack(sequence_joints)
        os.makedirs(os.path.dirname(os.path.abspath(output_npy_path)), exist_ok=True)
        np.save(output_npy_path, kinematics_matrix)

        kp2d_path = output_npy_path.replace(".npy", "_2d.npy")
        np.save(kp2d_path, np.stack(sequence_joints_2d))

        print(f"\nDone! shape={kinematics_matrix.shape}, skipped={len(skipped_frames)} frames")
        print(f"  3D: {output_npy_path}")
        print(f"  2D: {kp2d_path}")
        return kinematics_matrix


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--video",  required=True)
    p.add_argument("--output", default="/home/users/theo/code/data/sam3d_outputs/joints_fastsam3d.npy")
    p.add_argument("--gpu",    type=int, default=1)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    FastSAM3DExtractor(gpu_id=args.gpu).process_video(args.video, args.output)
