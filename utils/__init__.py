"""Shared pure-Python libraries for the live/offline tracking pipeline.

Small, dependency-light helpers imported by the entry-point scripts at the repo
root (kept out of those scripts so they stay focused):

- ``hand_view_select``       — per-hand best-view selection (numpy only)
- ``visualize_skeleton_video`` — skeleton drawing on frames (cv2 + numpy)
"""
