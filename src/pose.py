from __future__ import annotations

import numpy as np
from ultralytics import YOLO

KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12


class PoseEstimator:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def get_keypoints(self, image: np.ndarray) -> np.ndarray | None:
        results = self.model(image, verbose=False)
        if not results or results[0].keypoints is None:
            return None

        kps = results[0].keypoints.xy.cpu().numpy()
        if len(kps) == 0:
            return None
        return _pick_central_person(kps, image.shape)


def extract_torso_anchors(keypoints: np.ndarray, img_w: int, torso_x_margin: float) -> dict:
    y_shoulder = (keypoints[KP_LEFT_SHOULDER][1] + keypoints[KP_RIGHT_SHOULDER][1]) / 2
    y_hip = (keypoints[KP_LEFT_HIP][1] + keypoints[KP_RIGHT_HIP][1]) / 2

    x_shoulder_l = keypoints[KP_LEFT_SHOULDER][0]
    x_shoulder_r = keypoints[KP_RIGHT_SHOULDER][0]
    x_hip_l = keypoints[KP_LEFT_HIP][0]
    x_hip_r = keypoints[KP_RIGHT_HIP][0]

    margin = img_w * torso_x_margin
    all_x = [x_shoulder_l, x_shoulder_r, x_hip_l, x_hip_r]

    return {
        "y_shoulder": y_shoulder,
        "y_hip": y_hip,
        "torso_x_min": max(0, min(all_x) - margin),
        "torso_x_max": min(img_w, max(all_x) + margin),
    }


def _pick_central_person(kps: np.ndarray, shape: tuple) -> np.ndarray:
    h, w = shape[:2]
    cx, cy = w / 2, h / 2
    best_idx, best_dist = 0, float("inf")

    for i, person_kps in enumerate(kps):
        visible = person_kps[person_kps[:, 0] > 0]
        if len(visible) == 0:
            continue
        mx, my = visible[:, 0].mean(), visible[:, 1].mean()
        dist = (mx - cx) ** 2 + (my - cy) ** 2
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    return kps[best_idx]