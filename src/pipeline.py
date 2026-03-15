from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Any

import numpy as np

from .config import AppConfig, DEFAULT_CONFIG
from .io_utils import load_image
from .pose import PoseEstimator, extract_torso_anchors
from .silhouette import (
    auto_find_levels,
    ellipse_circumference,
    get_body_mask,
    get_height_pixels,
    get_torso_x_extent_robust,
    measure_width,
)
from .measurement import classify_body_type


LEVELS = ("chest", "waist", "hips")


def _collect_sizes(mask, anchors, ratio, silhouette_cfg):
    y_chest, y_waist, y_hips = auto_find_levels(
        mask,
        anchors["y_shoulder"],
        anchors["y_hip"],
        anchors["torso_x_min"],
        anchors["torso_x_max"],
        silhouette_cfg.scan_step_px,
        silhouette_cfg.smooth_window,
        silhouette_cfg.mask_window_px,
        silhouette_cfg.hip_search_extra,
    )
    y_levels = {"chest": y_chest, "waist": y_waist, "hips": y_hips}

    x_bounds = {
        "chest": (
            anchors.get("x_shoulder_min", anchors["torso_x_min"]),
            anchors.get("x_shoulder_max", anchors["torso_x_max"]),
        ),
        "waist": (anchors["torso_x_min"], anchors["torso_x_max"]),
        "hips":  (anchors["torso_x_min"], anchors["torso_x_max"]),
    }

    result = {}
    for key in LEVELS:
        x_min, x_max = x_bounds[key]
        width_px, x_left, x_right = measure_width(
            mask,
            y_levels[key],
            x_min,
            x_max,
            silhouette_cfg.mask_window_px,
        )
        result[key] = {
            "y": y_levels[key],
            "x_left": x_left,
            "x_right": x_right,
            "size_px": width_px,
            "size_cm": width_px * ratio,
        }
    return result


def run_pipeline_from_arrays(
    front_image: np.ndarray,
    side_image: np.ndarray,
    config: AppConfig,
    estimator: PoseEstimator | None = None,
) -> Dict[str, Any]:
    if estimator is None:
        estimator = PoseEstimator(config.model.yolo_model_path)

    front_keypoints = estimator.get_keypoints(front_image)
    side_keypoints = estimator.get_keypoints(side_image)
    if front_keypoints is None or side_keypoints is None:
        raise RuntimeError("YOLO26x-pose не обнаружил человека на одном из изображений.")

    front_anchors = extract_torso_anchors(
        front_keypoints, front_image.shape[1], config.silhouette.torso_x_margin
    )
    side_anchors_raw = extract_torso_anchors(
        side_keypoints, side_image.shape[1], config.silhouette.torso_x_margin
    )

    front_mask = get_body_mask(front_image)
    side_mask = get_body_mask(side_image)

    # вычисление X-диапазон для профиля из маски
    side_margin_px = int(side_image.shape[1] * config.silhouette.torso_x_margin)
    y_s = side_anchors_raw["y_shoulder"]
    y_h = side_anchors_raw["y_hip"]
    y_end_side = y_h + (y_h - y_s) * config.silhouette.hip_search_extra

    x_min_side, x_max_side = get_torso_x_extent_robust(
        side_mask,
        y_start=y_s,
        y_end=y_end_side,
        margin_px=side_margin_px,
        percentile=50.0,
    )

    side_anchors = {
        **side_anchors_raw,
        "torso_x_min": x_min_side,
        "torso_x_max": x_max_side,
        "x_shoulder_min": x_min_side,
        "x_shoulder_max": x_max_side,
    }

    front_height_px = get_height_pixels(front_mask)
    side_height_px = get_height_pixels(side_mask)
    px_to_cm = config.inputs.user_height_cm / ((front_height_px + side_height_px) / 2)

    front_sizes = _collect_sizes(front_mask, front_anchors, px_to_cm, config.silhouette)
    side_sizes = _collect_sizes(side_mask, side_anchors, px_to_cm, config.silhouette)

    final = {}
    for level in LEVELS:
        width_cm = front_sizes[level]["size_cm"]
        depth_cm = side_sizes[level]["size_cm"]
        final[level] = {
            "front_width_cm": width_cm,
            "side_depth_cm": depth_cm,
            "circumference_cm": ellipse_circumference(width_cm, depth_cm),
            "front_y": front_sizes[level]["y"],
            "side_y": side_sizes[level]["y"],
        }

    body_type = classify_body_type(
        chest_cm=final["chest"]["circumference_cm"],
        waist_cm=final["waist"]["circumference_cm"],
        hip_cm=final["hips"]["circumference_cm"],
    )

    return {
        "config": asdict(config),
        "scale_cm_per_px": px_to_cm,
        "front_height_px": front_height_px,
        "side_height_px": side_height_px,
        "measurements": final,
        "body_type": body_type,
    }


def run_pipeline(config: AppConfig = DEFAULT_CONFIG) -> dict:
    front_image = load_image(config.inputs.front_image, config.silhouette.max_image_side)
    side_image = load_image(config.inputs.side_image, config.silhouette.max_image_side)
    return run_pipeline_from_arrays(front_image, side_image, config)
