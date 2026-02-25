from __future__ import annotations

from dataclasses import asdict

from .config import AppConfig, DEFAULT_CONFIG
from .io_utils import load_image
from .pose import PoseEstimator, extract_torso_anchors
from .silhouette import auto_find_levels, ellipse_circumference, get_body_mask, get_height_pixels, measure_width
from .visualization import draw_measurement_lines, save_outputs


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

    result = {}
    for key in LEVELS:
        width_px, x_left, x_right = measure_width(
            mask,
            y_levels[key],
            anchors["torso_x_min"],
            anchors["torso_x_max"],
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


def run_pipeline(config: AppConfig = DEFAULT_CONFIG) -> dict:
    front_image = load_image(config.inputs.front_image, config.silhouette.max_image_side)
    side_image = load_image(config.inputs.side_image, config.silhouette.max_image_side)

    estimator = PoseEstimator(config.model.yolo_model_path)
    front_keypoints = estimator.get_keypoints(front_image)
    side_keypoints = estimator.get_keypoints(side_image)
    if front_keypoints is None or side_keypoints is None:
        raise RuntimeError("YOLO11x-pose не обнаружил человека на одном из изображений.")

    front_anchors = extract_torso_anchors(front_keypoints, front_image.shape[1], config.silhouette.torso_x_margin)
    side_anchors = extract_torso_anchors(side_keypoints, side_image.shape[1], config.silhouette.torso_x_margin)

    front_mask = get_body_mask(front_image)
    side_mask = get_body_mask(side_image)

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

    front_debug = draw_measurement_lines(front_image, front_sizes, (0, 220, 255))
    side_debug = draw_measurement_lines(side_image, side_sizes, (255, 160, 0))
    save_outputs(
        config.output.result_dir,
        front_debug,
        side_debug,
        front_mask,
        side_mask,
        config.output.front_debug_name,
        config.output.side_debug_name,
        config.output.front_mask_name,
        config.output.side_mask_name,
    )

    return {
        "config": asdict(config),
        "scale_cm_per_px": px_to_cm,
        "front_height_px": front_height_px,
        "side_height_px": side_height_px,
        "measurements": final,
    }