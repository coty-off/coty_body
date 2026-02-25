from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image
from rembg import remove


def get_body_mask(image: np.ndarray) -> np.ndarray:
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    output = remove(pil_image)
    alpha = np.array(output)[:, :, 3]
    return np.where(alpha > 128, 255, 0).astype(np.uint8)


def get_height_pixels(binary_mask: np.ndarray) -> int:
    y_coords = np.where(binary_mask == 255)[0]
    if len(y_coords) == 0:
        raise ValueError("Маска тела пустая — тело не найдено.")
    return int(np.max(y_coords) - np.min(y_coords))


def auto_find_levels(binary_mask, y_shoulder, y_hip, x_min, x_max, scan_step_px, smooth_window, mask_window_px, hip_search_extra):
    torso_h = y_hip - y_shoulder
    y_end = y_hip + torso_h * hip_search_extra
    widths = _scan_torso_widths(binary_mask, y_shoulder, y_end, x_min, x_max, scan_step_px, mask_window_px)

    if len(widths) < 10:
        return y_shoulder + torso_h * 0.25, y_shoulder + torso_h * 0.60, y_hip + torso_h * 0.10

    ys, smoothed = _smooth_widths(widths, smooth_window)

    zone_chest = (ys >= y_shoulder + torso_h * 0.05) & (ys <= y_shoulder + torso_h * 0.50)
    zone_hips = (ys >= y_shoulder + torso_h * 0.50) & (ys <= y_end)

    y_chest = int(ys[zone_chest][np.argmax(smoothed[zone_chest])]) if zone_chest.any() else int(y_shoulder + torso_h * 0.20)
    y_hips = int(ys[zone_hips][np.argmax(smoothed[zone_hips])]) if zone_hips.any() else int(y_hip + torso_h * 0.10)

    zone_waist = (ys > y_chest) & (ys < y_hips)
    y_waist = int(ys[zone_waist][np.argmin(smoothed[zone_waist])]) if zone_waist.any() and zone_waist.sum() > 3 else int((y_chest + y_hips) / 2)

    return y_chest, y_waist, y_hips


def measure_width(binary_mask, y_px, x_min, x_max, mask_window_px) -> tuple[int, int, int]:
    y = int(np.clip(y_px, mask_window_px, binary_mask.shape[0] - mask_window_px - 1))
    row = np.max(binary_mask[y - mask_window_px : y + mask_window_px + 1, :], axis=0).copy()

    xi, xa = _clip_x(x_min, x_max, binary_mask.shape[1])
    row[:xi] = 0
    row[xa:] = 0

    x0, x1 = _find_largest_segment(row)
    if x0 == x1 == 0:
        return 0, 0, 0
    return int(x1 - x0), int(x0), int(x1)


def ellipse_circumference(width_cm: float, depth_cm: float) -> float:
    a = width_cm / 2
    b = depth_cm / 2
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def _scan_torso_widths(binary_mask, y_start, y_end, x_min, x_max, scan_step_px, mask_window_px):
    widths = {}
    y_start = int(max(y_start, mask_window_px))
    y_end = int(min(y_end, binary_mask.shape[0] - mask_window_px - 1))
    xi, xa = _clip_x(x_min, x_max, binary_mask.shape[1])

    for y in range(y_start, y_end, scan_step_px):
        row = np.max(binary_mask[y - mask_window_px : y + mask_window_px + 1, :], axis=0).copy()
        row[:xi] = 0
        row[xa:] = 0
        x0, x1 = _find_largest_segment(row)
        widths[y] = x1 - x0

    return widths


def _smooth_widths(widths_dict: dict, window: int):
    ys = np.array(sorted(widths_dict.keys()))
    vals = np.array([widths_dict[y] for y in ys], dtype=float)
    kernel = np.ones(window) / window
    smoothed = np.convolve(vals, kernel, mode="same")
    return ys, smoothed


def _find_largest_segment(row: np.ndarray) -> tuple[int, int]:
    segments = []
    in_seg, seg_start = False, 0
    for x, val in enumerate(row):
        if val == 255 and not in_seg:
            in_seg, seg_start = True, x
        elif val != 255 and in_seg:
            in_seg = False
            segments.append((seg_start, x - 1))
    if in_seg:
        segments.append((seg_start, len(row) - 1))
    if not segments:
        return 0, 0
    return max(segments, key=lambda s: s[1] - s[0])


def _clip_x(x_min, x_max, width: int) -> tuple[int, int]:
    return int(np.clip(x_min, 0, width)), int(np.clip(x_max, 0, width))