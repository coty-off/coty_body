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

    # В верхней части торса ширину часто искажают руки/подмышки,
    # поэтому ищем грудь в более узком диапазоне и с мягким
    # приоритетом к «классическому» уровню ~35% от высоты торса.
    chest_zone = (ys >= y_shoulder + torso_h * 0.20) & (ys <= y_shoulder + torso_h * 0.55)
    y_chest = _pick_weighted_extremum(
        ys,
        smoothed,
        chest_zone,
        center_y=y_shoulder + torso_h * 0.36,
        extremum="max",
        fallback_y=y_shoulder + torso_h * 0.36,
    )

    # Бёдра оставляем близко к предыдущей стратегии — максимум ширины
    # в нижней части туловища.
    hips_zone = (ys >= y_shoulder + torso_h * 0.55) & (ys <= y_end)
    y_hips = _pick_weighted_extremum(
        ys,
        smoothed,
        hips_zone,
        center_y=y_hip + torso_h * 0.08,
        extremum="max",
        fallback_y=y_hip + torso_h * 0.10,
    )

    # Талию ищем как минимум между грудью и бёдрами, но тоже с приоритетом
    # к типичной зоне, чтобы не «залипать» слишком высоко.
    min_gap = torso_h * 0.10
    waist_zone = (ys >= y_chest + min_gap) & (ys <= y_hips - min_gap)
    y_waist = _pick_weighted_extremum(
        ys,
        smoothed,
        waist_zone,
        center_y=y_shoulder + torso_h * 0.62,
        extremum="min",
        fallback_y=(y_chest + y_hips) / 2,
    )

    return y_chest, y_waist, y_hips


def _pick_weighted_extremum(ys: np.ndarray, values: np.ndarray, zone: np.ndarray, center_y: float, extremum: str, fallback_y: float) -> int:
    if not zone.any():
        return int(fallback_y)

    zone_ys = ys[zone].astype(float)
    zone_vals = values[zone].astype(float)

    spread = max((zone_ys[-1] - zone_ys[0]) / 2.0, 1.0)
    distance = np.abs(zone_ys - center_y)
    weight = np.exp(-0.5 * (distance / spread) ** 2)

    if extremum == "max":
        score = zone_vals * (0.65 + 0.35 * weight)
        idx = int(np.argmax(score))
    else:
        norm = max(np.max(zone_vals), 1.0)
        score = (zone_vals / norm) + (1.0 - weight) * 0.20
        idx = int(np.argmin(score))

    return int(zone_ys[idx])


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
