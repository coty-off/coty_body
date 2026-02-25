import math
import numpy as np
from config import (
    SCAN_STEP_PX, SMOOTH_WINDOW, MASK_WINDOW_PX,
    HIP_SEARCH_EXTRA,
    DEPTH_RATIO_CHEST, DEPTH_RATIO_WAIST, DEPTH_RATIO_HIPS,
)

def auto_find_levels(binary_mask, y_shoulder, y_hip, x_min, x_max) -> tuple:
    """
    Автоматически находит Y-уровни груди, талии и бёдер по форме силуэта.

    Ключевая идея: талия — ЛОКАЛЬНЫЙ минимум между двумя горбами.
    Она не обязана быть уже плеч: у полных людей талия может быть широкой,
    но всё равно образует «впадину» на кривой ширин между грудью и бёдрами.

    Возвращает (y_chest, y_waist, y_hips) в пикселях.
    """
    torso_h = y_hip - y_shoulder
    y_end   = y_hip + torso_h * HIP_SEARCH_EXTRA

    widths = _scan_torso_widths(binary_mask, y_shoulder, y_end, x_min, x_max)

    if len(widths) < 10:

        return (
            y_shoulder + torso_h * 0.25,
            y_shoulder + torso_h * 0.60,
            y_hip      + torso_h * 0.10,
        )

    ys, smoothed = _smooth_widths(widths)

    # Грудь: максимум в верхней половине туловища
    zone_chest = (ys >= y_shoulder + torso_h * 0.05) & \
                 (ys <= y_shoulder + torso_h * 0.50)
    y_chest = int(ys[zone_chest][np.argmax(smoothed[zone_chest])]) \
              if zone_chest.any() \
              else int(y_shoulder + torso_h * 0.20)

    # Бёдра: максимум в нижней половине (до HIP_SEARCH_EXTRA ниже бедра)
    zone_hips = (ys >= y_shoulder + torso_h * 0.50) & (ys <= y_end)
    y_hips = int(ys[zone_hips][np.argmax(smoothed[zone_hips])]) \
             if zone_hips.any() \
             else int(y_hip + torso_h * 0.10)

    # Талия: локальный минимум строго МЕЖДУ грудью и бёдрами
    zone_waist = (ys > y_chest) & (ys < y_hips)
    if zone_waist.any() and zone_waist.sum() > 3:
        y_waist = int(ys[zone_waist][np.argmin(smoothed[zone_waist])])
    else:
        y_waist = int((y_chest + y_hips) / 2)

    return y_chest, y_waist, y_hips


def measure_width(binary_mask, y_px, x_min, x_max) -> tuple:
    """
    Измеряет ширину туловища в пикселях на уровне y_px.

    Внутри диапазона [x_min, x_max] находит наибольший непрерывный сегмент —
    это туловище (оно всегда шире рук в профиль).

    Возвращает (ширина_px, x_левый, x_правый).
    """
    y   = int(np.clip(y_px, MASK_WINDOW_PX, binary_mask.shape[0] - MASK_WINDOW_PX - 1))
    row = np.max(binary_mask[y - MASK_WINDOW_PX : y + MASK_WINDOW_PX + 1, :], axis=0).copy()

    xi, xa = _clip_x(x_min, x_max, binary_mask.shape[1])
    row[:xi] = 0
    row[xa:]  = 0

    x0, x1 = _find_largest_segment(row)
    if x0 == x1 == 0:
        return 0, 0, 0
    return int(x1 - x0), int(x0), int(x1)


def circumference(width_cm: float, depth_ratio: float) -> float:
    a = width_cm / 2
    b = a * depth_ratio
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def compute_all(binary_mask, anchors: dict, ratio: float) -> dict:
    x_min = anchors['torso_x_min']
    x_max = anchors['torso_x_max']
    y_shoulder = anchors['y_shoulder']
    y_hip      = anchors['y_hip']

    y_chest, y_waist, y_hips = auto_find_levels(
        binary_mask, y_shoulder, y_hip, x_min, x_max
    )

    chest_px, cl, cr = measure_width(binary_mask, y_chest, x_min, x_max)
    waist_px, wl, wr = measure_width(binary_mask, y_waist, x_min, x_max)
    hips_px,  hl, hr = measure_width(binary_mask, y_hips,  x_min, x_max)

    chest_w = chest_px * ratio
    waist_w = waist_px * ratio
    hips_w  = hips_px  * ratio

    return {
        'chest': {
            'y': y_chest, 'x_left': cl, 'x_right': cr,
            'width_cm': chest_w,
            'circ_cm':  circumference(chest_w, DEPTH_RATIO_CHEST),
        },
        'waist': {
            'y': y_waist, 'x_left': wl, 'x_right': wr,
            'width_cm': waist_w,
            'circ_cm':  circumference(waist_w, DEPTH_RATIO_WAIST),
        },
        'hips': {
            'y': y_hips, 'x_left': hl, 'x_right': hr,
            'width_cm': hips_w,
            'circ_cm':  circumference(hips_w, DEPTH_RATIO_HIPS),
        },
    }

def _scan_torso_widths(binary_mask, y_start, y_end, x_min, x_max) -> dict:
    widths  = {}
    y_start = int(max(y_start, MASK_WINDOW_PX))
    y_end   = int(min(y_end,   binary_mask.shape[0] - MASK_WINDOW_PX - 1))
    xi, xa  = _clip_x(x_min, x_max, binary_mask.shape[1])

    for y in range(y_start, y_end, SCAN_STEP_PX):
        row = np.max(binary_mask[y - MASK_WINDOW_PX : y + MASK_WINDOW_PX + 1, :], axis=0).copy()
        row[:xi] = 0
        row[xa:]  = 0
        x0, x1   = _find_largest_segment(row)
        widths[y] = x1 - x0

    return widths


def _smooth_widths(widths_dict: dict, window: int = SMOOTH_WINDOW):
    ys       = np.array(sorted(widths_dict.keys()))
    vals     = np.array([widths_dict[y] for y in ys], dtype=float)
    kernel   = np.ones(window) / window
    smoothed = np.convolve(vals, kernel, mode='same')
    return ys, smoothed


def _find_largest_segment(row: np.ndarray) -> tuple:
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
    best = max(segments, key=lambda s: s[1] - s[0])
    return best[0], best[1]


def _clip_x(x_min, x_max, width: int) -> tuple:
    return int(np.clip(x_min, 0, width)), int(np.clip(x_max, 0, width))
