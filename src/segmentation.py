from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

KP_NOSE          = 0
KP_LEFT_EYE      = 1
KP_RIGHT_EYE     = 2
KP_LEFT_EAR      = 3
KP_RIGHT_EAR     = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER= 6
KP_LEFT_ELBOW    = 7
KP_RIGHT_ELBOW   = 8
KP_LEFT_WRIST    = 9
KP_RIGHT_WRIST   = 10
KP_LEFT_HIP      = 11
KP_RIGHT_HIP     = 12
KP_LEFT_KNEE     = 13
KP_RIGHT_KNEE    = 14
KP_LEFT_ANKLE    = 15
KP_RIGHT_ANKLE   = 16

PALETTE: dict[str, tuple[int, int, int]] = {
    "head":       ( 86, 180, 233),   # голубой
    "torso":      ( 34, 139,  34),   # зелёный
    "left_arm":   (255, 140,   0),   # оранжевый
    "right_arm":  (214,  39,  40),   # красный
    "left_leg":   (148, 103, 189),   # фиолетовый
    "right_leg":  ( 23, 190, 207),   # бирюзовый
}

LABELS: dict[str, str] = {
    "head":       "Head",
    "torso":      "Torso",
    "left_arm":   "Left Arm",
    "right_arm":  "Right Arm",
    "left_leg":   "Left Leg",
    "right_leg":  "Right Leg",
}


@dataclass
class SegmentMask:
    name: str
    mask: np.ndarray          # uint8, 0/255
    color: tuple[int, int, int]
    centroid: tuple[int, int]

def draw_body_segments(
    image: np.ndarray,
    keypoints: np.ndarray,
    body_mask: np.ndarray,
    alpha: float = 0.40,
    draw_skeleton: bool = True,
    draw_labels: bool = True,
) -> np.ndarray:

    kp = keypoints  # shape (17, 2)
    h, w = image.shape[:2]

    segments = _build_segment_masks(kp, body_mask, h, w)

    canvas = image.copy()
    overlay = image.copy()

    for seg in segments:
        color = PALETTE[seg.name]
        overlay[seg.mask == 255] = color

        contours, _ = cv2.findContours(seg.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, thickness=2)

    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

    if draw_skeleton:
        _draw_skeleton(canvas, kp)

    _draw_keypoints(canvas, kp)

    if draw_labels:
        for seg in segments:
            if seg.centroid[0] > 0:
                _draw_label(canvas, LABELS[seg.name], seg.centroid, PALETTE[seg.name])

    canvas = _draw_legend(canvas)

    return canvas

def _build_segment_masks(
    kp: np.ndarray,
    body_mask: np.ndarray,
    h: int,
    w: int,
) -> list[SegmentMask]:
    y_nose     = _ky(kp, KP_NOSE)
    y_shoulder = _avg_y(kp, KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER)
    y_hip      = _avg_y(kp, KP_LEFT_HIP, KP_RIGHT_HIP)

    y_knee     = _avg_y(kp, KP_LEFT_KNEE, KP_RIGHT_KNEE)
    y_ankle    = _avg_y(kp, KP_LEFT_ANKLE, KP_RIGHT_ANKLE)
    if y_knee == 0:
        y_knee = y_hip + (y_hip - y_shoulder) * 0.90
    if y_ankle == 0:
        y_ankle = y_hip + (y_hip - y_shoulder) * 1.80

    x_ls = _kx(kp, KP_LEFT_SHOULDER);   x_rs = _kx(kp, KP_RIGHT_SHOULDER)
    x_lh = _kx(kp, KP_LEFT_HIP);        x_rh = _kx(kp, KP_RIGHT_HIP)
    torso_x_min = min(x_ls, x_rs, x_lh, x_rh)
    torso_x_max = max(x_ls, x_rs, x_lh, x_rh)
    x_center    = (torso_x_min + torso_x_max) // 2

    torso_margin = max(12, int((torso_x_max - torso_x_min) * 0.08))

    # 1. HEAD
    head_mask = _strip_mask(body_mask, 0, int(y_shoulder))

    # 2. TORSO
    torso_mask = _strip_mask(body_mask, int(y_shoulder), int(y_hip),
                             x_min=torso_x_min - torso_margin,
                             x_max=torso_x_max + torso_margin)

    # 3 & 4. ARMS
    # Руки — по бокам от торса и выше линии бёдер
    left_arm_mask  = _strip_mask(body_mask, int(y_shoulder * 0.90), int(y_hip),
                                  x_max=torso_x_min - torso_margin + 1)
    right_arm_mask = _strip_mask(body_mask, int(y_shoulder * 0.90), int(y_hip),
                                  x_min=torso_x_max + torso_margin - 1)

    # Если запястья обнаружены ниже бёдер (руки опущены вниз) — расширяем
    y_lw = _ky(kp, KP_LEFT_WRIST);  y_rw = _ky(kp, KP_RIGHT_WRIST)
    if y_lw > y_hip and y_lw > 0:
        extra = _strip_mask(body_mask, int(y_hip), int(y_lw) + 10,
                             x_max=torso_x_min - torso_margin + 1)
        left_arm_mask = cv2.bitwise_or(left_arm_mask, extra)
    if y_rw > y_hip and y_rw > 0:
        extra = _strip_mask(body_mask, int(y_hip), int(y_rw) + 10,
                             x_min=torso_x_max + torso_margin - 1)
        right_arm_mask = cv2.bitwise_or(right_arm_mask, extra)

    # 5 & 6. LEGS
    left_leg_mask  = _strip_mask(body_mask, int(y_hip), h,
                                  x_min=0, x_max=x_center)
    right_leg_mask = _strip_mask(body_mask, int(y_hip), h,
                                  x_min=x_center, x_max=w)

    raw = {
        "head":      head_mask,
        "torso":     torso_mask,
        "left_arm":  left_arm_mask,
        "right_arm": right_arm_mask,
        "left_leg":  left_leg_mask,
        "right_leg": right_leg_mask,
    }

    segments = []
    for name, mask in raw.items():
        centroid = _centroid(mask)
        segments.append(SegmentMask(name=name, mask=mask, color=PALETTE[name], centroid=centroid))

    return segments

def _strip_mask(
    body_mask: np.ndarray,
    y_top: int,
    y_bot: int,
    x_min: int | None = None,
    x_max: int | None = None,
) -> np.ndarray:
    h, w = body_mask.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    y_top = max(0, y_top)
    y_bot = min(h, y_bot)
    xi = 0    if x_min is None else int(np.clip(x_min, 0, w))
    xa = w    if x_max is None else int(np.clip(x_max, 0, w))
    out[y_top:y_bot, xi:xa] = body_mask[y_top:y_bot, xi:xa]
    return out

def _centroid(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.where(mask == 255)
    if len(xs) == 0:
        return (0, 0)
    return (int(xs.mean()), int(ys.mean()))


def _kx(kp: np.ndarray, idx: int) -> int:
    return int(kp[idx][0]) if kp[idx][0] > 0 else 0


def _ky(kp: np.ndarray, idx: int) -> int:
    return int(kp[idx][1]) if kp[idx][1] > 0 else 0


def _avg_y(kp: np.ndarray, idx_a: int, idx_b: int) -> int:
    ya, yb = kp[idx_a][1], kp[idx_b][1]
    if ya == 0 and yb == 0:
        return 0
    if ya == 0:
        return int(yb)
    if yb == 0:
        return int(ya)
    return int((ya + yb) / 2)

_SKELETON_LINKS: list[tuple[int, int, tuple[int, int, int]]] = [
    # голова
    (KP_LEFT_EAR, KP_LEFT_EYE,       (200, 200, 200)),
    (KP_RIGHT_EAR, KP_RIGHT_EYE,     (200, 200, 200)),
    (KP_LEFT_EYE, KP_NOSE,           (200, 200, 200)),
    (KP_RIGHT_EYE, KP_NOSE,          (200, 200, 200)),
    # торс
    (KP_LEFT_SHOULDER,  KP_RIGHT_SHOULDER, (255, 255, 255)),
    (KP_LEFT_SHOULDER,  KP_LEFT_HIP,       (255, 255, 255)),
    (KP_RIGHT_SHOULDER, KP_RIGHT_HIP,      (255, 255, 255)),
    (KP_LEFT_HIP,       KP_RIGHT_HIP,      (255, 255, 255)),
    # левая рука
    (KP_LEFT_SHOULDER, KP_LEFT_ELBOW,  (255, 140, 0)),
    (KP_LEFT_ELBOW,    KP_LEFT_WRIST,  (255, 140, 0)),
    # правая рука
    (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW, (214, 39, 40)),
    (KP_RIGHT_ELBOW,    KP_RIGHT_WRIST, (214, 39, 40)),
    # левая нога
    (KP_LEFT_HIP,   KP_LEFT_KNEE,  (148, 103, 189)),
    (KP_LEFT_KNEE,  KP_LEFT_ANKLE, (148, 103, 189)),
    # правая нога
    (KP_RIGHT_HIP,   KP_RIGHT_KNEE,  (23, 190, 207)),
    (KP_RIGHT_KNEE,  KP_RIGHT_ANKLE, (23, 190, 207)),
]


def _draw_skeleton(canvas: np.ndarray, kp: np.ndarray) -> None:
    for a, b, color in _SKELETON_LINKS:
        xa, ya = int(kp[a][0]), int(kp[a][1])
        xb, yb = int(kp[b][0]), int(kp[b][1])
        if xa == 0 or ya == 0 or xb == 0 or yb == 0:
            continue
        cv2.line(canvas, (xa, ya), (xb, yb), color, thickness=2, lineType=cv2.LINE_AA)


def _draw_keypoints(canvas: np.ndarray, kp: np.ndarray) -> None:
    for idx in range(len(kp)):
        x, y = int(kp[idx][0]), int(kp[idx][1])
        if x == 0 and y == 0:
            continue
        cv2.circle(canvas, (x, y), 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 4, (30, 30, 30),    1,  lineType=cv2.LINE_AA)

def _draw_label(
    canvas: np.ndarray,
    text: str,
    pos: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float = 0.55,
    thickness: int = 1,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cx, cy = pos
    x0 = max(0, cx - tw // 2)
    y0 = max(th + 4, cy)

    pad = 4
    rect_pt1 = (x0 - pad, y0 - th - pad)
    rect_pt2 = (x0 + tw + pad, y0 + baseline + pad)
    sub = canvas[rect_pt1[1]:rect_pt2[1], rect_pt1[0]:rect_pt2[0]]
    if sub.size > 0:
        bg = np.full_like(sub, 20)
        cv2.addWeighted(bg, 0.55, sub, 0.45, 0, sub)
        canvas[rect_pt1[1]:rect_pt2[1], rect_pt1[0]:rect_pt2[0]] = sub

    cv2.putText(canvas, text, (x0, y0), font, font_scale, color, thickness, cv2.LINE_AA)

def _draw_legend(canvas: np.ndarray) -> np.ndarray:
    h, w = canvas.shape[:2]
    item_h = 24
    pad = 10
    box_w = 180
    n = len(PALETTE)
    box_h = n * item_h + pad * 2

    x0 = w - box_w - pad
    y0 = h - box_h - pad

    sub = canvas[y0:y0 + box_h, x0:x0 + box_w]
    if sub.size > 0:
        bg = np.full_like(sub, 20)
        cv2.addWeighted(bg, 0.70, sub, 0.30, 0, sub)
        canvas[y0:y0 + box_h, x0:x0 + box_w] = sub

    for i, (key, color) in enumerate(PALETTE.items()):
        iy = y0 + pad + i * item_h
        cv2.rectangle(canvas, (x0 + pad, iy + 3), (x0 + pad + 14, iy + 17), color, -1)
        cv2.putText(canvas, LABELS[key], (x0 + pad + 20, iy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1, cv2.LINE_AA)

    return canvas