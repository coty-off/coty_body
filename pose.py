import numpy as np
from ultralytics import YOLO
from config import (
    YOLO_MODEL_PATH,
    KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER,
    KP_LEFT_HIP,      KP_RIGHT_HIP,
)


def get_keypoints(image) -> np.ndarray | None:
    """
    Запускает YOLO Pose на изображении.
    Если на фото несколько людей — выбирает ближайшего к центру кадра.

    Возвращает массив (17, 2) с координатами (x, y) в пикселях
    или None если человек не найден.
    """
    model = YOLO(YOLO_MODEL_PATH)
    results = model(image, verbose=False)

    if not results or results[0].keypoints is None:
        return None

    kps = results[0].keypoints.xy.cpu().numpy()   # (N_persons, 17, 2)
    if len(kps) == 0:
        return None

    return _pick_central_person(kps, image.shape)


def extract_torso_anchors(keypoints: np.ndarray, img_w: int, img_h: int) -> dict:
    """
    Извлекает из keypoints опорные Y и X-координаты туловища.

    Возвращает словарь:
      y_shoulder, y_hip          — средние Y плеч и бёдер
      x_shoulder_l/r, x_hip_l/r — X-координаты плеч и бёдер
      torso_x_min, torso_x_max   — X-границы зоны поиска (с запасом)
    """
    from config import TORSO_X_MARGIN

    y_shoulder = (keypoints[KP_LEFT_SHOULDER][1] + keypoints[KP_RIGHT_SHOULDER][1]) / 2
    y_hip      = (keypoints[KP_LEFT_HIP][1]      + keypoints[KP_RIGHT_HIP][1])      / 2

    x_shoulder_l = keypoints[KP_LEFT_SHOULDER][0]
    x_shoulder_r = keypoints[KP_RIGHT_SHOULDER][0]
    x_hip_l      = keypoints[KP_LEFT_HIP][0]
    x_hip_r      = keypoints[KP_RIGHT_HIP][0]

    margin = img_w * TORSO_X_MARGIN
    all_x  = [x_shoulder_l, x_shoulder_r, x_hip_l, x_hip_r]

    torso_x_min = max(0,     min(all_x) - margin)
    torso_x_max = min(img_w, max(all_x) + margin)

    return {
        'y_shoulder':   y_shoulder,
        'y_hip':        y_hip,
        'x_shoulder_l': x_shoulder_l,
        'x_shoulder_r': x_shoulder_r,
        'x_hip_l':      x_hip_l,
        'x_hip_r':      x_hip_r,
        'torso_x_min':  torso_x_min,
        'torso_x_max':  torso_x_max,
    }

def _pick_central_person(kps: np.ndarray, shape: tuple) -> np.ndarray:
    """Выбирает человека, чей центр масс ближе всего к центру кадра."""
    h, w = shape[:2]
    cx, cy = w / 2, h / 2
    best_idx, best_dist = 0, float('inf')

    for i, person_kps in enumerate(kps):
        visible = person_kps[person_kps[:, 0] > 0]
        if len(visible) == 0:
            continue
        mx, my = visible[:, 0].mean(), visible[:, 1].mean()
        dist = (mx - cx) ** 2 + (my - cy) ** 2
        if dist < best_dist:
            best_dist = dist
            best_idx  = i

    return kps[best_idx]
