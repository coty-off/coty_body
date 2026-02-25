import cv2
import numpy as np
from config import (
    COLOR_CHEST, COLOR_WAIST, COLOR_HIPS,
    COLOR_GUIDE, COLOR_DOTS,
    OUTPUT_RESULT, OUTPUT_MASK,
)


def draw_results(image: np.ndarray, results: dict, anchors: dict) -> np.ndarray:
    """
    Рисует три линии замеров и ориентиры скелета на копии изображения.

    image   — исходное BGR-изображение
    results — словарь из measurements.compute_all()
    anchors — словарь из pose.extract_torso_anchors()

    Возвращает новое изображение с разметкой.
    """
    img = image.copy()
    img_w = img.shape[1]

    _draw_guide_line(img, anchors['y_shoulder'], img_w)
    _draw_guide_line(img, anchors['y_hip'],      img_w)

    _draw_measurement(img, results['chest'],
                      f"Chest  W:{results['chest']['width_cm']:.1f}cm"
                      f"  C:{results['chest']['circ_cm']:.0f}cm",
                      COLOR_CHEST)

    _draw_measurement(img, results['waist'],
                      f"Waist  W:{results['waist']['width_cm']:.1f}cm"
                      f"  C:{results['waist']['circ_cm']:.0f}cm",
                      COLOR_WAIST)

    _draw_measurement(img, results['hips'],
                      f"Hips   W:{results['hips']['width_cm']:.1f}cm"
                      f"  C:{results['hips']['circ_cm']:.0f}cm",
                      COLOR_HIPS)

    return img


def save_results(debug_img: np.ndarray, binary_mask: np.ndarray) -> None:
    """Сохраняет итоговое фото с разметкой и маску на диск."""
    cv2.imwrite(OUTPUT_RESULT, debug_img)
    cv2.imwrite(OUTPUT_MASK,   binary_mask)
    print(f"Сохранено: {OUTPUT_RESULT}, {OUTPUT_MASK}")


def print_results(results: dict) -> None:
    """Выводит результаты в консоль."""
    print("\n─── Результаты ───────────────────────────────────")
    for part, label in [('chest', 'Грудь'), ('waist', 'Талия'), ('hips', 'Бёдра')]:
        r = results[part]
        print(f"{label}:  ширина {r['width_cm']:.1f} см  →  обхват ≈ {r['circ_cm']:.1f} см")
    print("──────────────────────────────────────────────────")
    print("Обхват — оценка по модели эллипса.")


# ── Приватные функции ──────────────────────────────────────────────────────────

def _draw_measurement(img, data: dict, label: str, color: tuple) -> None:
    """Рисует одну линию замера с точками и подписью."""
    y      = int(data['y'])
    x_left  = int(data['x_left'])
    x_right = int(data['x_right'])

    cv2.line(img,   (x_left, y), (x_right, y), color, 3)
    cv2.circle(img, (x_left,  y), 8, COLOR_DOTS, -1)
    cv2.circle(img, (x_right, y), 8, COLOR_DOTS, -1)
    cv2.putText(img, label, (x_left, y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)


def _draw_guide_line(img, y: float, img_w: int) -> None:
    """Рисует тонкую серую горизонталь на уровне ориентира скелета."""
    cv2.line(img, (0, int(y)), (img_w, int(y)), COLOR_GUIDE, 1)
