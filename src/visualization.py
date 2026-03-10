from pathlib import Path

import cv2
import numpy as np


def draw_measurement_lines(image: np.ndarray, measurements: dict, color: tuple[int, int, int]) -> np.ndarray:
    canvas = image.copy()
    for name, row in measurements.items():
        y = int(row["y"])
        x_left = int(row["x_left"])
        x_right = int(row["x_right"])
        cv2.line(canvas, (x_left, y), (x_right, y), color, 2)
        cv2.putText(
            canvas,
            f"{name}: {row['size_cm']:.1f} cm",
            (x_left, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return canvas


def save_outputs(result_dir: Path, front_debug: np.ndarray, side_debug: np.ndarray, front_mask: np.ndarray, side_mask: np.ndarray, front_debug_name: str, side_debug_name: str, front_mask_name: str, side_mask_name: str) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(result_dir / front_debug_name), front_debug)
    cv2.imwrite(str(result_dir / side_debug_name), side_debug)
    cv2.imwrite(str(result_dir / front_mask_name), front_mask)
    cv2.imwrite(str(result_dir / side_mask_name), side_mask)