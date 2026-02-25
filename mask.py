import cv2
import numpy as np
from rembg import remove
from PIL import Image


def get_body_mask(image: np.ndarray) -> np.ndarray:
    """
    Создаёт бинарную маску тела: удаляет фон через нейросеть rembg.

    Принимает BGR-изображение (numpy uint8).
    Возвращает uint8 массив (H, W) со значениями 0 (фон) или 255 (тело).
    """
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    output    = remove(pil_image)
    alpha     = np.array(output)[:, :, 3]
    return np.where(alpha > 128, 255, 0).astype(np.uint8)


def get_height_pixels(binary_mask: np.ndarray) -> int:
    """
    Возвращает высоту тела в пикселях по маске (от верхнего до нижнего белого пикселя).
    Бросает ValueError если маска пустая.
    """
    y_coords = np.where(binary_mask == 255)[0]
    if len(y_coords) == 0:
        raise ValueError("Маска тела пустая — тело не найдено.")
    return int(np.max(y_coords) - np.min(y_coords))
