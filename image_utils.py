import cv2
from config import MAX_IMAGE_SIDE


def load_image(image_path: str):
    """
    Загружает фото и приводит его к стандартному виду:
      - всегда BGR uint8 3-канальный (IMREAD_COLOR отсекает alpha)
      - сторона не больше MAX_IMAGE_SIDE пикселей

    Возвращает numpy-массив (H, W, 3) или бросает FileNotFoundError.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Не удалось открыть файл: {image_path}")

    image = _resize_if_needed(image)
    return image


def _resize_if_needed(image):
    h, w = image.shape[:2]
    if max(h, w) > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / max(h, w)
        image = cv2.resize(
            image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )
    return image
