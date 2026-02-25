from config       import IMAGE_PATH, USER_HEIGHT_CM
from image_utils  import load_image
from pose         import get_keypoints, extract_torso_anchors
from mask         import get_body_mask, get_height_pixels
from measurements import compute_all
from visualizer   import draw_results, save_results, print_results


def run(image_path: str = IMAGE_PATH, height_cm: float = USER_HEIGHT_CM):
    """
      1. Загрузка и нормализация фото
      2. Определение скелета (YOLO Pose)
      3. Создание маски тела (rembg)
      4. Расчёт масштаба (пиксель → сантиметр) по реальному росту
      5. Автоматический поиск уровней груди / талии / бёдер по силуэту
      6. Измерение ширины и расчёт обхватов
      7. Визуализация и сохранение результатов
    """
    print("Загружаем фото...")
    image = load_image(image_path)
    img_h, img_w = image.shape[:2]

    print("Ищем скелет (YOLO Pose)...")
    keypoints = get_keypoints(image)
    if keypoints is None:
        print("Ошибка: человек не найден на фото.")
        return
    print("Скелет найден!")

    print("Создаём маску тела (rembg)... может занять 10-30 сек")
    binary_mask = get_body_mask(image)
    print("Маска готова!")

    try:
        height_px = get_height_pixels(binary_mask)
    except ValueError as e:
        print(f"Ошибка: {e}")
        return

    ratio = height_cm / height_px
    print(f"Рост в пикселях: {height_px}  |  1 px = {ratio:.4f} см")

    print("Анализируем силуэт...")
    anchors = extract_torso_anchors(keypoints, img_w, img_h)
    results = compute_all(binary_mask, anchors, ratio)

    print(
        f"Найдены уровни →  "
        f"грудь: {results['chest']['y']}px  |  "
        f"талия: {results['waist']['y']}px  |  "
        f"бёдра: {results['hips']['y']}px"
    )

    print_results(results)
    debug_img = draw_results(image, results, anchors)
    save_results(debug_img, binary_mask)

if __name__ == '__main__':
    run()
