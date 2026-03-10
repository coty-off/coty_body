"""
Пайплайн обучения регрессионной модели для предсказания обхватов тела
из силуэтов анфас и профиль + рост человека.

Научное обоснование подхода:
    Weng et al. (2021) — Body measurement estimation from photos
    Tong Yao et al. (2023) — Body shape classification from 2D photos

Входные данные:
    - datasets/mask/          — бинарные маски силуэтов анфас (PNG)
    - datasets/mask_left/     — бинарные маски силуэтов профиль (PNG)
    - datasets/measurements.csv    — реальные обхваты (chest, waist, hip)
    - datasets/hwg_metadata.csv    — рост (height_cm), вес, пол
    - datasets/subject_to_photo_map.csv — связь subject_id ↔ photo_id

Выход:
    - models/measurement_regressor.joblib — обученная модель
    - models/scaler.joblib               — нормализатор признаков
    - результаты валидации в консоли
"""

import os
import numpy as np
import pandas as pd
import cv2
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error
import joblib

# ── Пути к данным ────────────────────────────────────────────────────────────
BASE_DIR = r"C:\Users\kroko\PycharmProjects\COTY_body\datasets"
MASK_DIR = os.path.join(BASE_DIR, "mask")
MASK_L_DIR = os.path.join(BASE_DIR, "mask_left")
PHOTO_MAP = os.path.join(BASE_DIR, "subject_to_photo_map.csv")
MEASURE_CSV = os.path.join(BASE_DIR, "measurements.csv")
HWG_CSV = os.path.join(BASE_DIR, "hwg_metadata.csv")
MODEL_DIR = r"C:\Users\kroko\PycharmProjects\COTY_body\models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Параметр: сколько точек семплировать из профиля высоты ──────────────────
# 100 точек — достаточная детализация и при этом небольшой вектор признаков.
# Увеличение до 200 почти не даёт прироста качества, но замедляет обучение.
N_POINTS = 100


def extract_width_profile(mask_path: str) -> np.ndarray | None:
    """
    Извлекает нормализованный профиль ширин из бинарной маски.

    Идея: для каждого из N_POINTS уровней по вертикали (равномерно
    распределённых от верхней до нижней границы силуэта) измеряем ширину
    белой области в пикселях и делим на полную высоту силуэта.

    Такая нормализация делает профиль инвариантным к росту человека
    и расстоянию от него до камеры.

    Возвращает вектор длиной N_POINTS или None, если файл не найден/пуст.
    """
    if not os.path.exists(mask_path):
        return None

    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Бинаризуем — пиксели силуэта должны быть белыми (>127)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Находим вертикальные границы силуэта (убираем пустые строки сверху/снизу)
    rows = np.any(binary > 0, axis=1)
    if not rows.any():
        return None
    y_top, y_bottom = np.where(rows)[0][[0, -1]]
    silhouette_height = y_bottom - y_top
    if silhouette_height < 10:  # слишком маленький силуэт — скорее всего баг
        return None

    # Семплируем N_POINTS равномерных уровней внутри силуэта
    levels = np.linspace(y_top, y_bottom, N_POINTS, dtype=int)
    widths = []
    for y in levels:
        row = binary[y, :]
        cols = np.where(row > 0)[0]
        if len(cols) >= 2:
            width = (cols[-1] - cols[0]) / silhouette_height  # нормируем!
        else:
            width = 0.0
        widths.append(width)

    return np.array(widths, dtype=np.float32)


def build_dataset():
    """
    Строит матрицу признаков X и матрицу целевых переменных y.

    Каждая строка — одна фотография (photo_id).
    Признаки: [профиль_анфас(100) | профиль_профиль(100) | рост_нормированный(1)]
    Цели:     [chest_cm, waist_cm, hip_cm]

    Также возвращает массив subject_id для правильного разбиения на train/val
    (нельзя, чтобы фото одного человека были и в train, и в val).
    """
    print("Загружаем таблицы...")
    photo_map = pd.read_csv(PHOTO_MAP)  # subject_id, photo_id
    measures = pd.read_csv(MEASURE_CSV)  # subject_id + обхваты
    hwg = pd.read_csv(HWG_CSV)  # subject_id, height_cm, weight_kg, gender

    # Посмотрим на столбцы, чтобы понять структуру
    print("Столбцы measurements.csv:", measures.columns.tolist())
    print("Столбцы hwg_metadata.csv:", hwg.columns.tolist())

    # Объединяем всё через subject_id
    df = photo_map.merge(measures, on="subject_id", how="inner")
    df = df.merge(hwg[["subject_id", "height_cm", "gender"]], on="subject_id", how="inner")

    # Оставляем только женщин — FFIT разработан для женских фигур
    df = df[df["gender"] == "female"].reset_index(drop=True)
    print(f"Записей после фильтрации по gender=female: {len(df)}")

    # Определяем названия столбцов с обхватами — они могут отличаться в датасете
    # Ищем столбцы, содержащие 'chest', 'waist', 'hip' в названии
    chest_col = [c for c in measures.columns if "chest" in c.lower()][0]
    waist_col = [c for c in measures.columns if "waist" in c.lower()][0]
    hip_col = [c for c in measures.columns if "hip" in c.lower()][0]
    print(f"Используем столбцы: {chest_col}, {waist_col}, {hip_col}")

    X_list, y_list, groups = [], [], []
    skipped = 0

    for _, row in df.iterrows():
        pid = row["photo_id"]
        mask_front_path = os.path.join(MASK_DIR, f"{pid}.png")
        mask_side_path = os.path.join(MASK_L_DIR, f"{pid}.png")

        profile_front = extract_width_profile(mask_front_path)
        profile_side = extract_width_profile(mask_side_path)

        # Пропускаем если хотя бы одна маска не нашлась или пуста
        if profile_front is None or profile_side is None:
            skipped += 1
            continue

        # Нормируем рост: делим на 200 см (типичный максимум),
        # чтобы привести к одному диапазону с нормированными ширинами
        height_norm = row["height_cm"] / 200.0

        # Итоговый вектор признаков для одного фото
        features = np.concatenate([profile_front, profile_side, [height_norm]])
        targets = np.array([row[chest_col], row[waist_col], row[hip_col]], dtype=np.float32)

        # Базовая санитарная проверка — если обхваты нереальные, пропускаем
        if any(np.isnan(targets)) or any(targets < 40) or any(targets > 200):
            skipped += 1
            continue

        X_list.append(features)
        y_list.append(targets)
        groups.append(row["subject_id"])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    groups = np.array(groups)

    print(f"Итого примеров: {len(X)}, пропущено: {skipped}")
    print(f"Размер матрицы признаков X: {X.shape}  (примеры × признаки)")
    return X, y, groups


def train_and_evaluate(X, y, groups):
    """
    Обучает регрессор и оценивает качество на валидационном наборе.

    Важный нюанс разбиения: используем GroupShuffleSplit с группами по
    subject_id. Это гарантирует, что все фото одного человека попадают
    ЛИБО в train, ЛИБО в val — но никогда в оба. Без этого модель просто
    «запоминала» бы конкретных людей, и метрики были бы нечестными.

    В качестве модели используем GradientBoostingRegressor — он хорошо
    работает на табличных данных с выраженными нелинейными зависимостями,
    не требует большого датасета (в отличие от нейросетей) и интерпретируем.
    """
    # Нормализуем признаки — важно для корректной работы многих моделей
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Разбиваем 80/20 по субъектам, а не по фото
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X_scaled, y, groups))

    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    print(f"\nTrain: {len(X_train)} примеров, Val: {len(X_val)} примеров")
    print(f"Уникальных субъектов в train: {len(set(groups[train_idx]))}")
    print(f"Уникальных субъектов в val:   {len(set(groups[val_idx]))}")

    # Обёртка MultiOutputRegressor позволяет обучить одну модель на три цели
    # одновременно — внутри она просто строит три независимых регрессора
    print("\nОбучаем модель (это может занять 1–3 минуты)...")
    model = MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=200,  # количество деревьев
            max_depth=4,  # глубина каждого дерева — ограничиваем чтобы не переобучиться
            learning_rate=0.05,  # медленное обучение = лучшая генерализация
            subsample=0.8,  # стохастический градиентный бустинг — снижает дисперсию
            random_state=42
        )
    )
    model.fit(X_train, y_train)

    # Оцениваем качество
    y_pred = model.predict(X_val)
    targets = ["chest_cm", "waist_cm", "hip_cm"]

    print("\n── Результаты на валидационном наборе ─────────────────────────────")
    print(f"{'Цель':<12} {'MAE (см)':>10}  {'Интерпретация'}")
    print("-" * 60)
    for i, name in enumerate(targets):
        mae = mean_absolute_error(y_val[:, i], y_pred[:, i])
        # Для справки: ручное измерение рулеткой даёт погрешность ~1-2 см,
        # приемлемый результат для фото-метода — около 3-5 см
        quality = "✓ хорошо" if mae < 5 else ("△ приемлемо" if mae < 8 else "✗ плохо")
        print(f"{name:<12} {mae:>10.2f}  {quality}")

    return model, scaler


def classify_body_type(chest, waist, hip):
    """FFIT-формулы (Lee et al., 2007), пороги конвертированы в сантиметры."""
    IN = 2.54
    T1, T2, T3, T4 = 1 * IN, 3.6 * IN, 9 * IN, 10 * IN

    bh = chest - hip
    hb = hip - chest
    bw = chest - waist
    hw = hip - waist

    if abs(bh) <= T1 and (bw >= T3 or hw >= T3):
        return "Hourglass"
    if T1 < bh < T4 and bw >= T3:
        return "Top Hourglass"
    if T2 <= hb < T4 and hw >= T3:
        return "Bottom Hourglass"
    if hb >= T2 and 0 <= hw < T3:
        return "Triangle"
    if bh >= T2 and bw < T3 and hw >= 0:
        return "Inverted Triangle"
    if hb < T2 and bh < T2 and 0 <= bw < T3 and 0 <= hw < T4:
        return "Rectangle"
    return "Undefined"


def predict_for_new_photo(model, scaler, front_mask_path, side_mask_path, height_cm):
    """
    Пример использования обученной модели для нового пользователя.
    Принимает пути к маскам и рост — возвращает обхваты и тип фигуры.
    """
    profile_front = extract_width_profile(front_mask_path)
    profile_side = extract_width_profile(side_mask_path)

    if profile_front is None or profile_side is None:
        return None, None

    features = np.concatenate([profile_front, profile_side, [height_cm / 200.0]])
    features_scaled = scaler.transform(features.reshape(1, -1))
    predictions = model.predict(features_scaled)[0]

    chest, waist, hip = predictions
    body_type = classify_body_type(chest, waist, hip)

    return {"chest_cm": round(float(chest), 1),
            "waist_cm": round(float(waist), 1),
            "hip_cm": round(float(hip), 1),
            "body_type": body_type}, predictions


if __name__ == "__main__":
    # 1. Собираем датасет из масок + измерений
    X, y, groups = build_dataset()

    # 2. Обучаем и оцениваем модель
    model, scaler = train_and_evaluate(X, y, groups)

    # 3. Сохраняем модель и нормализатор для использования в основном приложении
    model_path = os.path.join(MODEL_DIR, "measurement_regressor.joblib")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\nМодель сохранена: {model_path}")
    print(f"Нормализатор сохранён: {scaler_path}")