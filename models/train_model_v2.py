"""
models/train_model_v2.py

Улучшенная версия обучения с расширенным вектором признаков.

Ключевое изменение по сравнению с v1: функция extract_features()
теперь возвращает не просто "сырой" профиль ширин, а дополнительно
вычисляет явные признаки формы тела. Это помогает модели, потому что
на небольших датасетах (6000 примеров) явные признаки работают лучше,
чем надежда на то, что модель сама выучит нужные соотношения из сырых данных.

Структура итогового вектора признаков (было 201 → стало 229 чисел):
    [0:100]   — профиль ширин анфас (нормализованный)
    [100:200] — профиль глубин профиль (нормализованный)
    [200]     — рост нормализованный
    [201:229] — новые признаки формы (см. функцию _compute_shape_features)
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
BASE_DIR    = r"C:\Users\kroko\PycharmProjects\COTY_body\datasets"
MASK_DIR    = os.path.join(BASE_DIR, "mask")
MASK_L_DIR  = os.path.join(BASE_DIR, "mask_left")
PHOTO_MAP   = os.path.join(BASE_DIR, "subject_to_photo_map.csv")
MEASURE_CSV = os.path.join(BASE_DIR, "measurements.csv")
HWG_CSV     = os.path.join(BASE_DIR, "hwg_metadata.csv")
MODEL_DIR   = r"C:\Users\kroko\PycharmProjects\COTY_body\models"
os.makedirs(MODEL_DIR, exist_ok=True)

N_POINTS = 100

# Зоны тела как доли от полной высоты силуэта.
# Грудь — верхняя часть торса, талия — середина, бёдра — нижняя часть торса.
# Используем диапазоны, а не точки, потому что точное положение уровня
# варьируется от человека к человеку, а диапазон всегда накрывает нужную зону.
CHEST_ZONE = (0.20, 0.45)
WAIST_ZONE = (0.45, 0.65)
HIPS_ZONE  = (0.55, 0.80)


def extract_width_profile(mask_path: str) -> np.ndarray | None:
    """
    Извлекает нормализованный профиль ширин из бинарной маски.
    Эта функция ИДЕНТИЧНА той, что в measurement_model.py — не менять!
    """
    if not os.path.exists(mask_path):
        return None
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    rows = np.any(binary > 0, axis=1)
    if not rows.any():
        return None
    y_top, y_bottom = np.where(rows)[0][[0, -1]]
    silhouette_height = y_bottom - y_top
    if silhouette_height < 10:
        return None
    levels = np.linspace(y_top, y_bottom, N_POINTS, dtype=int)
    widths = []
    for y in levels:
        row = binary[y, :]
        cols = np.where(row > 0)[0]
        widths.append((cols[-1] - cols[0]) / silhouette_height if len(cols) >= 2 else 0.0)
    return np.array(widths, dtype=np.float32)


def _zone_indices(zone: tuple[float, float]) -> tuple[int, int]:
    """Переводит зону в долях высоты в индексы массива профиля."""
    return int(zone[0] * N_POINTS), int(zone[1] * N_POINTS)


def _compute_shape_features(front: np.ndarray, side: np.ndarray) -> np.ndarray:
    """
    Вычисляет 28 явных признаков формы тела из двух профилей.

    Почему это помогает? Представь, что модель — это студент, которому
    дали 200 чисел и просят найти "максимум в первых 45 числах минус
    минимум следующих 20". Гораздо проще дать студенту уже вычисленный
    результат. Здесь то же самое: мы предвычисляем соотношения, которые
    напрямую связаны с тем, как FFIT-формулы определяют тип фигуры.

    Группы признаков:
        [0:6]   — максимум ширины в зонах груди/талии/бёдер (анфас и профиль)
        [6:12]  — минимум ширины в тех же зонах (анфас и профиль)
        [12:18] — перепады chest-waist, hip-waist, chest-hip (анфас и профиль)
        [18:24] — std ширин внутри каждой зоны — "кривизна" контура
        [24:27] — отношение глубины к ширине в каждой зоне (форма сечения)
        [27]    — средняя "рябь" силуэта анфас (мера гладкости)
    """
    features = []
    zones = [CHEST_ZONE, WAIST_ZONE, HIPS_ZONE]

    # Максимум ширины в каждой зоне — прямой аналог "chest width", "hip width"
    for profile in (front, side):
        for zone in zones:
            s, e = _zone_indices(zone)
            features.append(float(np.max(profile[s:e])) if e > s else 0.0)

    # Минимум ширины — особенно важен для талии
    for profile in (front, side):
        for zone in zones:
            s, e = _zone_indices(zone)
            features.append(float(np.min(profile[s:e])) if e > s else 0.0)

    # Перепады между зонами — это нормализованные аналоги
    # "chest-waist drop" и "hip-waist drop" из FFIT-формул
    for profile in (front, side):
        s_c, e_c = _zone_indices(CHEST_ZONE)
        s_w, e_w = _zone_indices(WAIST_ZONE)
        s_h, e_h = _zone_indices(HIPS_ZONE)
        chest_val = float(np.max(profile[s_c:e_c])) if e_c > s_c else 0.0
        waist_val = float(np.min(profile[s_w:e_w])) if e_w > s_w else 0.0
        hips_val  = float(np.max(profile[s_h:e_h])) if e_h > s_h else 0.0
        features.append(chest_val - waist_val)   # chest-waist drop
        features.append(hips_val  - waist_val)   # hip-waist drop
        features.append(chest_val - hips_val)    # chest-hip difference

    # Стандартное отклонение ширин внутри зоны — "кривизна":
    # высокое std = резкое сужение (выраженная талия),
    # низкое std = равномерная ширина (прямоугольный силуэт)
    for profile in (front, side):
        for zone in zones:
            s, e = _zone_indices(zone)
            features.append(float(np.std(profile[s:e])) if e > s else 0.0)

    # Отношение глубины (профиль) к ширине (анфас) в каждой зоне —
    # характеристика формы поперечного сечения тела.
    # Если ratio ≈ 1 — сечение круглое; ratio << 1 — плоское;
    # ratio >> 1 — выступающий живот или ягодицы в профиль
    for zone in zones:
        s, e = _zone_indices(zone)
        front_mean = float(np.mean(front[s:e])) if e > s else 1.0
        side_mean  = float(np.mean(side[s:e]))  if e > s else 1.0
        ratio = side_mean / front_mean if front_mean > 1e-6 else 1.0
        features.append(ratio)

    # Средняя "рябь" силуэта — насколько гладко меняется ширина от строки к строке.
    # Высокое значение может означать артефакты маски или одежду с выраженной фактурой.
    diffs = np.abs(np.diff(front))
    features.append(float(np.mean(diffs)))

    return np.array(features, dtype=np.float32)


def extract_features(front_path: str, side_path: str, height_cm: float) -> np.ndarray | None:
    """
    Собирает полный вектор признаков для одного человека.
    Структура: [профиль_анфас(100) | профиль_профиль(100) | рост(1) | форма(28)]
    """
    profile_front = extract_width_profile(front_path)
    profile_side  = extract_width_profile(side_path)
    if profile_front is None or profile_side is None:
        return None

    shape_features = _compute_shape_features(profile_front, profile_side)

    return np.concatenate([
        profile_front,
        profile_side,
        [height_cm / 200.0],
        shape_features,
    ]).astype(np.float32)


def build_dataset():
    print("Загружаем таблицы...")
    photo_map = pd.read_csv(PHOTO_MAP)
    measures  = pd.read_csv(MEASURE_CSV)
    hwg       = pd.read_csv(HWG_CSV)

    chest_col = [c for c in measures.columns if "chest" in c.lower()][0]
    waist_col = [c for c in measures.columns if "waist" in c.lower()][0]
    hip_col   = [c for c in measures.columns if "hip"   in c.lower()][0]

    df = photo_map.merge(measures, on="subject_id", how="inner")
    df = df.merge(hwg[["subject_id", "height_cm", "gender"]], on="subject_id", how="inner")
    df = df[df["gender"] == "female"].reset_index(drop=True)
    print(f"Записей (female): {len(df)}")

    X_list, y_list, groups = [], [], []
    skipped = 0

    for _, row in df.iterrows():
        pid = row["photo_id"]
        feats = extract_features(
            os.path.join(MASK_DIR,   f"{pid}.png"),
            os.path.join(MASK_L_DIR, f"{pid}.png"),
            row["height_cm"],
        )
        if feats is None:
            skipped += 1
            continue

        targets = np.array([row[chest_col], row[waist_col], row[hip_col]], dtype=np.float32)
        if any(np.isnan(targets)) or any(targets < 40) or any(targets > 200):
            skipped += 1
            continue

        X_list.append(feats)
        y_list.append(targets)
        groups.append(row["subject_id"])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    print(f"Примеров: {len(X)}, пропущено: {skipped}")
    print(f"Размер вектора признаков: {X.shape[1]}  (было 201, стало {X.shape[1]})")
    return X, y, np.array(groups)


def train_and_evaluate(X, y, groups):
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X_scaled, y, groups))

    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}")
    print("Обучаем модель v2 (2–5 минут)...")

    model = MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=300,       # больше деревьев — лучше улавливает паттерны
            max_depth=4,            # оставляем — глубже начнётся переобучение
            learning_rate=0.04,     # чуть медленнее → стабильнее сходится
            subsample=0.8,
            min_samples_leaf=5,     # не делить узел если там меньше 5 примеров
            random_state=42,
        )
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    print("\n── Результаты v2 ──────────────────────────────────────────────")
    for i, name in enumerate(["chest", "waist", "hip"]):
        mae = mean_absolute_error(y_val[:, i], y_pred[:, i])
        print(f"  {name}: MAE = {mae:.2f} см")

    return model, scaler


if __name__ == "__main__":
    X, y, groups = build_dataset()
    model, scaler = train_and_evaluate(X, y, groups)

    # Сохраняем с суффиксом _v2, чтобы не затереть старую модель.
    # После проверки результатов можно переименовать в основной файл.
    joblib.dump(model,  os.path.join(MODEL_DIR, "measurement_regressor_v2.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_v2.joblib"))
    print("\nМодель v2 сохранена. Для использования в пайплайне переименуй файлы")
    print("или измени model_dir в config.py чтобы он указывал на нужную версию.")