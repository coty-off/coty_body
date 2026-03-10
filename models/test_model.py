"""
Трёхуровневая проверка качества обученной модели.

Уровень 1 — Численные метрики (MAE, RMSE, R²) на отложенной выборке.
Уровень 2 — Визуальный анализ ошибок: где модель ошибается и почему.
Уровень 3 — Сквозная проверка на конкретном человеке из датасета.

Запускай из папки проекта:
    python validate_model.py
"""

import os
import numpy as np
import pandas as pd
import cv2
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append(r"C:\Users\kroko\PycharmProjects\COTY_body\models")
from train_model_v2 import extract_features, _compute_shape_features

# ── Пути — те же, что в train_measurement_regressor.py ──────────────────────
BASE_DIR = r"C:\Users\kroko\PycharmProjects\COTY_body\datasets"
MASK_DIR = os.path.join(BASE_DIR, "mask")
MASK_L_DIR = os.path.join(BASE_DIR, "mask_left")
PHOTO_MAP = os.path.join(BASE_DIR, "subject_to_photo_map.csv")
MEASURE_CSV = os.path.join(BASE_DIR, "measurements.csv")
HWG_CSV = os.path.join(BASE_DIR, "hwg_metadata.csv")
MODEL_PATH = r"C:\Users\kroko\PycharmProjects\COTY_body\models\measurement_regressor_v2.joblib"
SCALER_PATH = r"C:\Users\kroko\PycharmProjects\COTY_body\models\scaler_v2.joblib"

N_POINTS = 100  # должно совпадать со значением при обучении!


# ── FFIT-формулы (Lee et al., 2007) ─────────────────────────────────────────
def classify_body_type(chest, waist, hip):
    IN = 2.54
    T1, T2, T3, T4 = 1 * IN, 3.6 * IN, 9 * IN, 10 * IN
    bh, hb = chest - hip, hip - chest
    bw, hw = chest - waist, hip - waist

    if abs(bh) <= T1 and (bw >= T3 or hw >= T3): return "Hourglass"
    if T1 < bh < T4 and bw >= T3:                return "Top Hourglass"
    if T2 <= hb < T4 and hw >= T3:               return "Bottom Hourglass"
    if hb >= T2 and 0 <= hw < T3:                return "Triangle"
    if bh >= T2 and bw < T3 and hw >= 0:         return "Inverted Triangle"
    if hb < T2 and bh < T2 and 0 <= bw < T3 and 0 <= hw < T4: return "Rectangle"
    return "Undefined"


def extract_width_profile(mask_path):
    """Та же функция, что при обучении — должна быть идентичной."""
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


def rebuild_validation_set():
    """
    Воссоздаём ровно ту же валидационную выборку, что использовалась при обучении.
    Ключевой момент: random_state=42 и GroupShuffleSplit гарантируют, что
    при одинаковых параметрах мы всегда получим одни и те же субъекты в val.
    Это значит, что мы честно проверяем на людях, которых модель не видела.
    """
    print("Восстанавливаем валидационную выборку...")
    photo_map = pd.read_csv(PHOTO_MAP)
    measures = pd.read_csv(MEASURE_CSV)
    hwg = pd.read_csv(HWG_CSV)

    # Определяем столбцы с обхватами автоматически
    chest_col = [c for c in measures.columns if "chest" in c.lower()][0]
    waist_col = [c for c in measures.columns if "waist" in c.lower()][0]
    hip_col = [c for c in measures.columns if "hip" in c.lower()][0]

    df = photo_map.merge(measures, on="subject_id", how="inner")
    df = df.merge(hwg[["subject_id", "height_cm", "gender"]], on="subject_id", how="inner")
    df = df[df["gender"] == "female"].reset_index(drop=True)

    X_list, y_list, groups, photo_ids = [], [], [], []
    for _, row in df.iterrows():
        pid = row["photo_id"]
        pf = extract_width_profile(os.path.join(MASK_DIR, f"{pid}.png"))
        ps = extract_width_profile(os.path.join(MASK_L_DIR, f"{pid}.png"))
        if pf is None or ps is None:
            continue
        targets = np.array([row[chest_col], row[waist_col], row[hip_col]], dtype=np.float32)
        if any(np.isnan(targets)) or any(targets < 40) or any(targets > 200):
            continue
        features = extract_features(
            os.path.join(MASK_DIR, f"{pid}.png"),
            os.path.join(MASK_L_DIR, f"{pid}.png"),
            row["height_cm"],
        )
        if features is None:
            continue
        X_list.append(features);
        y_list.append(targets)
        groups.append(row["subject_id"]);
        photo_ids.append(pid)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    groups = np.array(groups)

    # Воспроизводим то же разбиение, что при обучении
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, val_idx = next(splitter.split(X, y, groups))

    return (X[val_idx], y[val_idx],
            [photo_ids[i] for i in val_idx],
            [groups[i] for i in val_idx],
            chest_col, waist_col, hip_col)


# ════════════════════════════════════════════════════════════════════════════
# УРОВЕНЬ 1 — Численные метрики
# ════════════════════════════════════════════════════════════════════════════
def level1_metrics(y_true, y_pred):
    """
    MAE (Mean Absolute Error) — средняя абсолютная ошибка в сантиметрах.
        Самая интуитивная метрика: "в среднем модель ошибается на X см".

    RMSE (Root Mean Squared Error) — то же самое, но большие ошибки
        штрафуются сильнее. Если RMSE >> MAE, значит есть выбросы —
        редкие случаи с очень большой ошибкой.

    R² (коэффициент детерминации) — насколько хорошо модель объясняет
        вариацию данных. R²=1.0 — идеально, R²=0.0 — модель не лучше
        предсказания средним значением.
    """
    target_names = ["chest_cm", "waist_cm", "hip_cm"]
    print("\n" + "═" * 65)
    print("УРОВЕНЬ 1 — Численные метрики на валидационной выборке")
    print("═" * 65)
    print(f"{'Метрика':<14} {'chest':>10} {'waist':>10} {'hip':>10}")
    print("-" * 45)

    for metric_name, metric_fn, fmt in [
        ("MAE (см)", mean_absolute_error, ".2f"),
        ("RMSE (см)", lambda t, p: np.sqrt(mean_squared_error(t, p)), ".2f"),
        ("R²", r2_score, ".3f"),
    ]:
        vals = [metric_fn(y_true[:, i], y_pred[:, i]) for i in range(3)]
        row = f"{metric_name:<14}" + "".join(f"{v:>10.{fmt[1:]}}" for v in vals)
        print(row)

    print("\nОриентиры для оценки MAE:")
    print("  < 3 см  — отличный результат (лучше ручного замера у неспециалиста)")
    print("  3–5 см  — хороший результат (сопоставим с методами в литературе)")
    print("  5–8 см  — приемлемый результат для прототипа")
    print("  > 8 см  — модель работает плохо, нужна доработка")


# ════════════════════════════════════════════════════════════════════════════
# УРОВЕНЬ 2 — Визуальный анализ ошибок
# ════════════════════════════════════════════════════════════════════════════
def level2_error_analysis(y_true, y_pred):
    """
    Строит три типа графиков для каждого обхвата.

    График "предсказанное vs реальное": точки должны лежать вдоль диагонали
    y=x. Если они систематически выше/ниже — модель переоценивает/недооценивает.

    График "ошибка vs реальное значение": показывает, где модель ошибается
    больше — на маленьких или больших значениях. Хорошая модель даёт
    горизонтальное облако точек без выраженного тренда.

    Гистограмма ошибок: идеально должна быть симметричной и узкой.
    Смещённая влево/вправо означает систематическую ошибку (bias).
    """
    target_names = ["Грудь (chest)", "Талия (waist)", "Бёдра (hip)"]
    errors = y_pred - y_true  # положительное значение = переоценка

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Уровень 2 — Анализ ошибок модели", fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(3, 3, hspace=0.45, wspace=0.35)

    for i, name in enumerate(target_names):
        # ── Предсказанное vs реальное ─────────────────────────────────────
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.scatter(y_true[:, i], y_pred[:, i], alpha=0.4, s=12, color="#2196F3")
        lims = [y_true[:, i].min() - 5, y_true[:, i].max() + 5]
        ax1.plot(lims, lims, 'r--', linewidth=1.5, label='идеал (y=x)')
        ax1.set_xlabel("Реальное (см)");
        ax1.set_ylabel("Предсказанное (см)")
        ax1.set_title(f"{name}\nПредсказанное vs Реальное")
        ax1.legend(fontsize=8)

        # ── Ошибка vs реальное значение ──────────────────────────────────
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.scatter(y_true[:, i], errors[:, i], alpha=0.4, s=12, color="#FF9800")
        ax2.axhline(0, color='red', linewidth=1.5, linestyle='--')
        ax2.set_xlabel("Реальное (см)");
        ax2.set_ylabel("Ошибка (см)")
        ax2.set_title(f"{name}\nОшибка vs Реальное")
        # Скользящее среднее ошибки — показывает системный тренд
        sorted_idx = np.argsort(y_true[:, i])
        window = max(1, len(sorted_idx) // 20)
        smoothed = np.convolve(errors[sorted_idx, i],
                               np.ones(window) / window, mode='valid')
        x_smooth = y_true[sorted_idx, i][window // 2: window // 2 + len(smoothed)]
        ax2.plot(x_smooth, smoothed, 'b-', linewidth=2, label='тренд ошибки')
        ax2.legend(fontsize=8)

        # ── Гистограмма ошибок ────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[i, 2])
        ax3.hist(errors[:, i], bins=40, color="#4CAF50", edgecolor='white', alpha=0.8)
        ax3.axvline(0, color='red', linewidth=1.5, linestyle='--')
        ax3.axvline(errors[:, i].mean(), color='blue', linewidth=1.5,
                    linestyle='-', label=f'среднее={errors[:, i].mean():.1f}')
        ax3.set_xlabel("Ошибка (см)");
        ax3.set_ylabel("Частота")
        ax3.set_title(f"{name}\nРаспределение ошибок")
        ax3.legend(fontsize=8)

    plt.savefig("validation_error_analysis.png", dpi=150, bbox_inches='tight')
    print("\nГрафики сохранены: validation_error_analysis.png")
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# УРОВЕНЬ 3 — Сквозная проверка типа фигуры
# ════════════════════════════════════════════════════════════════════════════
def level3_body_type_accuracy(y_true, y_pred):
    """
    Проверяет: если FFIT-формулы применить к реальным обхватам и к
    предсказанным — как часто тип фигуры совпадает?

    Это самая практически важная метрика для твоего приложения.
    Модель может ошибаться на 3 см в обхвате, но при этом давать
    правильный тип фигуры — потому что FFIT-пороги имеют "буферные зоны".
    Или наоборот: маленькая ошибка в обхвате может переключить тип,
    если человек находится прямо на границе двух категорий.
    """
    types_true = [classify_body_type(*y_true[i]) for i in range(len(y_true))]
    types_pred = [classify_body_type(*y_pred[i]) for i in range(len(y_pred))]

    correct = sum(t == p for t, p in zip(types_true, types_pred))
    accuracy = correct / len(types_true) * 100

    print("\n" + "═" * 65)
    print("УРОВЕНЬ 3 — Точность классификации типа фигуры")
    print("═" * 65)
    print(f"Совпадений типа: {correct}/{len(types_true)}  ({accuracy:.1f}%)")

    # Показываем распределение типов в реальных данных
    from collections import Counter
    true_dist = Counter(types_true)
    pred_dist = Counter(types_pred)
    all_types = sorted(set(types_true) | set(types_pred))

    print(f"\n{'Тип фигуры':<22} {'Реальных':>10} {'Предсказанных':>15} {'Совпадений':>12}")
    print("-" * 62)
    for t in all_types:
        n_true = true_dist.get(t, 0)
        n_pred = pred_dist.get(t, 0)
        # Считаем совпадения именно для этого типа
        matches = sum(1 for tr, pr in zip(types_true, types_pred) if tr == t and pr == t)
        pct = matches / n_true * 100 if n_true > 0 else 0
        print(f"{t:<22} {n_true:>10} {n_pred:>15} {matches:>10} ({pct:.0f}%)")

    print(f"\nОриентиры: > 80% — отлично, 65–80% — хорошо, < 65% — нужна доработка")
    return accuracy


# ════════════════════════════════════════════════════════════════════════════
# БОНУС — Сквозной тест на одном конкретном человеке из датасета
# ════════════════════════════════════════════════════════════════════════════
def spot_check_single_person(model, scaler, photo_ids, y_true, y_pred, n=5):
    """
    Показывает детальный результат для N случайных людей из валидации.
    Это "проверка на здравый смысл" — смотришь глазами и чувствуешь,
    насколько предсказания реалистичны.
    """
    print("\n" + "═" * 65)
    print(f"БОНУС — Детальная проверка {n} случайных людей")
    print("═" * 65)

    indices = np.random.choice(len(photo_ids), size=min(n, len(photo_ids)), replace=False)
    for idx in indices:
        pid = photo_ids[idx]
        real = y_true[idx]
        pred = y_pred[idx]
        type_real = classify_body_type(*real)
        type_pred = classify_body_type(*pred)
        match = "✓" if type_real == type_pred else "✗"

        print(f"\nФото: {pid[:20]}...")
        print(f"  Реально:      грудь={real[0]:.1f}, талия={real[1]:.1f}, бёдра={real[2]:.1f}  → {type_real}")
        print(f"  Предсказано:  грудь={pred[0]:.1f}, талия={pred[1]:.1f}, бёдра={pred[2]:.1f}  → {type_pred}")
        print(
            f"  Ошибки:       Δгрудь={pred[0] - real[0]:+.1f}, Δталия={pred[1] - real[1]:+.1f}, Δбёдра={pred[2] - real[2]:+.1f}   {match}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Загружаем модель
    print("Загружаем модель и нормализатор...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Воссоздаём валидационную выборку
    X_val, y_true, photo_ids, subject_ids, *_ = rebuild_validation_set()
    X_val_scaled = scaler.transform(X_val)

    # Предсказываем
    print(f"Предсказываем для {len(X_val)} примеров...")
    y_pred = model.predict(X_val_scaled)

    # Три уровня проверки
    level1_metrics(y_true, y_pred)
    level3_body_type_accuracy(y_true, y_pred)
    spot_check_single_person(model, scaler, photo_ids, y_true, y_pred, n=5)

    # Графики строим в конце — они требуют matplotlib и открывают окно
    print("\nСтроим графики анализа ошибок...")
    level2_error_analysis(y_true, y_pred)