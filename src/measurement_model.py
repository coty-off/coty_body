"""
src/measurement_model.py

Модуль-обёртка над обученной регрессионной моделью.
Принимает бинарные маски анфас и профиль + рост → возвращает обхваты в см и тип фигуры.

Этот модуль намеренно изолирован от остального пайплайна:
все знания о модели (путь к файлу, формат входа, N_POINTS) сосредоточены здесь,
и больше нигде в проекте нет зависимости от joblib или sklearn.
"""

from __future__ import annotations

import numpy as np
import joblib
from pathlib import Path

# Количество точек профиля — должно совпадать со значением при обучении!
N_POINTS = 100

# FFIT-пороги (Lee, Istook, Nam & Park, 2007), конвертированы в сантиметры
_IN = 2.54
_T1 = 1   * _IN   # 2.54  см
_T2 = 3.6 * _IN   # 9.14  см
_T3 = 9   * _IN   # 22.86 см
_T4 = 10  * _IN   # 25.40 см


def classify_body_type(chest_cm: float, waist_cm: float, hip_cm: float) -> str:
    """
    Определяет тип фигуры по FFIT-формулам (Lee et al., 2007).
    Порядок условий важен — первое подходящее условие и есть результат.
    """
    bh = chest_cm - hip_cm
    hb = hip_cm   - chest_cm
    bw = chest_cm - waist_cm
    hw = hip_cm   - waist_cm

    if abs(bh) <= _T1 and (bw >= _T3 or hw >= _T3):   return "Hourglass"
    if _T1 < bh < _T4 and bw >= _T3:                  return "Top Hourglass"
    if _T2 <= hb < _T4 and hw >= _T3:                 return "Bottom Hourglass"
    if hb >= _T2 and 0 <= hw < _T3:                   return "Triangle"
    if bh >= _T2 and bw < _T3 and hw >= 0:            return "Inverted Triangle"
    if hb < _T2 and bh < _T2 and 0 <= bw < _T3 and 0 <= hw < _T4:
        return "Rectangle"
    return "Undefined"


def _extract_width_profile(binary_mask: np.ndarray) -> np.ndarray | None:
    """
    Извлекает нормализованный профиль ширин из бинарной маски.

    Для каждого из N_POINTS уровней по вертикали (равномерно от верхней
    до нижней границы силуэта) измеряем ширину белой области и делим
    на высоту силуэта в пикселях. Нормализация делает профиль инвариантным
    к росту и расстоянию до камеры.

    ВАЖНО: эта функция идентична той, что использовалась при обучении.
    Любое изменение здесь сломает совместимость с сохранённой моделью.
    """
    rows = np.any(binary_mask > 127, axis=1)
    if not rows.any():
        return None

    y_top, y_bottom = np.where(rows)[0][[0, -1]]
    silhouette_height = y_bottom - y_top
    if silhouette_height < 10:
        return None

    levels = np.linspace(y_top, y_bottom, N_POINTS, dtype=int)
    widths = []
    for y in levels:
        row = binary_mask[y, :]
        cols = np.where(row > 127)[0]
        widths.append((cols[-1] - cols[0]) / silhouette_height if len(cols) >= 2 else 0.0)

    return np.array(widths, dtype=np.float32)


class MeasurementPredictor:
    """
    Обёртка над обученной моделью для предсказания обхватов тела.

    Пример использования:
        predictor = MeasurementPredictor.load("models/")
        result = predictor.predict(front_mask, side_mask, height_cm=171.0)
        # result = {"chest_cm": 92.3, "waist_cm": 74.1, "hip_cm": 98.5,
        #           "body_type": "Hourglass"}
    """

    def __init__(self, model, scaler):
        self._model  = model
        self._scaler = scaler

    @classmethod
    def load(cls, model_dir: str | Path) -> "MeasurementPredictor":
        """
        Загружает модель и нормализатор из указанной папки.
        Ожидает файлы: measurement_regressor.joblib и scaler.joblib.
        """
        model_dir = Path(model_dir)
        model  = joblib.load(model_dir / "measurement_regressor_v2.joblib")
        scaler = joblib.load(model_dir / "scaler_v2.joblib")
        return cls(model, scaler)

    def predict(
        self,
        front_mask: np.ndarray,
        side_mask:  np.ndarray,
        height_cm:  float,
    ) -> dict[str, float | str]:
        """
        Предсказывает обхваты и тип фигуры из бинарных масок и роста.

        Параметры:
            front_mask — бинарная маска анфас (uint8, белый силуэт)
            side_mask  — бинарная маска профиль (аналогично)
            height_cm  — рост человека в сантиметрах

        Возвращает словарь: chest_cm, waist_cm, hip_cm, body_type.
        """
        profile_front = _extract_width_profile(front_mask)
        profile_side  = _extract_width_profile(side_mask)

        if profile_front is None or profile_side is None:
            raise ValueError(
                "Не удалось извлечь профиль — силуэт пустой или слишком маленький."
            )

        # Собираем вектор признаков точно так же, как при обучении
        features = np.concatenate([profile_front, profile_side, [height_cm / 200.0]])
        features_scaled = self._scaler.transform(features.reshape(1, -1))

        chest, waist, hip = self._model.predict(features_scaled)[0]

        return {
            "chest_cm":  round(float(chest), 1),
            "waist_cm":  round(float(waist), 1),
            "hip_cm":    round(float(hip),   1),
            "body_type": classify_body_type(chest, waist, hip),
        }