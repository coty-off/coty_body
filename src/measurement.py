from __future__ import annotations

# Количество точек профиля — должно совпадать со значением при обучении!
N_POINTS = 100

# FFIT-пороги (Lee, Istook, Nam & Park, 2007)
_T1 = 2.54
_T2 = 9.14
_T3 = 22.86
_T4 = 25.40


def classify_body_type(chest_cm: float, waist_cm: float, hip_cm: float) -> str:
    """
    Определяет тип фигуры по FFIT-формулам (Lee et al., 2007).
    Порядок условий важен — первое подходящее условие и есть результат.
    """
    bh = chest_cm - hip_cm
    hb = hip_cm   - chest_cm
    bw = chest_cm - waist_cm
    hw = hip_cm   - waist_cm

    if bh <= _T1 and hb < _T2 and (bw >= _T3 or hw >= _T4):
        return "Hourglass"
    if _T1 < bh < _T4 and bw >= _T3:
        return "Top Hourglass"
    if _T2 <= hb < _T4 and hw >= _T3:
        return "Bottom Hourglass"
    if hb >= _T2 and hw < _T3:
        return "Triangle"
    if bh >= _T2 and bw < _T3:
        return "Inverted Triangle"
    if hb < _T2 and bh < _T2 and bw < _T3 and hw < _T4:
        return "Rectangle"
    return "Undefined"
