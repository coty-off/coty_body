import pandas as pd

def body_type(chest_cm, waist_cm, hip_cm):
    _T1 = 2.54
    _T2 = 9.14
    _T3 = 22.86
    _T4 = 25.40

    bh = chest_cm - hip_cm
    hb = hip_cm - chest_cm
    bw = chest_cm - waist_cm
    hw = hip_cm - waist_cm

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


def create_dataset():
    df = pd.read_csv("datasets/new_dataset.csv")

    df["type_body"] = df.apply(
        lambda row: body_type(
            row["chest"],
            row["waist"],
            row["hip"]
        ),
        axis=1
    )

    df.to_csv("datasets/dataset_with_body_type.csv", index=False)


if __name__ == "__main__":
    create_dataset()