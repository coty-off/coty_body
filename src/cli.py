from __future__ import annotations

import argparse
from pathlib import Path

from .config import AppConfig, InputConfig, ModelConfig, OutputConfig, SilhouetteConfig
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Расчет мерок по фото (front+side) на базе YOLO11x-pose.")
    parser.add_argument("--front", type=Path, required=True, help="Путь к фото в анфас")
    parser.add_argument("--side", type=Path, required=True, help="Путь к фото в профиль")
    parser.add_argument("--height", type=float, required=True, help="Рост человека, см")
    parser.add_argument("--model", default="yolo11x-pose.pt", help="Путь к весам YOLO pose")
    parser.add_argument("--result-dir", type=Path, default=Path("result"), help="Папка для результатов")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = AppConfig(
        inputs=InputConfig(front_image=args.front, side_image=args.side, user_height_cm=args.height),
        model=ModelConfig(yolo_model_path=args.model),
        silhouette=SilhouetteConfig(),
        output=OutputConfig(result_dir=args.result_dir),
    )

    result = run_pipeline(config)

    print(f"Масштаб: 1 px = {result['scale_cm_per_px']:.4f} см")
    print("\nМерки (см):")
    for name, payload in result["measurements"].items():
        print(
            f"- {name}: width(front)={payload['front_width_cm']:.1f}, "
            f"depth(side)={payload['side_depth_cm']:.1f}, "
            f"circ≈{payload['circumference_cm']:.1f}"
        )


if __name__ == "__main__":
    main()