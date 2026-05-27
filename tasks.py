from __future__ import annotations

import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

from celery_app import celery
from src.config import AppConfig, InputConfig, ModelConfig, SilhouetteConfig
from src.io_utils import load_image_from_bytes
from src.pipeline import run_pipeline_from_arrays
from src.pose import PoseEstimator

load_dotenv(".env")

YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "yolo26x-pose.pt")
API_INTERNAL_URL: str = os.getenv("API_INTERNAL_URL", "http://api:8000")
INTERNAL_API_KEY: str = os.getenv("INTERNAL_API_KEY", "")

_estimator: PoseEstimator | None = None
_config: AppConfig | None = None


def _get_estimator() -> tuple[PoseEstimator, AppConfig]:
    global _estimator, _config
    if _estimator is None:
        _estimator = PoseEstimator(YOLO_MODEL_PATH)
        _config = AppConfig(
            inputs=InputConfig(),
            model=ModelConfig(yolo_model_path=YOLO_MODEL_PATH),
            silhouette=SilhouetteConfig(),
        )
    return _estimator, _config


@celery.task(
    name="analyze_photo",
    bind=True,
    max_retries=3,
    default_retry_delay=10,
)
def analyze_photo(
    self,
    front_path: str,
    side_path: str,
    height_cm: float,
    user_id: str,
) -> dict:
    front_file = Path(front_path)
    side_file = Path(side_path)

    try:
        if not front_file.exists() or not side_file.exists():
            raise FileNotFoundError(
                f"Фото не найдено: {front_path} / {side_path}"
            )

        front_bytes = front_file.read_bytes()
        side_bytes = side_file.read_bytes()

        estimator, base_config = _get_estimator()

        front_img = load_image_from_bytes(
            front_bytes, base_config.silhouette.max_image_side
        )
        side_img = load_image_from_bytes(
            side_bytes, base_config.silhouette.max_image_side
        )

        cfg = AppConfig(
            inputs=InputConfig(user_height_cm=height_cm),
            model=base_config.model,
            silhouette=base_config.silhouette,
        )

        result = run_pipeline_from_arrays(
            front_image=front_img,
            side_image=side_img,
            config=cfg,
            estimator=estimator,
        )

        measurements = result.get("measurements", {})

        _save_to_db(
            user_id=user_id,
            chest_cm=measurements.get("chest", {}).get("circumference_cm"),
            waist_cm=measurements.get("waist", {}).get("circumference_cm"),
            hips_cm=measurements.get("hips", {}).get("circumference_cm"),
            height_cm=height_cm,
            body_type=result.get("body_type"),
        )

        # Удаляем файлы ТОЛЬКО после успешного завершения всего
        front_file.unlink(missing_ok=True)
        side_file.unlink(missing_ok=True)

        return {
            "measurements": measurements,
            "body_type": result.get("body_type"),
            "scale_cm_per_px": result.get("scale_cm_per_px"),
        }

    except FileNotFoundError:
        # Файлов нет — retry бессмысленен
        raise

    except RuntimeError:
        # YOLO не нашёл человека — retry бессмысленен
        front_file.unlink(missing_ok=True)
        side_file.unlink(missing_ok=True)
        raise

    except Exception as exc:
        # Временная ошибка (сеть, память) — retry имеет смысл
        # Файлы НЕ удаляем, чтобы retry мог их использовать
        raise self.retry(exc=exc)


def _save_to_db(
    user_id: str,
    chest_cm: float | None,
    waist_cm: float | None,
    hips_cm: float | None,
    height_cm: float,
    body_type: str | None,
) -> None:
    with httpx.Client(timeout=10) as client:
        client.post(
            f"{API_INTERNAL_URL}/internal/measurements",
            json={
                "user_id": user_id,
                "chest_cm": chest_cm,
                "waist_cm": waist_cm,
                "hips_cm": hips_cm,
                "height_cm": height_cm,
                "body_type": body_type,
                "source": "auto",
            },
            headers={"x-internal-key": INTERNAL_API_KEY},
        )
