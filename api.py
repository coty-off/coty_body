from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from src.config import AppConfig, InputConfig, ModelConfig, OutputConfig, SilhouetteConfig
from src.io_utils import load_image_from_bytes
from src.pipeline import run_pipeline_from_arrays
from src.pose import PoseEstimator

app = FastAPI(title="Body Measurements API", version="1.0.0")


class AppState:
    def __init__(self) -> None:
        self.estimator: PoseEstimator | None = None
        self.base_config: AppConfig | None = None


state = AppState()


def _get_valid_api_keys() -> set[str]:
    raw = os.getenv("API_KEYS", "")
    return {item.strip() for item in raw.split(",") if item.strip()}


def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
    valid_keys = _get_valid_api_keys()
    if not valid_keys:
        raise HTTPException(status_code=500, detail="API_KEYS не настроены на сервере")
    if x_api_key is None or x_api_key not in valid_keys:
        raise HTTPException(status_code=401, detail="Неверный или отсутствующий API-ключ")
    return x_api_key


@app.on_event("startup")
def on_startup() -> None:
    load_dotenv(".env.apikey")
    model_path = "yolo26x-pose.pt"
    result_dir = Path("result")

    state.estimator = PoseEstimator(model_path)
    state.base_config = AppConfig(
        inputs=InputConfig(),
        model=ModelConfig(yolo_model_path=model_path),
        silhouette=SilhouetteConfig(),
        output=OutputConfig(result_dir=result_dir),
    )


@app.post("/measure")
async def measure(
    front_image: UploadFile = File(..., description="Фото анфас"),
    side_image: UploadFile = File(..., description="Фото профиль"),
    height_cm: float = Form(..., description="Рост человека в сантиметрах"),
    api_key: str = Depends(require_api_key),
) -> Dict[str, Any]:
    if state.estimator is None or state.base_config is None:
        raise HTTPException(status_code=500, detail="Сервис ещё не инициализирован")

    try:
        front_bytes = await front_image.read()
        side_bytes = await side_image.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Не удалось прочитать загруженные файлы")

    if not front_bytes or not side_bytes:
        raise HTTPException(status_code=400, detail="Пустой файл изображения")

    try:
        front_img = load_image_from_bytes(front_bytes, state.base_config.silhouette.max_image_side)
        side_img = load_image_from_bytes(side_bytes, state.base_config.silhouette.max_image_side)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    cfg = AppConfig(
        inputs=InputConfig(
            front_image=state.base_config.inputs.front_image,
            side_image=state.base_config.inputs.side_image,
            user_height_cm=height_cm,
        ),
        model=state.base_config.model,
        silhouette=state.base_config.silhouette,
        output=state.base_config.output,
    )

    try:
        result = run_pipeline_from_arrays(
            front_image=front_img,
            side_image=side_img,
            config=cfg,
            estimator=state.estimator,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка: {e}")

    return {
        "measurements": result.get("measurements", {}),
        "body_type": result.get("body_type"),
        "scale_cm_per_px": result.get("scale_cm_per_px"),
        "front_height_px": result.get("front_height_px"),
        "side_height_px": result.get("side_height_px"),
    }


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})

