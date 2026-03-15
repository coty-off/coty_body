from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InputConfig:
    front_image: Path = Path("images/mom.jpg")
    side_image: Path = Path("images/side.jpg")
    user_height_cm: float = 171.0


@dataclass(frozen=True)
class ModelConfig:
    yolo_model_path: str = "yolo26x-pose.pt"


@dataclass(frozen=True)
class SilhouetteConfig:
    max_image_side: int = 1920
    scan_step_px: int = 2
    smooth_window: int = 25
    mask_window_px: int = 5
    hip_search_extra: float = 0.40
    torso_x_margin: float = 0.08

@dataclass(frozen=True)
class AppConfig:
    inputs: InputConfig = InputConfig()
    model: ModelConfig = ModelConfig()
    silhouette: SilhouetteConfig = SilhouetteConfig()


DEFAULT_CONFIG = AppConfig()
