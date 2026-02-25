from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InputConfig:
    front_image: Path = Path("images/mom.jpg")
    side_image: Path = Path("images/side.jpg")
    user_height_cm: float = 171.0


@dataclass(frozen=True)
class ModelConfig:
    yolo_model_path: str = "yolo11x-pose.pt"


@dataclass(frozen=True)
class SilhouetteConfig:
    max_image_side: int = 1920
    scan_step_px: int = 2
    smooth_window: int = 25
    mask_window_px: int = 5
    hip_search_extra: float = 0.40
    torso_x_margin: float = 0.08


@dataclass(frozen=True)
class OutputConfig:
    result_dir: Path = Path("result")
    front_debug_name: str = "front_measurements.jpg"
    side_debug_name: str = "side_measurements.jpg"
    front_mask_name: str = "front_mask.jpg"
    side_mask_name: str = "side_mask.jpg"


@dataclass(frozen=True)
class AppConfig:
    inputs: InputConfig = InputConfig()
    model: ModelConfig = ModelConfig()
    silhouette: SilhouetteConfig = SilhouetteConfig()
    output: OutputConfig = OutputConfig()


DEFAULT_CONFIG = AppConfig()