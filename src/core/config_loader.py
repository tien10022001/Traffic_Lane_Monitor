from pathlib import Path
from typing import Dict, List, Optional
import threading
import yaml
from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = ["*"]


class PathsConfig(BaseModel):
    db_path: str = "data/traffic.db"
    evidence_dir: str = "data/evidence"
    logs_dir: str = "logs"
    model_path: str = "models/best.onnx"
    plate_model_path: str = "models/plate_detector.onnx"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    max_bytes: int = 52428800  # 50 MB
    backup_count: int = 5


class AIConfig(BaseModel):
    device: str = "cpu"  # auto, gpu, cpu
    providers: List[str] = ["CPUExecutionProvider"]
    imgsz: int = 640
    default_conf: float = 0.43
    iou_threshold: float = 0.45
    per_class_conf: Dict[int, float] = Field(default_factory=lambda: {
        0: 0.45, 1: 0.40, 2: 0.40, 3: 0.45, 4: 0.42, 5: 0.42, 6: 0.50, 7: 0.50
    })
    classes: Dict[int, str] = Field(default_factory=lambda: {
        0: "car", 1: "motorcycle", 2: "bicycle", 3: "bus",
        4: "truck", 5: "tricycle", 6: "ambulance", 7: "fire_truck"
    })
    exempt_classes: List[int] = Field(default_factory=lambda: [6, 7])


class TrackingConfig(BaseModel):
    track_buffer: int = 75
    class_smoothing_window: int = 8


class RulesConfig(BaseModel):
    stationary_speed_px_per_sec: float = 15.0
    grace_period_sec: float = 1.5
    occlusion_iou_threshold: float = 0.65
    occlusion_max_gap_sec: float = 5.0
    wrong_lane_min_frames: int = 6
    violation_cooldown_sec: float = 60.0


class DeduplicationConfig(BaseModel):
    enabled: bool = True
    plate_cooldown_sec: float = 600.0
    false_positive_cooldown_sec: float = 1800.0
    max_plate_edit_dist: int = 1
    spatial_iou_threshold: float = 0.65
    exit_grace_sec: float = 45.0
    update_existing_duration: bool = True


class PerspectiveAndRenderingConfig(BaseModel):
    roi_display_mode: str = "full"  # 'full', 'border_only', 'hidden', 'striped'
    roi_alpha: float = 0.20
    use_static_mask_cache: bool = True
    dual_contrast_border: bool = True
    ground_contact_mode: str = "bottom_center"  # 'bottom_center', 'bottom_strip_20'
    draw_ground_contact_marker: bool = True
    enable_homography: bool = False
    enable_homography_calibration: bool = False
    homography_landmarks: List[List[float]] = Field(default_factory=list)


class OCRConfig(BaseModel):
    workers: int = 1
    queue_maxsize: int = 100
    min_ocr_conf: float = 0.55
    vn_plate_regex: str = r"^[0-9]{2}[A-Z]{1,2}[0-9]?-?[0-9]{4,5}$"


class MonitoringConfig(BaseModel):
    interval_sec: float = 2.0
    history_size: int = 180
    warn_cpu_percent: float = 85.0
    warn_ram_percent: float = 85.0
    warn_gpu_percent: float = 90.0
    warn_disk_free_gb: float = 5.0


class RetentionConfig(BaseModel):
    enabled: bool = True
    run_at_hour: int = 2
    retention_days: int = 30


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    rules: RulesConfig = Field(default_factory=RulesConfig)
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    perspective_and_rendering: PerspectiveAndRenderingConfig = Field(default_factory=PerspectiveAndRenderingConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)


_CONFIG_LOCK = threading.RLock()
_CURRENT_CONFIG: Optional[AppConfig] = None
_DEFAULT_CONFIG_PATH = "config/config.yaml"


def load_config(config_path: str = _DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load and validate configuration from yaml file."""
    path = Path(config_path)
    if not path.is_file():
        # Fallback to default
        cfg = AppConfig()
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
        cfg = AppConfig(**raw_data)

    global _CURRENT_CONFIG
    with _CONFIG_LOCK:
        _CURRENT_CONFIG = cfg
    return cfg


def get_config(config_path: str = _DEFAULT_CONFIG_PATH) -> AppConfig:
    """Thread-safe access to cached config, loading if not initialized."""
    global _CURRENT_CONFIG
    with _CONFIG_LOCK:
        if _CURRENT_CONFIG is None:
            _CURRENT_CONFIG = load_config(config_path)
        return _CURRENT_CONFIG


def reload_config(config_path: str = _DEFAULT_CONFIG_PATH) -> AppConfig:
    """Hot-reload configuration from file."""
    return load_config(config_path)


def save_ai_thresholds(
    default_conf: float,
    iou_threshold: float,
    per_class_conf: Dict[int, float],
    config_path: str = _DEFAULT_CONFIG_PATH,
) -> None:
    """Persist updated AI thresholds back to config.yaml and update cached config."""
    path = Path(config_path)
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        if "ai" not in raw:
            raw["ai"] = {}
        raw["ai"]["default_conf"] = round(float(default_conf), 3)
        raw["ai"]["iou_threshold"] = round(float(iou_threshold), 3)
        raw["ai"]["per_class_conf"] = {
            int(k): round(float(v), 3) for k, v in per_class_conf.items()
        }

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=True)

    global _CURRENT_CONFIG
    with _CONFIG_LOCK:
        if _CURRENT_CONFIG is not None:
            _CURRENT_CONFIG.ai.default_conf = round(float(default_conf), 3)
            _CURRENT_CONFIG.ai.iou_threshold = round(float(iou_threshold), 3)
            for cid, val in per_class_conf.items():
                _CURRENT_CONFIG.ai.per_class_conf[int(cid)] = round(float(val), 3)


def save_deduplication_config(
    enabled: bool,
    plate_cooldown_sec: float,
    false_positive_cooldown_sec: float,
    max_plate_edit_dist: int,
    spatial_iou_threshold: float,
    exit_grace_sec: float,
    update_existing_duration: bool = True,
    config_path: str = _DEFAULT_CONFIG_PATH,
) -> None:
    """Persist updated deduplication parameters to config.yaml and update cached config."""
    path = Path(config_path)
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        raw["deduplication"] = {
            "enabled": enabled,
            "plate_cooldown_sec": round(float(plate_cooldown_sec), 1),
            "false_positive_cooldown_sec": round(float(false_positive_cooldown_sec), 1),
            "max_plate_edit_dist": int(max_plate_edit_dist),
            "spatial_iou_threshold": round(float(spatial_iou_threshold), 2),
            "exit_grace_sec": round(float(exit_grace_sec), 1),
            "update_existing_duration": bool(update_existing_duration),
        }

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=True)

    global _CURRENT_CONFIG
    with _CONFIG_LOCK:
        if _CURRENT_CONFIG is not None:
            _CURRENT_CONFIG.deduplication = DeduplicationConfig(
                enabled=enabled,
                plate_cooldown_sec=plate_cooldown_sec,
                false_positive_cooldown_sec=false_positive_cooldown_sec,
                max_plate_edit_dist=max_plate_edit_dist,
                spatial_iou_threshold=spatial_iou_threshold,
                exit_grace_sec=exit_grace_sec,
                update_existing_duration=update_existing_duration,
            )


def save_perspective_config(
    roi_display_mode: str,
    roi_alpha: float,
    use_static_mask_cache: bool = True,
    dual_contrast_border: bool = True,
    ground_contact_mode: str = "bottom_center",
    draw_ground_contact_marker: bool = True,
    enable_homography: bool = False,
    homography_landmarks: Optional[List[List[float]]] = None,
    config_path: str = _DEFAULT_CONFIG_PATH,
) -> None:
    """Persist updated perspective and rendering parameters to config.yaml and update cached config."""
    landmarks = homography_landmarks or []
    path = Path(config_path)
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        raw["perspective_and_rendering"] = {
            "roi_display_mode": str(roi_display_mode),
            "roi_alpha": round(float(roi_alpha), 3),
            "use_static_mask_cache": bool(use_static_mask_cache),
            "dual_contrast_border": bool(dual_contrast_border),
            "ground_contact_mode": str(ground_contact_mode),
            "draw_ground_contact_marker": bool(draw_ground_contact_marker),
            "enable_homography": bool(enable_homography),
            "homography_landmarks": landmarks,
        }

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=True)

    global _CURRENT_CONFIG
    with _CONFIG_LOCK:
        if _CURRENT_CONFIG is not None:
            _CURRENT_CONFIG.perspective_and_rendering = PerspectiveAndRenderingConfig(
                roi_display_mode=roi_display_mode,
                roi_alpha=roi_alpha,
                use_static_mask_cache=use_static_mask_cache,
                dual_contrast_border=dual_contrast_border,
                ground_contact_mode=ground_contact_mode,
                draw_ground_contact_marker=draw_ground_contact_marker,
                enable_homography=enable_homography,
                homography_landmarks=landmarks,
            )
