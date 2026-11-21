from typing import Any, Dict, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel
from src.ai.detector import SharedDetector
from src.core.config_loader import (
    get_config,
    save_ai_thresholds,
    save_deduplication_config,
    save_perspective_config,
)
from src.core.logger import logger
from src.monitoring.resource_monitor import ResourceMonitor

router = APIRouter(prefix="/api", tags=["System & Resources"])


class AIThresholdUpdateRequest(BaseModel):
    default_conf: float
    iou_threshold: float
    per_class_conf: Dict[int, float]


class DeduplicationUpdateRequest(BaseModel):
    enabled: bool = True
    plate_cooldown_sec: float = 600.0
    false_positive_cooldown_sec: float = 1800.0
    max_plate_edit_dist: int = 1
    spatial_iou_threshold: float = 0.65
    exit_grace_sec: float = 45.0
    update_existing_duration: bool = True


class PerspectiveUpdateRequest(BaseModel):
    roi_display_mode: str = "full"  # "full", "border_only", "striped", "hidden"
    roi_alpha: float = 0.22
    use_static_mask_cache: bool = True
    dual_contrast_border: bool = True
    ground_contact_mode: str = "bottom_center"  # "bottom_center" or "bottom_strip_20"
    draw_ground_contact_marker: bool = True
    enable_homography_calibration: bool = False
    homography_landmarks: List[List[float]] = []


@router.get("/config/perspective")
def get_perspective_config() -> Dict[str, Any]:
    """Return current perspective and rendering configuration."""
    cfg = get_config().perspective_and_rendering
    return cfg.model_dump()


@router.put("/config/perspective")
def update_perspective_config(payload: PerspectiveUpdateRequest) -> Dict[str, Any]:
    """Update perspective and ROI rendering settings dynamically and persist to config.yaml."""
    save_perspective_config(
        roi_display_mode=payload.roi_display_mode,
        roi_alpha=payload.roi_alpha,
        use_static_mask_cache=payload.use_static_mask_cache,
        dual_contrast_border=payload.dual_contrast_border,
        ground_contact_mode=payload.ground_contact_mode,
        draw_ground_contact_marker=payload.draw_ground_contact_marker,
        enable_homography=payload.enable_homography_calibration,
        homography_landmarks=payload.homography_landmarks,
    )
    # Clear cached masks in CameraManager so new visual styles apply immediately
    try:
        from src.core.camera_manager import CameraManager
        db_path = get_config().paths.db_path
        manager = CameraManager.get_instance(db_path)
        if manager and manager.roi_renderer:
            manager.roi_renderer.clear_cache()
    except Exception as e:
        logger.warning(f"Could not clear roi_renderer mask cache: {e}")

    return {
        "message": "Perspective and rendering config updated successfully",
        "config": payload.model_dump(),
    }


@router.get("/config/deduplication")
def get_deduplication_config() -> Dict[str, Any]:
    """Return current deduplication configuration."""
    cfg = get_config().deduplication
    return cfg.model_dump()


@router.put("/config/deduplication")
def update_deduplication_config(payload: DeduplicationUpdateRequest) -> Dict[str, Any]:
    """Update deduplication settings dynamically and persist to config.yaml."""
    save_deduplication_config(
        enabled=payload.enabled,
        plate_cooldown_sec=payload.plate_cooldown_sec,
        false_positive_cooldown_sec=payload.false_positive_cooldown_sec,
        max_plate_edit_dist=payload.max_plate_edit_dist,
        spatial_iou_threshold=payload.spatial_iou_threshold,
        exit_grace_sec=payload.exit_grace_sec,
        update_existing_duration=payload.update_existing_duration,
    )
    return {
        "message": "Deduplication config updated successfully",
        "config": payload.model_dump(),
    }


@router.get("/config/ai")
def get_ai_config() -> Dict[str, Any]:
    """Return current AI thresholds and class mappings."""
    cfg = get_config().ai
    detector = SharedDetector._instance
    if detector is not None:
        return {
            "default_conf": detector.config.default_conf,
            "iou_threshold": detector.config.iou_threshold,
            "per_class_conf": detector.config.per_class_conf,
            "classes": detector.config.classes,
        }
    return {
        "default_conf": cfg.default_conf,
        "iou_threshold": cfg.iou_threshold,
        "per_class_conf": cfg.per_class_conf,
        "classes": cfg.classes,
    }


@router.put("/config/ai")
def update_ai_config(payload: AIThresholdUpdateRequest) -> Dict[str, Any]:
    """Update AI detection thresholds dynamically in memory and persist to config.yaml."""
    detector = SharedDetector.get_instance()
    detector.update_thresholds(
        default_conf=payload.default_conf,
        iou_threshold=payload.iou_threshold,
        per_class_conf=payload.per_class_conf,
    )

    save_ai_thresholds(
        default_conf=payload.default_conf,
        iou_threshold=payload.iou_threshold,
        per_class_conf=payload.per_class_conf,
    )

    return {
        "message": "AI thresholds updated successfully",
        "default_conf": detector.config.default_conf,
        "iou_threshold": detector.config.iou_threshold,
        "per_class_conf": detector.config.per_class_conf,
    }


@router.get("/health")
def get_health() -> Dict[str, Any]:
    monitor = ResourceMonitor.get_instance()
    snap = monitor.get_latest_snapshot()
    return {
        "status": "ok",
        "timestamp": snap.get("timestamp"),
        "uptime_sec": snap.get("uptime_sec", 0),
        "gpu_available": snap.get("gpu", {}).get("available", False),
        "active_provider": snap.get("gpu", {}).get("provider", "CPUExecutionProvider"),
    }


@router.get("/system/resources")
def get_system_resources() -> Dict[str, Any]:
    """Get current snapshot of host, process, GPU, and pipeline metrics."""
    monitor = ResourceMonitor.get_instance()
    return monitor.get_latest_snapshot()


@router.get("/system/history")
def get_system_history() -> List[Dict[str, Any]]:
    """Get recent time-series metrics from ring buffer (up to 180 points)."""
    monitor = ResourceMonitor.get_instance()
    return monitor.get_history()
