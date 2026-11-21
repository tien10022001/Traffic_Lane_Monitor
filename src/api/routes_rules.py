from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.core.camera_manager import CameraManager
from src.core.config_loader import get_config
from src.db.repositories import RuleRepository

router = APIRouter(prefix="/api/cameras/{camera_id}/rules", tags=["Rules"])


class RuleCreateRequest(BaseModel):
    name: str
    rule_type: str  # 'no_parking', 'sidewalk_encroachment', 'lane_roi', 'lane_line'
    geometry_json: str  # JSON list of normalized [x, y] coordinates
    direction: str = "both"  # 'both', 'left_to_right', 'right_to_left'
    allowed_classes_json: Optional[str] = None
    forbidden_classes_json: Optional[str] = None
    dwell_threshold_sec: float = 10.0
    is_active: int = 1


class RuleUpdateRequest(BaseModel):
    name: Optional[str] = None
    rule_type: Optional[str] = None
    geometry_json: Optional[str] = None
    direction: Optional[str] = None
    allowed_classes_json: Optional[str] = None
    forbidden_classes_json: Optional[str] = None
    dwell_threshold_sec: Optional[float] = None
    is_active: Optional[int] = None


@router.get("", response_model=List[Dict[str, Any]])
def list_rules(camera_id: int, active_only: bool = False):
    db_path = get_config().paths.db_path
    repo = RuleRepository(db_path)
    return repo.get_all(camera_id=camera_id, active_only=active_only)


@router.post("", status_code=201)
def create_rule(camera_id: int, payload: RuleCreateRequest):
    db_path = get_config().paths.db_path
    repo = RuleRepository(db_path)
    rule_id = repo.create(
        camera_id=camera_id,
        name=payload.name,
        rule_type=payload.rule_type,
        geometry_json=payload.geometry_json,
        direction=payload.direction,
        allowed_classes_json=payload.allowed_classes_json,
        forbidden_classes_json=payload.forbidden_classes_json,
        dwell_threshold_sec=payload.dwell_threshold_sec,
        is_active=payload.is_active,
    )
    # Hot-reload in camera manager
    manager = CameraManager.get_instance(db_path)
    manager.reload_rules(camera_id)
    return {"id": rule_id, "message": "Rule created and hot-reloaded successfully"}


@router.put("/{rule_id}")
def update_rule(camera_id: int, rule_id: int, payload: RuleUpdateRequest):
    db_path = get_config().paths.db_path
    repo = RuleRepository(db_path)
    updated = repo.update(
        rule_id=rule_id,
        name=payload.name,
        rule_type=payload.rule_type,
        geometry_json=payload.geometry_json,
        direction=payload.direction,
        allowed_classes_json=payload.allowed_classes_json,
        forbidden_classes_json=payload.forbidden_classes_json,
        dwell_threshold_sec=payload.dwell_threshold_sec,
        is_active=payload.is_active,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Rule not found or no changes made")

    # Hot-reload in camera manager
    manager = CameraManager.get_instance(db_path)
    manager.reload_rules(camera_id)
    return {"message": "Rule updated and hot-reloaded successfully"}


@router.delete("/{rule_id}")
def delete_rule(camera_id: int, rule_id: int):
    db_path = get_config().paths.db_path
    repo = RuleRepository(db_path)
    deleted = repo.delete(rule_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Rule not found")

    # Hot-reload in camera manager
    manager = CameraManager.get_instance(db_path)
    manager.reload_rules(camera_id)
    return {"message": "Rule deleted and hot-reloaded successfully"}
