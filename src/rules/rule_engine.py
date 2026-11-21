from dataclasses import dataclass
import json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.ai.tracker import TrackedVehicle
from src.core.config_loader import get_config
from src.core.logger import logger
from src.rules.geometry import (
    check_line_crossing,
    compute_homography_matrix,
    is_vehicle_ground_in_polygon,
    point_in_polygon,
)
from src.rules.state_manager import StateManager


@dataclass
class RuleDefinition:
    id: int
    camera_id: int
    name: str
    rule_type: str  # 'no_parking', 'sidewalk_encroachment', 'lane_roi', 'lane_line'
    geometry: List[Tuple[float, float]]  # Normalized [0..1] points: polygon or line [start, end]
    direction: str  # 'both', 'left_to_right', 'right_to_left'
    allowed_classes: Optional[List[int]]
    forbidden_classes: Optional[List[int]]
    dwell_threshold_sec: float
    is_active: bool


@dataclass
class ViolationEvent:
    camera_id: int
    rule_id: Optional[int]
    track_id: int
    violation_type: str
    vehicle_class_id: int
    vehicle_class_name: str
    detect_conf: float
    bbox: Tuple[float, float, float, float]
    bbox_norm: Tuple[float, float, float, float]  # [cx, cy, w, h] normalized
    anchor: Tuple[float, float]
    anchor_norm: Tuple[float, float]
    dwell_duration_sec: float


class RuleEngine:
    """
    Evaluates vehicle tracks against active camera rules.
    Runs independently per camera.
    """

    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        self.rules: List[RuleDefinition] = []
        self.state_manager = StateManager()

    def set_rules(self, rules_data: List[Dict[str, Any]]) -> None:
        """Update active rules for this camera (supports dynamic hot-reload)."""
        parsed_rules: List[RuleDefinition] = []
        for r in rules_data:
            if not r.get("is_active", 1):
                continue

            # Parse geometry_json
            geom_raw = r.get("geometry_json", "[]")
            try:
                geometry = [tuple(p) for p in json.loads(geom_raw)]
            except Exception as e:
                logger.error(f"Failed to parse rule geometry JSON: {geom_raw}, error: {e}")
                continue

            allowed = None
            if r.get("allowed_classes_json"):
                try:
                    allowed = [int(x) for x in json.loads(r["allowed_classes_json"])]
                except Exception:
                    pass

            forbidden = None
            if r.get("forbidden_classes_json"):
                try:
                    forbidden = [int(x) for x in json.loads(r["forbidden_classes_json"])]
                except Exception:
                    pass

            parsed_rules.append(
                RuleDefinition(
                    id=r["id"],
                    camera_id=self.camera_id,
                    name=r["name"],
                    rule_type=r["rule_type"],
                    geometry=geometry,
                    direction=r.get("direction", "both"),
                    allowed_classes=allowed,
                    forbidden_classes=forbidden,
                    dwell_threshold_sec=float(r.get("dwell_threshold_sec", 10.0)),
                    is_active=bool(r.get("is_active", 1)),
                )
            )

        self.rules = parsed_rules
        logger.info(f"Camera {self.camera_id} loaded {len(self.rules)} active rules.")

        # Homography perspective transformation matrix (if configured)
        self.H: Optional[np.ndarray] = None
        try:
            p_cfg = getattr(get_config(), "perspective_and_rendering", None)
            homo_on = getattr(p_cfg, "enable_homography_calibration", getattr(p_cfg, "enable_homography", False)) if p_cfg else False
            if p_cfg and homo_on and p_cfg.homography_landmarks:
                landmarks = [(float(p[0]), float(p[1])) for p in p_cfg.homography_landmarks]
                if len(landmarks) == 4:
                    self.H = compute_homography_matrix(landmarks)
                    logger.info(f"Camera {self.camera_id} initialized Homography perspective calibration matrix.")
        except Exception as e:
            logger.warning(f"Camera {self.camera_id} could not initialize homography matrix: {e}")

    def evaluate(
        self,
        tracked_vehicles: List[TrackedVehicle],
        frame_shape: Tuple[int, int],
        fps: float = 15.0,
    ) -> List[ViolationEvent]:
        """
        Evaluate all active tracked vehicles against configured rules.
        frame_shape: (height, width)
        Returns list of newly triggered ViolationEvents.
        """
        if not self.rules or not tracked_vehicles:
            return []

        orig_h, orig_w = frame_shape
        exempt_classes = get_config().ai.exempt_classes
        stationary_speed_px_per_sec = get_config().rules.stationary_speed_px_per_sec
        wrong_lane_min_frames = get_config().rules.wrong_lane_min_frames

        # Perspective & Ground-Contact parameters
        p_cfg = getattr(get_config(), "perspective_and_rendering", None)
        ground_mode = getattr(p_cfg, "ground_contact_mode", "bottom_center") if p_cfg else "bottom_center"
        homo_enabled = getattr(p_cfg, "enable_homography_calibration", getattr(p_cfg, "enable_homography", False)) if p_cfg else False
        h_matrix = self.H if homo_enabled else None

        active_track_ids = {v.track_id for v in tracked_vehicles}
        self.state_manager.prune_inactive_tracks(active_track_ids)

        violations: List[ViolationEvent] = []

        for vehicle in tracked_vehicles:
            # Check vehicle exemption (ambulance, fire_truck)
            if vehicle.class_id in exempt_classes:
                continue

            # Update track state in state manager
            state = self.state_manager.update_track(
                track_id=vehicle.track_id,
                class_id=vehicle.class_id,
                bbox=vehicle.bbox,
                anchor=vehicle.anchor,
            )

            # Compute vehicle speed (px/sec) using recent history
            speed_px_per_sec = 0.0
            if len(vehicle.history) >= 2:
                p_prev = vehicle.history[-2]
                p_curr = vehicle.history[-1]
                dist_px = np.hypot(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
                speed_px_per_sec = dist_px * max(1.0, fps)

            is_stationary = speed_px_per_sec <= stationary_speed_px_per_sec

            for rule in self.rules:
                # Class permission check
                if rule.forbidden_classes and vehicle.class_id in rule.forbidden_classes:
                    is_violating_class = True
                elif rule.allowed_classes is not None and vehicle.class_id not in rule.allowed_classes:
                    is_violating_class = True
                elif rule.rule_type in ("no_parking", "sidewalk_encroachment"):
                    is_violating_class = True
                else:
                    is_violating_class = False

                if not is_violating_class:
                    continue

                # 1. Parking / Sidewalk Encroachment (ROI Dwell Time with Ground-Contact)
                if rule.rule_type in ("no_parking", "sidewalk_encroachment"):
                    is_inside = is_vehicle_ground_in_polygon(
                        bbox_norm=(vehicle.cx_norm, vehicle.cy_norm, vehicle.w_norm, vehicle.h_norm),
                        polygon=rule.geometry,
                        mode=ground_mode,
                        H=h_matrix,
                    )
                    dwell_sec = self.state_manager.update_roi_presence(
                        vehicle.track_id, is_inside, is_stationary, rule_type=rule.rule_type
                    )

                    if dwell_sec >= rule.dwell_threshold_sec:
                        if self.state_manager.can_trigger_violation(vehicle.track_id):
                            self.state_manager.mark_violation_triggered(vehicle.track_id, violation_type=rule.rule_type)
                            violations.append(
                                ViolationEvent(
                                    camera_id=self.camera_id,
                                    rule_id=rule.id,
                                    track_id=vehicle.track_id,
                                    violation_type=rule.rule_type,
                                    vehicle_class_id=vehicle.class_id,
                                    vehicle_class_name=vehicle.class_name,
                                    detect_conf=vehicle.conf,
                                    bbox=vehicle.bbox,
                                    bbox_norm=(vehicle.cx_norm, vehicle.cy_norm, vehicle.w_norm, vehicle.h_norm),
                                    anchor=vehicle.anchor,
                                    anchor_norm=vehicle.anchor_norm,
                                    dwell_duration_sec=round(dwell_sec, 1),
                                )
                            )

                # 2. Wrong Lane ROI (lane_roi with Ground-Contact)
                elif rule.rule_type == "lane_roi":
                    is_inside = is_vehicle_ground_in_polygon(
                        bbox_norm=(vehicle.cx_norm, vehicle.cy_norm, vehicle.w_norm, vehicle.h_norm),
                        polygon=rule.geometry,
                        mode=ground_mode,
                        H=h_matrix,
                    )
                    if is_inside:
                        state.consecutive_wrong_lane_frames += 1
                        if state.consecutive_wrong_lane_frames >= wrong_lane_min_frames:
                            if self.state_manager.can_trigger_violation(vehicle.track_id):
                                self.state_manager.mark_violation_triggered(vehicle.track_id, violation_type="wrong_lane")
                                violations.append(
                                    ViolationEvent(
                                        camera_id=self.camera_id,
                                        rule_id=rule.id,
                                        track_id=vehicle.track_id,
                                        violation_type="wrong_lane",
                                        vehicle_class_id=vehicle.class_id,
                                        vehicle_class_name=vehicle.class_name,
                                        detect_conf=vehicle.conf,
                                        bbox=vehicle.bbox,
                                        bbox_norm=(vehicle.cx_norm, vehicle.cy_norm, vehicle.w_norm, vehicle.h_norm),
                                        anchor=vehicle.anchor,
                                        anchor_norm=vehicle.anchor_norm,
                                        dwell_duration_sec=0.0,
                                    )
                                )
                    else:
                        state.consecutive_wrong_lane_frames = 0

                # 3. Wrong Lane Line Crossing (lane_line)
                elif rule.rule_type == "lane_line":
                    if len(rule.geometry) >= 2 and len(vehicle.history) >= 2:
                        p_prev_norm = (vehicle.history[-2][0] / orig_w, vehicle.history[-2][1] / orig_h)
                        p_curr_norm = (vehicle.history[-1][0] / orig_w, vehicle.history[-1][1] / orig_h)
                        line_start = rule.geometry[0]
                        line_end = rule.geometry[1]

                        if check_line_crossing(p_prev_norm, p_curr_norm, line_start, line_end, rule.direction):
                            if self.state_manager.can_trigger_violation(vehicle.track_id):
                                self.state_manager.mark_violation_triggered(vehicle.track_id, violation_type="wrong_lane")
                                violations.append(
                                    ViolationEvent(
                                        camera_id=self.camera_id,
                                        rule_id=rule.id,
                                        track_id=vehicle.track_id,
                                        violation_type="wrong_lane",
                                        vehicle_class_id=vehicle.class_id,
                                        vehicle_class_name=vehicle.class_name,
                                        detect_conf=vehicle.conf,
                                        bbox=vehicle.bbox,
                                        bbox_norm=(vehicle.cx_norm, vehicle.cy_norm, vehicle.w_norm, vehicle.h_norm),
                                        anchor=vehicle.anchor,
                                        anchor_norm=vehicle.anchor_norm,
                                        dwell_duration_sec=0.0,
                                    )
                                )

        return violations
