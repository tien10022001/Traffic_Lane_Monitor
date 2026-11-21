from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time
from src.core.config_loader import RulesConfig, get_config
from src.rules.geometry import compute_iou


@dataclass
class TrackState:
    track_id: int
    class_id: int
    first_seen_monotonic: float
    last_seen_monotonic: float
    bbox: Tuple[float, float, float, float]
    anchor: Tuple[float, float]
    # Dwell timing
    in_roi_since: Optional[float] = None
    accumulated_dwell_sec: float = 0.0
    last_in_roi_monotonic: Optional[float] = None
    # Violation state
    has_triggered_violation: bool = False
    last_violation_time: float = 0.0
    last_violation_type: Optional[str] = None
    current_rule_type: Optional[str] = None
    # Consecutive frame counts for wrong_lane
    consecutive_wrong_lane_frames: int = 0


@dataclass
class OcclusionRecord:
    class_id: int
    bbox: Tuple[float, float, float, float]
    accumulated_dwell_sec: float
    disappeared_at_monotonic: float


class StateManager:
    """
    Manages track state, monotonic dwell time calculation, grace periods,
    and occlusion recovery per camera.
    """

    def __init__(self, config: Optional[RulesConfig] = None):
        self.config = config or get_config().rules
        self.active_states: Dict[int, TrackState] = {}
        self.occlusion_buffer: List[OcclusionRecord] = []

    def update_track(
        self,
        track_id: int,
        class_id: int,
        bbox: Tuple[float, float, float, float],
        anchor: Tuple[float, float],
    ) -> TrackState:
        now = time.monotonic()

        # Clean expired occlusion records
        max_gap = self.config.occlusion_max_gap_sec
        self.occlusion_buffer = [
            rec for rec in self.occlusion_buffer if (now - rec.disappeared_at_monotonic) <= max_gap
        ]

        if track_id not in self.active_states:
            # Check if this new track can inherit dwell time from recently occluded track
            inherited_dwell = 0.0
            best_iou = 0.0
            best_idx = -1

            for idx, rec in enumerate(self.occlusion_buffer):
                if rec.class_id == class_id:
                    iou = compute_iou(bbox, rec.bbox)
                    if iou >= self.config.occlusion_iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_idx = idx

            if best_idx >= 0:
                inherited_dwell = self.occlusion_buffer[best_idx].accumulated_dwell_sec
                self.occlusion_buffer.pop(best_idx)

            state = TrackState(
                track_id=track_id,
                class_id=class_id,
                first_seen_monotonic=now,
                last_seen_monotonic=now,
                bbox=bbox,
                anchor=anchor,
                accumulated_dwell_sec=inherited_dwell,
                in_roi_since=now if inherited_dwell > 0 else None,
                last_in_roi_monotonic=now if inherited_dwell > 0 else None,
            )
            self.active_states[track_id] = state
        else:
            state = self.active_states[track_id]
            state.last_seen_monotonic = now
            state.bbox = bbox
            state.anchor = anchor
            state.class_id = class_id

        return state

    def update_roi_presence(
        self,
        track_id: int,
        is_inside_roi: bool,
        is_stationary: bool = True,
        rule_type: str = "no_parking",
    ) -> float:
        """
        Updates ROI dwell time using time.monotonic() and grace period.
        Returns total current accumulated dwell time in seconds.
        """
        state = self.active_states.get(track_id)
        if not state:
            return 0.0

        now = time.monotonic()
        grace_period = self.config.grace_period_sec

        if is_inside_roi and is_stationary:
            state.current_rule_type = rule_type
            if state.in_roi_since is None:
                state.in_roi_since = now
            # Accumulate time
            delta = now - (state.last_in_roi_monotonic or now)
            state.accumulated_dwell_sec += max(0.0, delta)
            state.last_in_roi_monotonic = now
        else:
            # Outside ROI or moving
            if state.last_in_roi_monotonic is not None:
                time_outside = now - state.last_in_roi_monotonic
                if time_outside > grace_period:
                    # Grace period expired, reset dwell timer
                    state.accumulated_dwell_sec = 0.0
                    state.in_roi_since = None
                    state.last_in_roi_monotonic = None
                    state.current_rule_type = None

        return state.accumulated_dwell_sec

    def can_trigger_violation(self, track_id: int) -> bool:
        """Check if track can trigger violation (respects cooldown and one-shot rule)."""
        state = self.active_states.get(track_id)
        if not state:
            return False

        now = time.monotonic()
        if not state.has_triggered_violation:
            return True

        # Check cooldown
        if (now - state.last_violation_time) >= self.config.violation_cooldown_sec:
            return True

        return False

    def mark_violation_triggered(self, track_id: int, violation_type: str = "no_parking") -> None:
        state = self.active_states.get(track_id)
        if state:
            state.has_triggered_violation = True
            state.last_violation_time = time.monotonic()
            state.last_violation_type = violation_type

    def prune_inactive_tracks(self, active_track_ids: set) -> None:
        """Save disappeared tracks to occlusion buffer and remove from active states."""
        now = time.monotonic()
        disappeared_ids = [tid for tid in self.active_states if tid not in active_track_ids]

        for tid in disappeared_ids:
            state = self.active_states[tid]
            # If it had accumulated dwell time, save to occlusion buffer
            if state.accumulated_dwell_sec > 1.0:
                self.occlusion_buffer.append(
                    OcclusionRecord(
                        class_id=state.class_id,
                        bbox=state.bbox,
                        accumulated_dwell_sec=state.accumulated_dwell_sec,
                        disappeared_at_monotonic=now,
                    )
                )
            del self.active_states[tid]
