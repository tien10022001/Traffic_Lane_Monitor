"""
Smart Violation Deduplication & Event Session Management Module.
Implements 4-layer deduplication:
1. Plate Cooldown with fuzzy Levenshtein distance matching.
2. Spatial / IoU Lock for stationary vehicles without readable plates.
3. Event Session Lifecycle (ACTIVE -> CLOSED on exit_grace_sec expiration).
4. Review Status Separation (extended cooldown on false positives).
"""

from dataclasses import dataclass, field
from datetime import datetime
import re
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.config_loader import DeduplicationConfig, get_config
from src.core.logger import logger
from src.rules.geometry import compute_iou


def normalize_plate(plate: Optional[str]) -> str:
    """Strip all non-alphanumeric characters and convert to uppercase."""
    if not plate:
        return ""
    return re.sub(r"[^A-Z0-9]", "", plate.upper())


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev[j + 1] + 1
            deletions = curr[j] + 1
            substitutions = prev[j] + (c1 != c2)
            curr.append(min(insertions, deletions, substitutions))
        prev = curr
    return prev[-1]


def is_plate_match(p1: str, p2: str, max_edit_dist: int = 1) -> bool:
    """
    Check if two normalized plates match within max_edit_dist tolerance.
    Requires minimum length of 4 characters to prevent false matches on noise.
    """
    n1, n2 = normalize_plate(p1), normalize_plate(p2)
    if len(n1) < 4 or len(n2) < 4:
        return False
    if n1 == n2:
        return True
    if abs(len(n1) - len(n2)) > max_edit_dist:
        return False
    return levenshtein_distance(n1, n2) <= max_edit_dist


@dataclass
class ViolationSession:
    session_id: str
    violation_id: int
    camera_id: int
    rule_id: Optional[int]
    violation_type: str
    track_id: int
    bbox: Tuple[float, float, float, float]
    bbox_norm: Tuple[float, float, float, float]
    normalized_plate: str = ""
    start_time_monotonic: float = field(default_factory=time.monotonic)
    last_seen_monotonic: float = field(default_factory=time.monotonic)
    dwell_duration_sec: float = 0.0
    repeat_count: int = 1
    status: str = "ACTIVE"  # 'ACTIVE' or 'CLOSED'
    review_status: str = "pending"  # 'pending', 'confirmed', 'false_positive'
    cooldown_until_monotonic: float = 0.0


@dataclass
class PlateCooldownEntry:
    camera_id: int
    rule_id: Optional[int]
    violation_type: str
    normalized_plate: str
    violation_id: int
    session_id: str
    expires_at_monotonic: float
    is_false_positive: bool = False


class DeduplicationManager:
    """
    Central manager for violation deduplication, spatial locking,
    plate cooldown cache, and session lifecycle.
    """

    _instance: Optional["DeduplicationManager"] = None
    _singleton_lock = threading.Lock()

    def __init__(self, db_path: str = "data/traffic.db"):
        self.db_path = db_path
        self.lock = threading.RLock()
        # Active sessions: violation_id -> ViolationSession
        self.sessions: Dict[int, ViolationSession] = {}
        # Mapping camera_id -> Set of active violation_ids
        self.active_camera_sessions: Dict[int, Set[int]] = {}
        # Plate cooldown cache: (camera_id, violation_type) -> List[PlateCooldownEntry]
        self.plate_cache: Dict[Tuple[int, str], List[PlateCooldownEntry]] = {}

    @classmethod
    def get_instance(cls, db_path: str = "data/traffic.db") -> "DeduplicationManager":
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls(db_path)
            return cls._instance

    def get_config(self) -> DeduplicationConfig:
        try:
            return get_config().deduplication
        except Exception:
            return DeduplicationConfig()

    def check_candidate(
        self,
        camera_id: int,
        rule_id: Optional[int],
        violation_type: str,
        track_id: int,
        bbox: Tuple[float, float, float, float],
        dwell_duration_sec: float,
        plate_candidate: Optional[str] = None,
    ) -> Tuple[bool, Optional[int]]:
        """
        Check if an incoming violation candidate matches an existing active session
        or cooldown entry.
        Returns:
            (is_duplicate, existing_violation_id)
        """
        cfg = self.get_config()
        if not cfg.enabled:
            return False, None

        now = time.monotonic()

        with self.lock:
            self._cleanup_expired(now)

            # LAYER 1: Check Plate Cooldown (if candidate has a plate)
            norm_cand_plate = normalize_plate(plate_candidate)
            if norm_cand_plate and len(norm_cand_plate) >= 4:
                cache_key = (camera_id, violation_type)
                entries = self.plate_cache.get(cache_key, [])
                for entry in entries:
                    if entry.expires_at_monotonic > now:
                        # Check rule_id match if both specify it
                        if rule_id is not None and entry.rule_id is not None and rule_id != entry.rule_id:
                            continue
                        if is_plate_match(norm_cand_plate, entry.normalized_plate, cfg.max_plate_edit_dist):
                            # Matched existing plate in cooldown
                            matched_vid = entry.violation_id
                            self._touch_session(matched_vid, bbox, dwell_duration_sec, now)
                            logger.info(
                                f"[DEDUP:PLATE] Camera {camera_id} matched plate [{norm_cand_plate}] "
                                f"~ [{entry.normalized_plate}] -> existing violation #{matched_vid}"
                            )
                            return True, matched_vid

            # LAYER 2: Spatial / IoU Lock
            active_vids = self.active_camera_sessions.get(camera_id, set())
            for vid in list(active_vids):
                sess = self.sessions.get(vid)
                if not sess or sess.status != "ACTIVE":
                    continue
                if sess.violation_type != violation_type:
                    continue
                # If rule_id is specified on both, ensure same rule
                if rule_id is not None and sess.rule_id is not None and rule_id != sess.rule_id:
                    continue

                # Compute IoU between candidate and locked session bbox
                iou = compute_iou(bbox, sess.bbox)
                if iou >= cfg.spatial_iou_threshold:
                    # Same stationary vehicle in lock!
                    self._touch_session(vid, bbox, dwell_duration_sec, now)
                    logger.info(
                        f"[DEDUP:SPATIAL] Camera {camera_id} bbox IoU {iou:.2f} >= {cfg.spatial_iou_threshold} "
                        f"-> existing violation #{vid} (Track #{sess.track_id} -> #{track_id})"
                    )
                    return True, vid

        return False, None

    def _touch_session(
        self,
        violation_id: int,
        bbox: Tuple[float, float, float, float],
        dwell_duration_sec: float,
        now: float,
    ) -> None:
        """Update existing session duration, repeat counter, and last_seen time."""
        sess = self.sessions.get(violation_id)
        if not sess:
            return
        sess.last_seen_monotonic = now
        sess.repeat_count += 1
        sess.bbox = bbox
        if dwell_duration_sec > sess.dwell_duration_sec:
            sess.dwell_duration_sec = dwell_duration_sec

    def register_session(
        self,
        violation_id: int,
        camera_id: int,
        rule_id: Optional[int],
        violation_type: str,
        track_id: int,
        bbox: Tuple[float, float, float, float],
        bbox_norm: Tuple[float, float, float, float],
        dwell_duration_sec: float,
        plate_number: Optional[str] = None,
    ) -> str:
        """
        Register a newly created violation as an ACTIVE event session.
        Returns the session_id.
        """
        now = time.monotonic()
        session_id = f"sess_{camera_id}_{violation_id}_{int(now)}"
        norm_plate = normalize_plate(plate_number)

        with self.lock:
            sess = ViolationSession(
                session_id=session_id,
                violation_id=violation_id,
                camera_id=camera_id,
                rule_id=rule_id,
                violation_type=violation_type,
                track_id=track_id,
                bbox=bbox,
                bbox_norm=bbox_norm,
                normalized_plate=norm_plate,
                start_time_monotonic=now,
                last_seen_monotonic=now,
                dwell_duration_sec=dwell_duration_sec,
                repeat_count=1,
                status="ACTIVE",
                review_status="pending",
            )
            self.sessions[violation_id] = sess

            if camera_id not in self.active_camera_sessions:
                self.active_camera_sessions[camera_id] = set()
            self.active_camera_sessions[camera_id].add(violation_id)

            if norm_plate and len(norm_plate) >= 4:
                self._add_plate_entry(sess, norm_plate, now)

        logger.info(
            f"[SESSION:START] Created ACTIVE session '{session_id}' for violation #{violation_id} (Cam #{camera_id})"
        )
        return session_id

    def update_session_plate(self, violation_id: int, plate_number: str) -> None:
        """Called when OCR finishes recognizing a plate for a violation."""
        norm_plate = normalize_plate(plate_number)
        if not norm_plate or len(norm_plate) < 4:
            return

        now = time.monotonic()
        with self.lock:
            sess = self.sessions.get(violation_id)
            if sess:
                sess.normalized_plate = norm_plate
                self._add_plate_entry(sess, norm_plate, now)
                logger.info(
                    f"[SESSION:PLATE] Updated plate [{norm_plate}] for violation #{violation_id} (Session {sess.session_id})"
                )

    def _add_plate_entry(self, sess: ViolationSession, norm_plate: str, now: float) -> None:
        cfg = self.get_config()
        ttl = cfg.false_positive_cooldown_sec if sess.review_status == "false_positive" else cfg.plate_cooldown_sec
        cache_key = (sess.camera_id, sess.violation_type)
        if cache_key not in self.plate_cache:
            self.plate_cache[cache_key] = []

        # Remove existing entry for same violation_id if any
        self.plate_cache[cache_key] = [
            e for e in self.plate_cache[cache_key] if e.violation_id != sess.violation_id
        ]

        self.plate_cache[cache_key].append(
            PlateCooldownEntry(
                camera_id=sess.camera_id,
                rule_id=sess.rule_id,
                violation_type=sess.violation_type,
                normalized_plate=norm_plate,
                violation_id=sess.violation_id,
                session_id=sess.session_id,
                expires_at_monotonic=now + ttl,
                is_false_positive=(sess.review_status == "false_positive"),
            )
        )

    def mark_review_status(self, violation_id: int, review_status: str) -> None:
        """
        Update review status on session.
        If marked 'false_positive', extends cooldown to false_positive_cooldown_sec.
        Does not close session so the vehicle remains locked.
        """
        now = time.monotonic()
        cfg = self.get_config()

        with self.lock:
            sess = self.sessions.get(violation_id)
            if not sess:
                return

            sess.review_status = review_status
            if review_status == "false_positive":
                ext_cooldown = now + cfg.false_positive_cooldown_sec
                sess.cooldown_until_monotonic = ext_cooldown
                # Also update in plate cache
                if sess.normalized_plate:
                    cache_key = (sess.camera_id, sess.violation_type)
                    for entry in self.plate_cache.get(cache_key, []):
                        if entry.violation_id == violation_id:
                            entry.expires_at_monotonic = ext_cooldown
                            entry.is_false_positive = True
                logger.info(
                    f"[SESSION:FALSE_POSITIVE] Violation #{violation_id} locked with extended cooldown "
                    f"({cfg.false_positive_cooldown_sec}s)"
                )

    def _cleanup_expired(self, now: float) -> None:
        """
        Clean up expired plate cache entries and close sessions where vehicle
        has disappeared continuously beyond exit_grace_sec.
        """
        cfg = self.get_config()

        # 1. Close sessions exceeding exit_grace_sec
        for cam_id, vids in list(self.active_camera_sessions.items()):
            to_remove = set()
            for vid in list(vids):
                sess = self.sessions.get(vid)
                if not sess:
                    to_remove.add(vid)
                    continue

                if sess.status == "ACTIVE":
                    time_since_seen = now - sess.last_seen_monotonic
                    if time_since_seen > cfg.exit_grace_sec:
                        sess.status = "CLOSED"
                        to_remove.add(vid)
                        logger.info(
                            f"[SESSION:CLOSED] Vehicle left ROI (time outside {time_since_seen:.1f}s > "
                            f"{cfg.exit_grace_sec}s) -> closed session '{sess.session_id}' for #{vid}"
                        )
                else:
                    to_remove.add(vid)

            self.active_camera_sessions[cam_id] -= to_remove

        # 2. Cleanup plate cache entries
        for cache_key, entries in list(self.plate_cache.items()):
            active_entries = [e for e in entries if e.expires_at_monotonic > now]
            if active_entries:
                self.plate_cache[cache_key] = active_entries
            else:
                self.plate_cache.pop(cache_key, None)

    def get_session(self, violation_id: int) -> Optional[ViolationSession]:
        with self.lock:
            return self.sessions.get(violation_id)

    def get_active_sessions_for_camera(self, camera_id: int) -> List[ViolationSession]:
        with self.lock:
            vids = self.active_camera_sessions.get(camera_id, set())
            return [self.sessions[vid] for vid in vids if vid in self.sessions]

    def clear_cache(self) -> None:
        """Reset and purge all active sessions, spatial locks and cooldown entries."""
        with self.lock:
            self.sessions.clear()
            self.active_camera_sessions.clear()
            self.plate_cache.clear()
        logger.info("[DEDUP:RESET] Cleared all active deduplication sessions and caches.")
