from datetime import datetime
from pathlib import Path
import json
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional
import cv2
import numpy as np

from src.ai.ocr_worker import AsyncOCRWorkerPool, OCRJob
from src.core.config_loader import get_config
from src.core.deduplication import DeduplicationManager
from src.core.logger import logger
from src.db.repositories import ViolationRepository
from src.rules.rule_engine import ViolationEvent


class ViolationService:
    """
    Handles saving evidence images (raw, annotated, crop, plate),
    recording violations into SQLite WAL, submitting OCR jobs,
    smart deduplication (plate cooldown, spatial lock, session lifecycle), and broadcasting events.
    """

    _instance: Optional["ViolationService"] = None
    _singleton_lock = threading.Lock()

    def __init__(self, db_path: str = "data/traffic.db"):
        self.db_path = db_path
        self.violation_repo = ViolationRepository(db_path)
        self.ocr_pool = AsyncOCRWorkerPool.get_instance(db_path)
        self.dedup = DeduplicationManager.get_instance(db_path)
        self.event_listeners: List[Callable[[Dict[str, Any]], None]] = []
        self.lock = threading.Lock()

        # Listen for OCR completions to update plate cooldown cache
        self.ocr_pool.add_event_listener(self._on_ocr_pool_event)

    def _on_ocr_pool_event(self, event_data: Dict[str, Any]) -> None:
        if event_data.get("type") == "ocr_completed":
            vid = event_data.get("violation_id")
            plate = event_data.get("plate_number")
            if vid and plate:
                self.dedup.update_session_plate(vid, plate)

    @classmethod
    def get_instance(cls, db_path: str = "data/traffic.db") -> "ViolationService":
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls(db_path)
            return cls._instance

    def add_event_listener(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        """Register listener for new violation events (e.g. WebSocket)."""
        with self.lock:
            if listener not in self.event_listeners:
                self.event_listeners.append(listener)

    def remove_event_listener(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        with self.lock:
            if listener in self.event_listeners:
                self.event_listeners.remove(listener)

    def process_violation(
        self,
        event: ViolationEvent,
        raw_frame: np.ndarray,
        annotated_frame: np.ndarray,
        crop_img: np.ndarray,
    ) -> int:
        """
        Check smart deduplication (Plate Cooldown + Spatial/IoU Lock + Session Lifecycle).
        If duplicate: accumulates duration to existing event and broadcasts update.
        If new: saves images, inserts to SQLite, registers active session, and queues OCR.
        Returns violation_id.
        """
        # 1. SMART DEDUPLICATION CHECK (Layer 1 & Layer 2)
        is_dup, existing_id = self.dedup.check_candidate(
            camera_id=event.camera_id,
            rule_id=event.rule_id,
            violation_type=event.violation_type,
            track_id=event.track_id,
            bbox=event.bbox,
            dwell_duration_sec=event.dwell_duration_sec,
        )

        if is_dup and existing_id:
            sess = self.dedup.get_session(existing_id)
            updated_duration = sess.dwell_duration_sec if sess else event.dwell_duration_sec
            repeat_cnt = sess.repeat_count if sess else 1

            # Update SQLite with accumulated duration and last_seen_at
            self.violation_repo.update_duration_and_last_seen(
                violation_id=existing_id,
                dwell_duration_sec=updated_duration,
                repeat_count=repeat_cnt,
            )

            # Broadcast real-time update event to WebSocket clients (UI deduplication)
            update_broadcast = {
                "type": "violation_updated",
                "violation_id": existing_id,
                "camera_id": event.camera_id,
                "track_id": event.track_id,
                "dwell_duration_sec": updated_duration,
                "repeat_count": repeat_cnt,
                "last_seen_at": datetime.now().isoformat(),
                "session_status": "ACTIVE",
            }

            with self.lock:
                listeners = list(self.event_listeners)
            for l in listeners:
                try:
                    l(update_broadcast)
                except Exception:
                    pass

            return existing_id

        # 2. NEW VIOLATION SESSION
        now = datetime.now()
        date_folder = now.strftime("%Y-%m-%d")
        timestamp_str = now.strftime("%Y%m%d_%H%M%S_%f")[:19]

        evidence_base = Path(get_config().paths.evidence_dir) / date_folder / f"cam_{event.camera_id}"
        evidence_base.mkdir(parents=True, exist_ok=True)

        prefix = f"{timestamp_str}_trk{event.track_id}_{event.violation_type}"
        raw_path = str(evidence_base / f"{prefix}_raw.jpg")
        annotated_path = str(evidence_base / f"{prefix}_annotated.jpg")
        crop_path = str(evidence_base / f"{prefix}_crop.jpg")
        plate_path = str(evidence_base / f"{prefix}_plate.jpg")

        # Save images
        cv2.imwrite(raw_path, raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(annotated_path, annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(crop_path, crop_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        bbox_norm_json = json.dumps(list(event.bbox_norm))

        # Insert to SQLite
        violation_id = self.violation_repo.create(
            camera_id=event.camera_id,
            rule_id=event.rule_id,
            track_id=event.track_id,
            violation_type=event.violation_type,
            vehicle_class_id=event.vehicle_class_id,
            vehicle_class_name=event.vehicle_class_name,
            detect_conf=round(event.detect_conf, 3),
            bbox_norm_json=bbox_norm_json,
            dwell_duration_sec=event.dwell_duration_sec,
            raw_image_path=raw_path,
            annotated_image_path=annotated_path,
            crop_image_path=crop_path,
            plate_image_path=None,
            plate_number=None,
            ocr_conf=0.0,
            ocr_status="pending",
            review_status="pending",
            session_status="ACTIVE",
            repeat_count=1,
        )

        # Register session in DeduplicationManager (Layer 3)
        session_id = self.dedup.register_session(
            violation_id=violation_id,
            camera_id=event.camera_id,
            rule_id=event.rule_id,
            violation_type=event.violation_type,
            track_id=event.track_id,
            bbox=event.bbox,
            bbox_norm=event.bbox_norm,
            dwell_duration_sec=event.dwell_duration_sec,
            plate_number=None,
        )

        # Update session_id in DB
        with self.lock:
            try:
                with self.violation_repo.db_session(self.db_path) as conn:
                    conn.execute("UPDATE violations SET session_id = ? WHERE id = ?", (session_id, violation_id))
            except Exception:
                pass

        logger.info(
            f"Recorded violation #{violation_id} (Session: {session_id}) for Camera {event.camera_id} "
            f"(Type: {event.violation_type}, Track: {event.track_id})"
        )

        # Submit to async OCR worker
        ocr_job = OCRJob(
            violation_id=violation_id,
            crop_img=crop_img,
            plate_save_path=plate_path,
            vehicle_class_name=event.vehicle_class_name,
        )
        self.ocr_pool.submit_job(ocr_job)

        # Broadcast new violation event
        broadcast_data = {
            "type": "new_violation",
            "violation": {
                "id": violation_id,
                "session_id": session_id,
                "camera_id": event.camera_id,
                "rule_id": event.rule_id,
                "track_id": event.track_id,
                "violation_type": event.violation_type,
                "vehicle_class_name": event.vehicle_class_name,
                "detect_conf": round(event.detect_conf, 3),
                "dwell_duration_sec": event.dwell_duration_sec,
                "repeat_count": 1,
                "session_status": "ACTIVE",
                "raw_image_path": raw_path,
                "annotated_image_path": annotated_path,
                "crop_image_path": crop_path,
                "plate_image_path": None,
                "plate_number": None,
                "ocr_status": "pending",
                "review_status": "pending",
                "created_at": now.isoformat(),
            },
        }

        with self.lock:
            listeners = list(self.event_listeners)
        for l in listeners:
            try:
                l(broadcast_data)
            except Exception:
                pass

        return violation_id

        return violation_id
