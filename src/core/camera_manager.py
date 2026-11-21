from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import queue
import threading
import time
import cv2
import numpy as np

from src.ai.best_shot import BestShotSelector
from src.ai.class_smoother import ClassSmoother
from src.ai.detector import Detection, SharedDetector
from src.ai.tracker import ByteTracker, TrackedVehicle
from src.core.camera_reader import CameraReader
from src.core.config_loader import get_config
from src.core.logger import logger
from src.core.roi_renderer import ROIRenderer
from src.db.repositories import CameraRepository, RuleRepository
from src.rules.rule_engine import RuleEngine, ViolationEvent


@dataclass
class CameraContext:
    camera_id: int
    name: str
    reader: CameraReader
    tracker: ByteTracker
    smoother: ClassSmoother
    rule_engine: RuleEngine
    best_shot: BestShotSelector
    active_tracks_count: int = 0
    latest_annotated_jpeg: Optional[bytes] = None
    latest_raw_frame: Optional[np.ndarray] = None
    lock: threading.Lock = None  # Initialized in post_init or __init__

    def __post_init__(self):
        if self.lock is None:
            self.lock = threading.Lock()


class CameraManager:
    """
    Coordinates multi-camera pipelines, shared AI inference, tracking,
    rule evaluations, and overlay generation.
    """

    _instance: Optional["CameraManager"] = None
    _singleton_lock = threading.Lock()

    def __init__(self, db_path: str = "data/traffic.db"):
        self.db_path = db_path
        self.cam_repo = CameraRepository(db_path)
        self.rule_repo = RuleRepository(db_path)
        self.detector = SharedDetector.get_instance()
        self.roi_renderer = ROIRenderer()

        self.contexts: Dict[int, CameraContext] = {}
        self.worker_threads: Dict[int, threading.Thread] = {}
        self.running = False
        self.lock = threading.Lock()

        # Callbacks
        self.on_violation: Optional[Callable[[ViolationEvent, np.ndarray, np.ndarray, np.ndarray], None]] = None

    @classmethod
    def get_instance(cls, db_path: str = "data/traffic.db") -> "CameraManager":
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls(db_path)
            return cls._instance

    def set_violation_callback(
        self,
        callback: Callable[[ViolationEvent, np.ndarray, np.ndarray, np.ndarray], None],
    ) -> None:
        """Register handler for violation events: callback(event, raw_frame, annotated_frame, crop_frame)."""
        self.on_violation = callback

    def start_all(self) -> None:
        """Load all active cameras from database and start reader and processing threads."""
        with self.lock:
            if self.running:
                return
            self.running = True

            active_cameras = self.cam_repo.get_all(active_only=True)
            logger.info(f"Starting CameraManager with {len(active_cameras)} active cameras...")

            for cam in active_cameras:
                self._start_camera_context(cam)

    def stop_all(self) -> None:
        """Stop all camera reader and processing threads."""
        with self.lock:
            self.running = False

        for cid, ctx in list(self.contexts.items()):
            ctx.reader.stop()

        for cid, th in list(self.worker_threads.items()):
            if th.is_alive():
                th.join(timeout=2.0)

        with self.lock:
            self.contexts.clear()
            self.worker_threads.clear()
        logger.info("CameraManager stopped all cameras.")

    def _start_camera_context(self, cam_dict: Dict[str, Any]) -> None:
        cid = cam_dict["id"]
        if cid in self.contexts:
            return

        reader = CameraReader(
            camera_id=cid,
            name=cam_dict["name"],
            source_url=cam_dict["source_url"],
            source_type=cam_dict.get("source_type", "rtsp"),
            fps_limit=cam_dict.get("fps_limit", 15),
        )
        tracker = ByteTracker(track_buffer=get_config().tracking.track_buffer)
        smoother = ClassSmoother(window_size=get_config().tracking.class_smoothing_window)
        rule_engine = RuleEngine(camera_id=cid)
        # Load rules from DB
        rules_data = self.rule_repo.get_all(camera_id=cid, active_only=True)
        rule_engine.set_rules(rules_data)

        best_shot = BestShotSelector()

        ctx = CameraContext(
            camera_id=cid,
            name=cam_dict["name"],
            reader=reader,
            tracker=tracker,
            smoother=smoother,
            rule_engine=rule_engine,
            best_shot=best_shot,
        )
        self.contexts[cid] = ctx

        # Start reader
        reader.start()

        # Start processing worker thread
        th = threading.Thread(
            target=self._process_camera_loop,
            args=(ctx,),
            name=f"CameraWorker-{cid}",
            daemon=True,
        )
        self.worker_threads[cid] = th
        th.start()
        logger.info(f"Initialized processing pipeline for Camera [{ctx.name}] (ID: {cid})")

    def reload_rules(self, camera_id: int) -> None:
        """Hot-reload rules for a specific camera."""
        with self.lock:
            ctx = self.contexts.get(camera_id)
            if ctx:
                rules_data = self.rule_repo.get_all(camera_id=camera_id, active_only=True)
                ctx.rule_engine.set_rules(rules_data)
                if self.roi_renderer:
                    self.roi_renderer.clear_cache()
                logger.info(f"Hot-reloaded rules for Camera ID {camera_id}.")

    def reload_camera(self, camera_id: int) -> None:
        """Hot-reload camera settings or restart stream."""
        cam = self.cam_repo.get_by_id(camera_id)
        with self.lock:
            # Stop existing if running
            if camera_id in self.contexts:
                self.contexts[camera_id].reader.stop()
                del self.contexts[camera_id]
            if cam and cam.get("is_active", 1):
                self._start_camera_context(cam)
        logger.info(f"Hot-reloaded Camera ID {camera_id}.")

    def remove_camera(self, camera_id: int) -> None:
        """Stop and remove a camera pipeline."""
        with self.lock:
            if camera_id in self.contexts:
                self.contexts[camera_id].reader.stop()
                del self.contexts[camera_id]
            if camera_id in self.worker_threads:
                del self.worker_threads[camera_id]
        logger.info(f"Removed Camera ID {camera_id}.")

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Telemetry snapshot for ResourceMonitor."""
        cameras_info = []
        total_fps = 0.0
        active_count = 0

        with self.lock:
            all_cams = self.cam_repo.get_all()
            total_cameras = len(all_cams)

            for c in all_cams:
                cid = c["id"]
                ctx = self.contexts.get(cid)
                if ctx and ctx.reader.status == "online":
                    active_count += 1
                    fps = ctx.reader.current_fps
                    total_fps += fps
                    cameras_info.append(
                        {
                            "id": cid,
                            "name": ctx.name,
                            "status": ctx.reader.status,
                            "fps": fps,
                            "active_tracks": ctx.active_tracks_count,
                        }
                    )
                elif ctx:
                    cameras_info.append(
                        {
                            "id": cid,
                            "name": ctx.name,
                            "status": ctx.reader.status,
                            "fps": 0.0,
                            "active_tracks": 0,
                        }
                    )
                else:
                    cameras_info.append(
                        {
                            "id": cid,
                            "name": c["name"],
                            "status": "offline",
                            "fps": 0.0,
                            "active_tracks": 0,
                        }
                    )

        return {
            "active_cameras": active_count,
            "total_cameras": total_cameras,
            "total_fps": round(total_fps, 1),
            "ocr_queue_size": 0,
            "ocr_queue_maxsize": get_config().ocr.queue_maxsize,
            "cameras": cameras_info,
        }

    def get_latest_annotated_jpeg(self, camera_id: int) -> Optional[bytes]:
        ctx = self.contexts.get(camera_id)
        if ctx:
            with ctx.lock:
                return ctx.latest_annotated_jpeg
        return None

    def get_snapshot_jpeg(self, camera_id: int) -> Optional[bytes]:
        ctx = self.contexts.get(camera_id)
        if ctx:
            frame = ctx.reader.get_latest_frame()
            if frame is not None:
                ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    return buf.tobytes()
        return None

    def _process_camera_loop(self, ctx: CameraContext) -> None:
        """Main processing loop for a single camera."""
        cid = ctx.camera_id
        exempt_classes = get_config().ai.exempt_classes
        class_map = get_config().ai.classes

        while self.running and ctx.reader.running:
            try:
                frame = ctx.reader.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            raw_frame_copy = frame.copy()
            frame_h, frame_w = frame.shape[:2]

            # 1. AI Object Detection (Shared ONNX)
            detections = self.detector.detect(frame)

            # 2. Tracking update
            tracked = ctx.tracker.update(detections, (frame_h, frame_w), class_map)
            ctx.active_tracks_count = len(tracked)

            # 3. Class smoothing & Best-shot updates
            active_ids = set()
            for v in tracked:
                active_ids.add(v.track_id)
                smoothed_cid = ctx.smoother.update(v.track_id, v.class_id)
                v.class_id = smoothed_cid
                v.class_name = class_map.get(smoothed_cid, f"class_{smoothed_cid}")

                # Update best-shot candidate
                ctx.best_shot.update(v.track_id, raw_frame_copy, v.bbox, v.conf)

            # Prune departed tracks
            ctx.smoother.prune(active_ids)
            ctx.best_shot.prune(active_ids)

            # 4. Evaluate rules & violations
            fps = ctx.reader.current_fps or 15.0
            violations = ctx.rule_engine.evaluate(tracked, (frame_h, frame_w), fps=fps)

            # 5. Render overlays on frame
            annotated_frame = self._render_annotations(frame, tracked, ctx.rule_engine, fps, exempt_classes)

            # Dispatch violations
            for event in violations:
                best_crop = ctx.best_shot.get_best_shot(event.track_id)
                if best_crop is None:
                    # Fallback to current crop
                    bx1, by1, bx2, by2 = [int(round(x)) for x in event.bbox]
                    bx1, by1 = max(0, bx1), max(0, by1)
                    bx2, by2 = min(frame_w, bx2), min(frame_h, by2)
                    best_crop = raw_frame_copy[by1:by2, bx1:bx2].copy()

                if self.on_violation:
                    try:
                        self.on_violation(event, raw_frame_copy, annotated_frame, best_crop)
                    except Exception as e:
                        logger.error(f"Error executing violation callback: {e}")

            # Compress annotated frame to JPEG for MJPEG stream cache
            ret, buf = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ret:
                jpeg_bytes = buf.tobytes()
                with ctx.lock:
                    ctx.latest_annotated_jpeg = jpeg_bytes
                    ctx.latest_raw_frame = raw_frame_copy

    def _draw_dashed_line(
        self,
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 2,
        dash_len: int = 18,
    ) -> None:
        """Draw dashed line between two points."""
        dist = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        if dist == 0:
            return
        dashes = max(1, int(dist / dash_len))
        for i in range(0, dashes, 2):
            s = (
                int(pt1[0] + (pt2[0] - pt1[0]) * (i / dashes)),
                int(pt1[1] + (pt2[1] - pt1[1]) * (i / dashes)),
            )
            e = (
                int(pt1[0] + (pt2[0] - pt1[0]) * (min(dashes, i + 1) / dashes)),
                int(pt1[1] + (pt2[1] - pt1[1]) * (min(dashes, i + 1) / dashes)),
            )
            cv2.line(img, s, e, color, thickness)

    def _render_annotations(
        self,
        frame: np.ndarray,
        tracked: List[TrackedVehicle],
        rule_engine: RuleEngine,
        fps: float,
        exempt_classes: List[int],
    ) -> np.ndarray:
        """Render ROI polygons, dashed lane lines, bboxes with violation banners, and dwell timers."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        VN_CLASSES = {
            0: "XE O TO",
            1: "XE MAY",
            2: "XE DAP",
            3: "XE BUYT",
            4: "XE TAI",
            5: "BA GAC",
            6: "CUU THUONG",
            7: "CUU HOA",
        }

        # 1. High-Performance Static Cached ROI Rendering (Bounded-box blend & Dual-contrast border)
        self.roi_renderer.render_roi_rules(annotated, rule_engine.rules)

        # 2. Draw Tracked Vehicles & Perspective Ground-Contact Footprints
        now = time.monotonic()
        for v in tracked:
            x1, y1, x2, y2 = [int(round(c)) for c in v.bbox]
            is_exempt = v.class_id in exempt_classes

            # Check if this vehicle is currently violating or recently triggered a violation
            st = rule_engine.state_manager.active_states.get(v.track_id)
            is_violating = False
            violation_title = "SAI LAN"
            dwell = 0.0

            if st:
                dwell = st.accumulated_dwell_sec
                if st.has_triggered_violation or (st.last_violation_time > 0 and (now - st.last_violation_time < 8.0)):
                    is_violating = True
                elif st.consecutive_wrong_lane_frames >= 2:
                    is_violating = True
                elif dwell >= 10.0:
                    is_violating = True
                    violation_title = "DUNG DO"

            if is_exempt:
                box_color = (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                lbl = f"#{v.track_id} {VN_CLASSES.get(v.class_id, v.class_name.upper())} [EXEMPT]"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), box_color, -1)
                cv2.putText(annotated, lbl, (x1 + 3, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
            elif is_violating:
                # RED Bounding box with prominent Violation Banner matching reference image
                box_color = (0, 0, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3)

                # Draw red banner badge on top
                banner_w = max(135, x2 - x1)
                banner_h = 34
                by1 = max(0, y1 - banner_h)
                by2 = y1
                bx1 = x1
                bx2 = x1 + banner_w
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 0, 210), -1)
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                # Text line 1: "! SAI LAN"
                cv2.putText(annotated, f"! {violation_title}", (bx1 + 5, by1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                # Text line 2: "ID: 0319 | XE O TO"
                cv2.putText(annotated, f"ID: {v.track_id:04d} | {VN_CLASSES.get(v.class_id, v.class_name.upper())}", (bx1 + 5, by1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
            else:
                # Normal vehicle: Dodger Blue / Cyan bounding box
                box_color = (255, 175, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                dwell_str = f" | {dwell:.1f}s" if dwell > 0 else ""
                lbl = f"#{v.track_id} {VN_CLASSES.get(v.class_id, v.class_name.upper())} {v.conf:.2f}{dwell_str}"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), box_color, -1)
                cv2.putText(annotated, lbl, (x1 + 3, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)

            # Draw high-visibility perspective ground-contact footpad & wheel-base line
            self.roi_renderer.render_ground_contact_marker(annotated, v.bbox, is_violating=is_violating)

        # Draw HUD status top-left
        hud_text = f"FPS: {fps:.1f} | Active Tracks: {len(tracked)}"
        cv2.putText(annotated, hud_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        return annotated
