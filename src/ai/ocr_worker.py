from dataclasses import dataclass
from pathlib import Path
import queue
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.core.config_loader import OCRConfig, get_config
from src.core.logger import logger
from src.db.repositories import ViolationRepository


@dataclass
class OCRJob:
    violation_id: int
    crop_img: np.ndarray
    plate_save_path: str
    vehicle_class_name: str = "car"


def clean_vietnamese_plate(raw_text: str) -> str:
    """Clean and standardize Vietnamese license plate string."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()
    return cleaned


def format_vietnamese_plate(cleaned: str, vehicle_class_name: str = "car") -> str:
    """
    Format plate nicely according to official Vietnamese license plate regulations:
    - Cars/Trucks/Buses:
        * 5 digits: 92A00529 -> 92A-005.29, 51K73218 -> 51K-732.18, 51LD12345 -> 51LD-123.45
        * 4 digits: 29A1234 -> 29A-1234
    - Motorcycles:
        * 5 digits: 29B112345 -> 29B1-123.45
        * 4 digits: 29B11234 -> 29B1-1234
    """
    cleaned = re.sub(r"[^A-Z0-9]", "", cleaned.upper())
    n = len(cleaned)
    if n < 5:
        return cleaned

    prov = cleaned[:2]
    rest = cleaned[2:]

    is_car_type = vehicle_class_name.lower() in ("car", "truck", "bus", "ambulance", "fire_truck")

    if is_car_type:
        # Pattern 1: 1 letter series + 4 or 5 digits (e.g. 92A-005.29 or 29A-1234)
        m1 = re.match(r"^([A-Z])(\d{4,5})$", rest)
        if m1:
            series, digits = m1.group(1), m1.group(2)
            if len(digits) == 5:
                return f"{prov}{series}-{digits[:3]}.{digits[3:]}"
            return f"{prov}{series}-{digits}"

        # Pattern 2: 2 letters series + 4 or 5 digits (e.g. 51LD-123.45)
        m2 = re.match(r"^([A-Z]{2})(\d{4,5})$", rest)
        if m2:
            series, digits = m2.group(1), m2.group(2)
            if len(digits) == 5:
                return f"{prov}{series}-{digits[:3]}.{digits[3:]}"
            return f"{prov}{series}-{digits}"

        # Pattern 3: OCR artifact (e.g. extra digit between series and number)
        m3 = re.match(r"^([A-Z0-9]{1,2})[0-9]?(\d{5})$", rest)
        if m3:
            return f"{prov}{m3.group(1)}-{m3.group(2)[:3]}.{m3.group(2)[3:]}"
    else:
        # Motorcycle series: 1 letter + 1 digit (e.g. 29B1) or 2 letters (50cc: 29AA)
        m_moto = re.match(r"^([A-Z][0-9A-Z])(\d{4,5})$", rest)
        if m_moto:
            series, digits = m_moto.group(1), m_moto.group(2)
            if len(digits) == 5:
                return f"{prov}{series}-{digits[:3]}.{digits[3:]}"
            return f"{prov}{series}-{digits}"

    # General fallback
    if n == 7:
        return f"{cleaned[:3]}-{cleaned[3:]}"
    elif n == 8:
        if cleaned[3:].isdigit():
            return f"{cleaned[:3]}-{cleaned[3:6]}.{cleaned[6:]}"
        return f"{cleaned[:4]}-{cleaned[4:]}"
    elif n == 9:
        return f"{cleaned[:4]}-{cleaned[4:7]}.{cleaned[7:]}"

    return cleaned


class AsyncOCRWorkerPool:
    """
    Background worker pool for asynchronous Vietnamese license plate OCR.
    Consumes jobs from a bounded queue and updates SQLite records without blocking live stream.
    """

    _instance: Optional["AsyncOCRWorkerPool"] = None
    _singleton_lock = threading.Lock()

    def __init__(self, db_path: str = "data/traffic.db", config: Optional[OCRConfig] = None):
        self.db_path = db_path
        self.config = config or get_config().ocr
        self.violation_repo = ViolationRepository(db_path)

        self.queue: queue.Queue = queue.Queue(maxsize=self.config.queue_maxsize)
        self.workers: List[threading.Thread] = []
        self.running = False
        self.lock = threading.Lock()

        # Engine cache (RapidOCR initialized in worker thread or shared)
        self.listeners: List[Callable[[Dict[str, Any]], None]] = []

    @classmethod
    def get_instance(cls, db_path: str = "data/traffic.db") -> "AsyncOCRWorkerPool":
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls(db_path)
            return cls._instance

    def add_event_listener(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        """Add WebSocket listener for OCR completion events."""
        with self.lock:
            if listener not in self.listeners:
                self.listeners.append(listener)

    def remove_event_listener(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        with self.lock:
            if listener in self.listeners:
                self.listeners.remove(listener)

    # Convenience aliases
    add_listener = add_event_listener
    remove_listener = remove_event_listener

    def submit_job(self, job: OCRJob) -> bool:
        """Submit an OCR job to the worker queue."""
        if not self.running:
            return False

        try:
            self.queue.put_nowait(job)
            return True
        except queue.Full:
            logger.warning(
                f"OCR Queue is full ({self.config.queue_maxsize} jobs). Dropping OCR job for violation {job.violation_id}."
            )
            # Mark as failed in DB
            self.violation_repo.update_ocr(job.violation_id, "", 0.0, "failed")
            return False

    def get_queue_size(self) -> int:
        return self.queue.qsize()

    def start(self) -> None:
        with self.lock:
            if self.running:
                return
            self.running = True

            worker_count = max(1, self.config.workers)
            for i in range(worker_count):
                th = threading.Thread(
                    target=self._worker_loop,
                    name=f"OCRWorker-{i + 1}",
                    daemon=True,
                )
                self.workers.append(th)
                th.start()
            logger.info(f"Started {worker_count} OCR worker threads.")

            # Auto-recovery: Re-queue pending OCR jobs from DB on startup
            try:
                from src.db.database import db_session
                with db_session(self.db_path) as conn:
                    rows = conn.execute(
                        "SELECT id, crop_image_path, vehicle_class_name FROM violations WHERE ocr_status = 'pending' ORDER BY id ASC LIMIT 50"
                    ).fetchall()
                    for r in rows:
                        vid = r["id"]
                        crop_p = r["crop_image_path"]
                        vclass = r["vehicle_class_name"] or "car"
                        if crop_p and Path(crop_p).is_file():
                            img = cv2.imread(crop_p)
                            if img is not None:
                                self.queue.put_nowait(
                                    OCRJob(
                                        violation_id=vid,
                                        crop_img=img,
                                        plate_save_path=str(Path(crop_p).with_name(f"{Path(crop_p).stem}_plate.jpg")),
                                        vehicle_class_name=vclass,
                                    )
                                )
                                logger.info(f"Re-queued pending OCR job for violation #{vid}")
            except Exception as e:
                logger.warning(f"Error during OCR pending recovery: {e}")

    def stop(self) -> None:
        with self.lock:
            self.running = False

        # Wait for workers to finish
        for th in self.workers:
            if th.is_alive():
                th.join(timeout=3.0)
        self.workers.clear()
        logger.info("OCR Worker Pool stopped.")

    def _worker_loop(self) -> None:
        # Lazy import and init RapidOCR inside worker
        from rapidocr_onnxruntime import RapidOCR

        try:
            # Check for local models in models/ocr/
            det_p = Path("models/ocr/ch_PP-OCRv4_det_infer.onnx")
            rec_p = Path("models/ocr/ch_PP-OCRv4_rec_infer.onnx")
            cls_p = Path("models/ocr/ch_ppocr_mobile_v2.0_cls_infer.onnx")

            kwargs = {}
            if det_p.is_file():
                kwargs["det_model_path"] = str(det_p)
            if rec_p.is_file():
                kwargs["rec_model_path"] = str(rec_p)
            if cls_p.is_file():
                kwargs["cls_model_path"] = str(cls_p)

            engine = RapidOCR(**kwargs)
            logger.info("RapidOCR initialized in worker thread.")
        except Exception as e:
            logger.error(f"Failed to initialize RapidOCR in worker: {e}")
            engine = None

        plate_regex = re.compile(self.config.vn_plate_regex)

        while self.running:
            try:
                job: OCRJob = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                if engine is None:
                    self.violation_repo.update_ocr(job.violation_id, "", 0.0, "failed")
                    self.queue.task_done()
                    continue

                crop = job.crop_img
                if crop is None or crop.size == 0:
                    self.violation_repo.update_ocr(job.violation_id, "", 0.0, "failed")
                    self.queue.task_done()
                    continue

                # Run OCR
                res, _ = engine(crop)
                if not res:
                    self.violation_repo.update_ocr(job.violation_id, "", 0.0, "failed")
                    self.queue.task_done()
                    continue

                # Parse detected text boxes: [[box_pts, text, score], ...]
                # Sort boxes primarily by vertical position (Y), then horizontal (X)
                # To group 2-line plates (line 1 top, line 2 bottom)
                boxes_with_text = []
                all_pts = []
                for item in res:
                    box_pts, text, score = item[0], str(item[1]), float(item[2])
                    y_center = (box_pts[0][1] + box_pts[2][1]) / 2.0
                    x_center = (box_pts[0][0] + box_pts[2][0]) / 2.0
                    boxes_with_text.append((y_center, x_center, box_pts, text, score))
                    all_pts.extend(box_pts)

                # Sort by Y center
                boxes_with_text.sort(key=lambda x: (x[0], x[1]))

                combined_texts = []
                total_conf = 0.0
                for _, _, _, text, score in boxes_with_text:
                    clean_part = clean_vietnamese_plate(text)
                    if clean_part:
                        combined_texts.append(clean_part)
                        total_conf += score

                avg_conf = (total_conf / len(boxes_with_text)) if boxes_with_text else 0.0
                joined_clean = "".join(combined_texts)

                # Crop bounding box of detected plate
                plate_saved = False
                if all_pts:
                    pts_np = np.array(all_pts, dtype=np.int32)
                    px1 = max(0, int(np.min(pts_np[:, 0]) - 5))
                    py1 = max(0, int(np.min(pts_np[:, 1]) - 5))
                    px2 = min(crop.shape[1], int(np.max(pts_np[:, 0]) + 5))
                    py2 = min(crop.shape[0], int(np.max(pts_np[:, 1]) + 5))

                    if px2 > px1 and py2 > py1:
                        plate_crop = crop[py1:py2, px1:px2]
                        Path(job.plate_save_path).parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(job.plate_save_path, plate_crop)
                        plate_saved = True

                # Validation & Formatting
                formatted_plate = format_vietnamese_plate(joined_clean, job.vehicle_class_name)
                is_valid = bool(plate_regex.match(joined_clean)) or (
                    len(joined_clean) in (7, 8, 9) and avg_conf >= self.config.min_ocr_conf
                )

                ocr_status = "done" if is_valid else ("failed" if not joined_clean else "done")
                plate_path_record = job.plate_save_path if plate_saved else None

                # Update SQLite
                self.violation_repo.update_ocr(
                    violation_id=job.violation_id,
                    plate_number=formatted_plate if joined_clean else "",
                    ocr_conf=round(avg_conf, 3),
                    ocr_status=ocr_status,
                    plate_image_path=plate_path_record,
                )

                # Dispatch event to dashboard listeners
                event_data = {
                    "type": "ocr_completed",
                    "violation_id": job.violation_id,
                    "plate_number": formatted_plate if joined_clean else "",
                    "ocr_conf": round(avg_conf, 3),
                    "ocr_status": ocr_status,
                    "plate_image_path": plate_path_record,
                }
                with self.lock:
                    listeners = list(self.listeners)
                for l in listeners:
                    try:
                        l(event_data)
                    except Exception:
                        pass

                logger.info(
                    f"OCR finished for violation {job.violation_id}: [{formatted_plate}] (conf: {avg_conf:.2f}, status: {ocr_status})"
                )

            except Exception as e:
                logger.error(f"Error processing OCR job for violation {job.violation_id}: {e}")
                self.violation_repo.update_ocr(job.violation_id, "", 0.0, "failed")
            finally:
                self.queue.task_done()
