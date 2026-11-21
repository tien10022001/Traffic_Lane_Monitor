from dataclasses import dataclass
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import onnxruntime as ort
from src.core.config_loader import AIConfig, get_config
from src.core.logger import logger


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    class_id: int
    class_name: str
    cx_norm: float
    cy_norm: float
    w_norm: float
    h_norm: float
    anchor: Tuple[float, float]        # (x, y) bottom-center in original pixels
    anchor_norm: Tuple[float, float]   # (x, y) bottom-center in [0..1]


def letterbox(
    im: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scaleFill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resize and pad image while meeting stride-multiple constraints."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # Compute padding
    ratio = r
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


class SharedDetector:
    """
    Thread-safe shared ONNX Runtime detector for YOLO11s.
    Loads a single model instance into VRAM/RAM and serves all camera threads.
    Supports flexible device selection (auto, gpu, cpu).
    """

    _instance: Optional["SharedDetector"] = None
    _singleton_lock = threading.Lock()

    def __init__(self, model_path: Optional[str] = None, config: Optional[AIConfig] = None):
        self.lock = threading.Lock()
        cfg = config or get_config().ai
        self.config = cfg

        path = model_path or get_config().paths.model_path
        if not Path(path).is_file():
            # Check alternative locations
            alt_path = Path("models/yolo/best.onnx")
            if alt_path.is_file():
                path = str(alt_path)
            else:
                raise FileNotFoundError(f"ONNX model not found at {path}")

        self.model_path = path
        self.session, self.active_provider = self._create_session(path, cfg)
        self.device_type = "gpu" if ("CUDA" in self.active_provider or "Dml" in self.active_provider) else "cpu"
        
        # Cache input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        logger.info(
            f"SharedDetector initialized with {self.model_path} on {self.active_provider} (device: {self.device_type})"
        )

    def _create_session(self, model_path: str, cfg: AIConfig) -> Tuple[ort.InferenceSession, str]:
        available = ort.get_available_providers()
        logger.info(f"Available ONNX Runtime execution providers: {available}")

        chosen_providers: List[str] = []
        target_device = (cfg.device or "auto").lower()

        if target_device == "cpu":
            chosen_providers = ["CPUExecutionProvider"]
        elif target_device == "gpu":
            # Prefer GPU providers
            for p in cfg.providers:
                if p in available and p != "CPUExecutionProvider":
                    chosen_providers.append(p)
            if not chosen_providers:
                logger.warning("GPU device requested but no GPU provider available. Falling back to CPU.")
                chosen_providers = ["CPUExecutionProvider"]
        else:  # auto
            for p in cfg.providers:
                if p in available:
                    chosen_providers.append(p)
            if "CPUExecutionProvider" not in chosen_providers:
                chosen_providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Safe thread allocation (prevents CPU lockup, thermal throttling and OS freezing)
        safe_threads = min(4, max(1, (os.cpu_count() or 4) // 2))
        sess_options.intra_op_num_threads = safe_threads
        sess_options.inter_op_num_threads = 1
        try:
            cv2.setNumThreads(safe_threads)
        except Exception:
            pass

        # Try initializing with requested providers, fallback if fails
        for prov in chosen_providers:
            try:
                session = ort.InferenceSession(model_path, sess_options=sess_options, providers=[prov])
                active = session.get_providers()[0]
                logger.info(f"Successfully loaded model with provider: {active}")
                return session, active
            except Exception as e:
                logger.warning(f"Failed to initialize session with provider {prov}: {e}")

        # Final fallback to CPU
        logger.warning("Falling back to CPUExecutionProvider...")
        session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        return session, "CPUExecutionProvider"

    @classmethod
    def get_instance(cls, model_path: Optional[str] = None) -> "SharedDetector":
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls(model_path)
            return cls._instance

    def update_thresholds(
        self,
        default_conf: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        per_class_conf: Optional[Dict[int, float]] = None,
    ) -> None:
        """Dynamically update detection thresholds with immediate effect on all cameras."""
        with self.lock:
            if default_conf is not None:
                self.config.default_conf = round(float(default_conf), 3)
            if iou_threshold is not None:
                self.config.iou_threshold = round(float(iou_threshold), 3)
            if per_class_conf is not None:
                for cid, val in per_class_conf.items():
                    self.config.per_class_conf[int(cid)] = round(float(val), 3)
        logger.info(
            f"Updated detector thresholds: default_conf={self.config.default_conf}, "
            f"iou={self.config.iou_threshold}, per_class={self.config.per_class_conf}"
        )

    def detect(self, frame: np.ndarray, conf_threshold: Optional[float] = None) -> List[Detection]:
        """
        Thread-safe inference on a single frame.
        Returns list of Detection objects.
        """
        if frame is None or frame.size == 0:
            return []

        orig_h, orig_w = frame.shape[:2]
        imgsz = self.config.imgsz

        # Letterbox preprocessing
        img, ratio, (dw, dh) = letterbox(frame, (imgsz, imgsz), auto=False)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        blob = np.expand_dims(img, axis=0)

        # Thread-safe inference
        with self.lock:
            outputs = self.session.run([self.output_name], {self.input_name: blob})

        # Parse YOLO output shape: [1, 12, 8400]
        preds = outputs[0]
        if preds.ndim == 3 and preds.shape[1] == 12:
            preds = np.transpose(preds[0], (1, 0))  # (8400, 12)
        elif preds.ndim == 3 and preds.shape[2] == 12:
            preds = preds[0]  # (8400, 12)
        else:
            logger.error(f"Unexpected prediction shape: {preds.shape}")
            return []

        boxes_cxcywh = preds[:, :4]
        class_scores = preds[:, 4:12]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        # Filter by per-class threshold
        default_conf = conf_threshold if conf_threshold is not None else self.config.default_conf
        per_class_conf = self.config.per_class_conf

        valid_indices = []
        for i, (cid, score) in enumerate(zip(class_ids, confidences)):
            thresh = per_class_conf.get(int(cid), default_conf)
            if score >= thresh:
                valid_indices.append(i)

        if not valid_indices:
            return []

        valid_indices = np.array(valid_indices)
        boxes_cxcywh = boxes_cxcywh[valid_indices]
        class_ids = class_ids[valid_indices]
        confidences = confidences[valid_indices]

        # Convert cx, cy, w, h in 640x640 to x1, y1, x2, y2 in original image
        cx = boxes_cxcywh[:, 0]
        cy = boxes_cxcywh[:, 1]
        w = boxes_cxcywh[:, 2]
        h = boxes_cxcywh[:, 3]

        x1_lb = cx - w / 2.0
        y1_lb = cy - h / 2.0
        x2_lb = cx + w / 2.0
        y2_lb = cy + h / 2.0

        x1 = np.clip((x1_lb - dw) / ratio, 0, orig_w)
        y1 = np.clip((y1_lb - dh) / ratio, 0, orig_h)
        x2 = np.clip((x2_lb - dw) / ratio, 0, orig_w)
        y2 = np.clip((y2_lb - dh) / ratio, 0, orig_h)

        boxes_xyxy = np.column_stack([x1, y1, x2, y2])

        # Non-Maximum Suppression (OpenCV NMSBoxes)
        # NMSBoxes expects boxes as [x, y, w, h] integers or floats
        nms_boxes = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes_xyxy]
        nms_scores = [float(s) for s in confidences]
        indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, score_threshold=0.0, nms_threshold=self.config.iou_threshold)

        if len(indices) == 0:
            return []

        # Flatten indices (cv2.dnn.NMSBoxes returns list or array)
        selected_indices = [int(i[0]) if isinstance(i, (list, tuple, np.ndarray)) else int(i) for i in indices]

        results: List[Detection] = []
        classes_map = self.config.classes

        for idx in selected_indices:
            bx1, by1, bx2, by2 = boxes_xyxy[idx]
            conf = float(confidences[idx])
            cid = int(class_ids[idx])
            cname = classes_map.get(cid, f"class_{cid}")

            # Normalized YOLO coordinates [cx, cy, w, h] in [0..1]
            bw = bx2 - bx1
            bh = by2 - by1
            cx_norm = float(((bx1 + bx2) / 2.0) / orig_w)
            cy_norm = float(((by1 + by2) / 2.0) / orig_h)
            w_norm = float(bw / orig_w)
            h_norm = float(bh / orig_h)

            # Bottom-center anchor
            anchor_x = float((bx1 + bx2) / 2.0)
            anchor_y = float(by2)
            anchor_x_norm = float(anchor_x / orig_w)
            anchor_y_norm = float(anchor_y / orig_h)

            results.append(
                Detection(
                    x1=float(bx1),
                    y1=float(by1),
                    x2=float(bx2),
                    y2=float(by2),
                    conf=conf,
                    class_id=cid,
                    class_name=cname,
                    cx_norm=cx_norm,
                    cy_norm=cy_norm,
                    w_norm=w_norm,
                    h_norm=h_norm,
                    anchor=(anchor_x, anchor_y),
                    anchor_norm=(anchor_x_norm, anchor_y_norm),
                )
            )

        return results
