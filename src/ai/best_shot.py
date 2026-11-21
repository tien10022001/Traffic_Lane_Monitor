from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import cv2
import numpy as np


@dataclass
class BestShotRecord:
    track_id: int
    score: float
    crop_img: np.ndarray
    conf: float
    bbox: Tuple[float, float, float, float]


class BestShotSelector:
    """
    Selects the highest quality vehicle crop per track_id using:
    Score = w1 * Area_Norm + w2 * Sharpness_Laplacian + w3 * Detect_Conf - w4 * Edge_Penalty
    """

    def __init__(
        self,
        w_area: float = 0.35,
        w_sharpness: float = 0.35,
        w_conf: float = 0.20,
        w_edge: float = 0.10,
    ):
        self.w_area = w_area
        self.w_sharpness = w_sharpness
        self.w_conf = w_conf
        self.w_edge = w_edge
        self.records: Dict[int, BestShotRecord] = {}

    def compute_score(
        self,
        crop: np.ndarray,
        bbox: Tuple[float, float, float, float],
        conf: float,
        frame_shape: Tuple[int, int],
    ) -> float:
        """Compute visual quality score for the vehicle crop."""
        if crop is None or crop.size == 0:
            return 0.0

        orig_h, orig_w = frame_shape
        x1, y1, x2, y2 = bbox
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        # 1. Normalized area (larger is better, capped at 0.5 frame size)
        area_norm = min(1.0, (bw * bh) / (0.35 * orig_w * orig_h))

        # 2. Sharpness using Laplacian variance (normalized ~ [0..1])
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            sharpness_norm = min(1.0, lap_var / 500.0)
        except Exception:
            sharpness_norm = 0.0

        # 3. Detection confidence
        conf_norm = float(np.clip(conf, 0.0, 1.0))

        # 4. Edge penalty: penalize if bounding box touches image boundary (cut-off vehicle)
        edge_penalty = 0.0
        margin = 5.0
        if x1 <= margin or y1 <= margin or x2 >= (orig_w - margin) or y2 >= (orig_h - margin):
            edge_penalty = 1.0

        score = (
            self.w_area * area_norm
            + self.w_sharpness * sharpness_norm
            + self.w_conf * conf_norm
            - self.w_edge * edge_penalty
        )
        return float(max(0.0, score))

    def update(
        self,
        track_id: int,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        conf: float,
    ) -> None:
        """Evaluate current frame crop for track_id and save if better than previous best."""
        orig_h, orig_w = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(orig_w, x2)
        y2 = min(orig_h, y2)

        if (x2 - x1) < 20 or (y2 - y1) < 20:
            return

        crop = frame[y1:y2, x1:x2].copy()
        score = self.compute_score(crop, (x1, y1, x2, y2), conf, (orig_h, orig_w))

        if track_id not in self.records or score > self.records[track_id].score:
            self.records[track_id] = BestShotRecord(
                track_id=track_id,
                score=score,
                crop_img=crop,
                conf=conf,
                bbox=(x1, y1, x2, y2),
            )

    def get_best_shot(self, track_id: int) -> Optional[np.ndarray]:
        record = self.records.get(track_id)
        if record is not None and record.crop_img is not None:
            return record.crop_img.copy()
        return None

    def prune(self, active_track_ids: set) -> None:
        stale_ids = [tid for tid in self.records if tid not in active_track_ids]
        for tid in stale_ids:
            del self.records[tid]
