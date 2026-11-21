from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.ai.detector import Detection
from src.rules.geometry import compute_iou


@dataclass
class TrackedVehicle:
    track_id: int
    bbox: Tuple[float, float, float, float]  # [x1, y1, x2, y2] in pixels
    conf: float
    class_id: int
    class_name: str
    cx_norm: float
    cy_norm: float
    w_norm: float
    h_norm: float
    anchor: Tuple[float, float]             # (x, y) bottom-center in pixels
    anchor_norm: Tuple[float, float]        # (x, y) in [0..1]
    history: List[Tuple[float, float]]       # list of past anchors for trajectory


class KalmanBoxTracker:
    """Simple 2D bounding box Kalman filter tracker."""
    count = 0

    def __init__(self, bbox: Tuple[float, float, float, float], class_id: int, conf: float):
        # State: [cx, cy, w, h, vx, vy, vw, vh]
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0

        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # Covariance
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1000.0, 1000.0, 1000.0, 1000.0]).astype(np.float32)
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0]).astype(np.float32)
        self.R = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float32)

        self.time_since_update = 0
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.history: List[Tuple[float, float]] = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.class_id = class_id
        self.conf = conf

    def update(self, bbox: Tuple[float, float, float, float], class_id: int, conf: float) -> None:
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.class_id = class_id
        self.conf = conf

        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        z = np.array([cx, cy, w, h], dtype=np.float32)

        # Measurement update
        H = np.zeros((4, 8), dtype=np.float32)
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0

        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(8, dtype=np.float32)
        self.P = (I - K @ H) @ self.P

        # Record anchor
        curr_box = self.get_state()
        anchor = ((curr_box[0] + curr_box[2]) / 2.0, curr_box[3])
        self.history.append(anchor)
        if len(self.history) > 60:
            self.history.pop(0)

    def predict(self) -> Tuple[float, float, float, float]:
        # State transition: x' = F * x
        F = np.eye(8, dtype=np.float32)
        F[0, 4] = F[1, 5] = F[2, 6] = F[3, 7] = 1.0

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.get_state()

    def get_state(self) -> Tuple[float, float, float, float]:
        cx, cy, w, h = self.x[:4]
        w = max(1.0, w)
        h = max(1.0, h)
        return (float(cx - w / 2.0), float(cy - h / 2.0), float(cx + w / 2.0), float(cy + h / 2.0))


def iou_batch(boxesA: List[Tuple[float, float, float, float]], boxesB: List[Tuple[float, float, float, float]]) -> np.ndarray:
    """Compute IoU matrix between list A and list B of boxes [x1, y1, x2, y2]."""
    if not boxesA or not boxesB:
        return np.empty((len(boxesA), len(boxesB)), dtype=np.float32)

    A = np.array(boxesA, dtype=np.float32)
    B = np.array(boxesB, dtype=np.float32)

    min_x = np.maximum(A[:, 0, None], B[None, :, 0])
    min_y = np.maximum(A[:, 1, None], B[None, :, 1])
    max_x = np.minimum(A[:, 2, None], B[None, :, 2])
    max_y = np.minimum(A[:, 3, None], B[None, :, 3])

    inter_w = np.maximum(0.0, max_x - min_x)
    inter_h = np.maximum(0.0, max_y - min_y)
    intersection = inter_w * inter_h

    area_A = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    area_B = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    union = area_A[:, None] + area_B[None, :] - intersection

    return np.where(union > 0, intersection / union, 0.0)


class ByteTracker:
    """
    ByteTrack algorithm implementation per camera.
    Maintains independent track IDs, track buffer, and handles occlusions.
    """

    def __init__(self, track_buffer: int = 75, match_thresh: float = 0.5):
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(
        self,
        detections: List[Detection],
        frame_shape: Tuple[int, int],
        class_map: Optional[Dict[int, str]] = None,
    ) -> List[TrackedVehicle]:
        """
        Update tracker with new detections and return current active TrackedVehicles.
        frame_shape: (height, width)
        """
        self.frame_count += 1
        orig_h, orig_w = frame_shape

        # Step 1: Predict new locations of existing trackers
        for t in self.trackers:
            t.predict()

        # Step 2: Separate high confidence and low confidence detections
        # Note: All detections in `detections` have already met SharedDetector's per-class threshold.
        # Threshold at 0.20 allows small/distant vehicles (like motorcycles at 0.25-0.35) to spawn and maintain tracks.
        high_thresh = 0.20
        high_dets = [d for d in detections if d.conf >= high_thresh]
        low_dets = [d for d in detections if d.conf < high_thresh]

        trks_boxes = [t.get_state() for t in self.trackers]

        # First association: High confidence detections with existing tracks
        matched, unmatched_dets, unmatched_trks = self._associate(
            high_dets, self.trackers, self.match_thresh
        )

        for d_idx, t_idx in matched:
            d = high_dets[d_idx]
            self.trackers[t_idx].update((d.x1, d.y1, d.x2, d.y2), d.class_id, d.conf)

        # Second association: Low confidence detections with remaining tracks
        remaining_trks = [self.trackers[i] for i in unmatched_trks]
        matched_low, _, unmatched_trks_2 = self._associate(
            low_dets, remaining_trks, match_thresh=0.4
        )

        for d_idx, r_idx in matched_low:
            d = low_dets[d_idx]
            remaining_trks[r_idx].update((d.x1, d.y1, d.x2, d.y2), d.class_id, d.conf)

        # Create new tracks for unmatched high confidence detections
        for d_idx in unmatched_dets:
            d = high_dets[d_idx]
            new_tracker = KalmanBoxTracker((d.x1, d.y1, d.x2, d.y2), d.class_id, d.conf)
            self.trackers.append(new_tracker)

        # Remove dead tracks older than track_buffer
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.track_buffer]

        # Return active tracks visible in current frame
        active_tracked: List[TrackedVehicle] = []
        for t in self.trackers:
            if t.time_since_update < 1:  # Updated in current frame
                box = t.get_state()
                bx1, by1, bx2, by2 = box
                bw = max(1.0, bx2 - bx1)
                bh = max(1.0, by2 - by1)

                cx_norm = float(((bx1 + bx2) / 2.0) / orig_w)
                cy_norm = float(((by1 + by2) / 2.0) / orig_h)
                w_norm = float(bw / orig_w)
                h_norm = float(bh / orig_h)

                anchor_x = float((bx1 + bx2) / 2.0)
                anchor_y = float(by2)
                anchor_x_norm = float(anchor_x / orig_w)
                anchor_y_norm = float(anchor_y / orig_h)

                cname = class_map.get(t.class_id, f"class_{t.class_id}") if class_map else f"class_{t.class_id}"

                active_tracked.append(
                    TrackedVehicle(
                        track_id=t.id,
                        bbox=(bx1, by1, bx2, by2),
                        conf=float(t.conf),
                        class_id=int(t.class_id),
                        class_name=cname,
                        cx_norm=cx_norm,
                        cy_norm=cy_norm,
                        w_norm=w_norm,
                        h_norm=h_norm,
                        anchor=(anchor_x, anchor_y),
                        anchor_norm=(anchor_x_norm, anchor_y_norm),
                        history=list(t.history),
                    )
                )

        return active_tracked

    def _associate(
        self,
        dets: List[Detection],
        trks: List[KalmanBoxTracker],
        match_thresh: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not dets or not trks:
            return [], list(range(len(dets))), list(range(len(trks)))

        det_boxes = [(d.x1, d.y1, d.x2, d.y2) for d in dets]
        trk_boxes = [t.get_state() for t in trks]

        iou_matrix = iou_batch(det_boxes, trk_boxes)
        cost_matrix = 1.0 - iou_matrix

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched: List[Tuple[int, int]] = []
        unmatched_dets = set(range(len(dets)))
        unmatched_trks = set(range(len(trks)))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= match_thresh:
                matched.append((r, c))
                unmatched_dets.discard(r)
                unmatched_trks.discard(c)

        return matched, list(unmatched_dets), list(unmatched_trks)
