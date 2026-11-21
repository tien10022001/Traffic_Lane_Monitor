from typing import List, Optional, Tuple
import numpy as np


def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """
    Check if a 2D point (x, y) is inside a polygon using Ray Casting algorithm.
    Coordinates can be normalized [0..1] or pixel coordinates, as long as point and polygon use the same scale.
    """
    if len(polygon) < 3:
        return False

    x, y = point
    inside = False
    n = len(polygon)

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        x_inters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= x_inters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> float:
    """Compute 2D cross product (B - A) x (C - A)."""
    return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])


def intersect_segments(
    A: Tuple[float, float],
    B: Tuple[float, float],
    C: Tuple[float, float],
    D: Tuple[float, float],
) -> bool:
    """Return True if line segment AB and line segment CD intersect."""
    ccw1 = ccw(A, B, C)
    ccw2 = ccw(A, B, D)
    ccw3 = ccw(C, D, A)
    ccw4 = ccw(C, D, B)

    return (
        ((ccw1 > 0 and ccw2 < 0) or (ccw1 < 0 and ccw2 > 0))
        and ((ccw3 > 0 and ccw4 < 0) or (ccw3 < 0 and ccw4 > 0))
    )


def check_line_crossing(
    p_prev: Tuple[float, float],
    p_curr: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
    direction: str = "both",
) -> bool:
    """
    Check if moving from p_prev to p_curr crosses the line from line_start to line_end.
    direction options: 'both', 'left_to_right', 'right_to_left'.
    'left_to_right' means crossing from the left side of vector (line_start -> line_end) to the right.
    """
    if not intersect_segments(p_prev, p_curr, line_start, line_end):
        return False

    if direction == "both":
        return True

    # Orientation of p_prev and p_curr relative to directed line (line_start -> line_end)
    # ccw > 0: point is to the left of the directed line
    # ccw < 0: point is to the right of the directed line
    sign_prev = ccw(line_start, line_end, p_prev)
    sign_curr = ccw(line_start, line_end, p_curr)

    if direction == "left_to_right":
        return sign_prev > 0 and sign_curr < 0
    elif direction == "right_to_left":
        return sign_prev < 0 and sign_curr > 0

    return True


def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Compute Intersection over Union (IoU) of two boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    denom = boxAArea + boxBArea - interArea
    if denom <= 0:
        return 0.0
    return interArea / denom


def compute_homography_matrix(
    src_landmarks: List[Tuple[float, float]],
    dst_landmarks: Optional[List[Tuple[float, float]]] = None,
) -> Optional[np.ndarray]:
    """
    Compute 3x3 Homography transformation matrix from 4 road landmark points.
    src_landmarks: 4 points [P1, P2, P3, P4] in camera normalized coordinates [0..1].
    dst_landmarks: Target rectified road section (default: unit square [0,0], [1,0], [1,1], [0,1]).
    """
    import cv2
    if len(src_landmarks) != 4:
        return None

    src = np.array(src_landmarks, dtype=np.float32)
    if dst_landmarks and len(dst_landmarks) == 4:
        dst = np.array(dst_landmarks, dtype=np.float32)
    else:
        # Default rectified bird's eye view
        dst = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)

    try:
        H = cv2.getPerspectiveTransform(src, dst)
        return H
    except Exception:
        return None


def project_point_homography(point: Tuple[float, float], H: np.ndarray) -> Tuple[float, float]:
    """Project 2D normalized point (x, y) through 3x3 Homography matrix."""
    vec = np.array([point[0], point[1], 1.0], dtype=np.float64)
    res = H @ vec
    if abs(res[2]) < 1e-9:
        return (float(res[0]), float(res[1]))
    return (float(res[0] / res[2]), float(res[1] / res[2]))


def get_ground_contact_point(
    bbox: Tuple[float, float, float, float],
    mode: str = "bottom_center",
) -> Tuple[float, float]:
    """
    Extract vehicle ground-contact point (where wheels touch the asphalt).
    bbox: [x1, y1, x2, y2]
    mode:
      - 'bottom_center': midpoint of bottom edge ((x1 + x2)/2, y2)
      - 'bottom_strip_20': center of the bottom 20% footprint strip ((x1 + x2)/2, y2 - 0.1*(y2 - y1))
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    h = max(1.0, y2 - y1)

    if mode == "bottom_strip_20":
        return (cx, y2 - 0.1 * h)
    return (cx, y2)


def is_vehicle_ground_in_polygon(
    bbox_norm: Tuple[float, float, float, float],
    polygon: List[Tuple[float, float]],
    mode: str = "bottom_center",
    H: Optional[np.ndarray] = None,
) -> bool:
    """
    Evaluate if vehicle ground-contact footprint intersects or lies inside a normalized polygon.
    bbox_norm: [cx, cy, w, h] normalized or [x1, y1, x2, y2] normalized.
    Supports Homography projection if matrix H is provided.
    """
    # Detect if bbox_norm is [cx, cy, w, h] or [x1, y1, x2, y2]
    val0, val1, val2, val3 = bbox_norm
    if val2 <= 1.0 and val3 <= 1.0 and (val0 + val2 / 2.0) <= 1.1:
        # [cx, cy, w, h] format
        cx, cy, w, h = val0, val1, val2, val3
        x1 = cx - w / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
    else:
        # [x1, y1, x2, y2] format
        x1, _, x2, y2 = val0, val1, val2, val3
        cx = (x1 + x2) / 2.0

    test_points = [(cx, y2)]
    if mode == "bottom_strip_20":
        # Multi-point footprint test along wheel base
        test_points.append((x1 + 0.2 * (x2 - x1), y2))
        test_points.append((x2 - 0.2 * (x2 - x1), y2))

    for pt in test_points:
        eval_pt = pt
        poly_to_check = polygon
        if H is not None:
            eval_pt = project_point_homography(pt, H)
            poly_to_check = [project_point_homography(p, H) for p in polygon]

        if point_in_polygon(eval_pt, poly_to_check):
            return True

    return False
