from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import json
from src.core.logger import mask_rtsp_url, logger
from src.db.database import db_session


class CameraRepository:
    def __init__(self, db_path: str = "data/traffic.db"):
        self.db_path = db_path

    def get_all(self, active_only: bool = False, mask_url: bool = False) -> List[Dict[str, Any]]:
        query = "SELECT * FROM cameras"
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY id ASC"
        with db_session(self.db_path) as conn:
            rows = conn.execute(query).fetchall()
            result = []
            for r in rows:
                item = dict(r)
                if mask_url:
                    item["source_url"] = mask_rtsp_url(item["source_url"])
                result.append(item)
            return result

    def get_by_id(self, camera_id: int, mask_url: bool = False) -> Optional[Dict[str, Any]]:
        with db_session(self.db_path) as conn:
            row = conn.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,)).fetchone()
            if not row:
                return None
            item = dict(row)
            if mask_url:
                item["source_url"] = mask_rtsp_url(item["source_url"])
            return item

    def create(
        self,
        name: str,
        source_url: str,
        source_type: str = "rtsp",
        is_active: int = 1,
        fps_limit: int = 15,
    ) -> int:
        with db_session(self.db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO cameras (name, source_url, source_type, is_active, fps_limit, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (name, source_url, source_type, is_active, fps_limit),
            )
            return cur.lastrowid

    def update(
        self,
        camera_id: int,
        name: Optional[str] = None,
        source_url: Optional[str] = None,
        source_type: Optional[str] = None,
        is_active: Optional[int] = None,
        fps_limit: Optional[int] = None,
    ) -> bool:
        fields = []
        values = []
        if name is not None:
            fields.append("name = ?")
            values.append(name)
        if source_url is not None:
            fields.append("source_url = ?")
            values.append(source_url)
        if source_type is not None:
            fields.append("source_type = ?")
            values.append(source_type)
        if is_active is not None:
            fields.append("is_active = ?")
            values.append(is_active)
        if fps_limit is not None:
            fields.append("fps_limit = ?")
            values.append(fps_limit)

        if not fields:
            return False

        fields.append("updated_at = CURRENT_TIMESTAMP")
        values.append(camera_id)
        sql = f"UPDATE cameras SET {', '.join(fields)} WHERE id = ?"
        with db_session(self.db_path) as conn:
            cur = conn.execute(sql, tuple(values))
            return cur.rowcount > 0

    def delete(self, camera_id: int) -> bool:
        with db_session(self.db_path) as conn:
            cur = conn.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))
            return cur.rowcount > 0


class RuleRepository:
    def __init__(self, db_path: str = "data/traffic.db"):
        self.db_path = db_path

    def get_all(
        self, camera_id: Optional[int] = None, active_only: bool = False
    ) -> List[Dict[str, Any]]:
        clauses = []
        params = []
        if camera_id is not None:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if active_only:
            clauses.append("is_active = 1")

        query = "SELECT * FROM rules"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id ASC"

        with db_session(self.db_path) as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            return [dict(r) for r in rows]

    def get_by_id(self, rule_id: int) -> Optional[Dict[str, Any]]:
        with db_session(self.db_path) as conn:
            row = conn.execute("SELECT * FROM rules WHERE id = ?", (rule_id,)).fetchone()
            return dict(row) if row else None

    def create(
        self,
        camera_id: int,
        name: str,
        rule_type: str,
        geometry_json: str,
        direction: str = "both",
        allowed_classes_json: Optional[str] = None,
        forbidden_classes_json: Optional[str] = None,
        dwell_threshold_sec: float = 10.0,
        is_active: int = 1,
    ) -> int:
        with db_session(self.db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO rules (
                    camera_id, name, rule_type, geometry_json, direction,
                    allowed_classes_json, forbidden_classes_json, dwell_threshold_sec,
                    is_active, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    camera_id,
                    name,
                    rule_type,
                    geometry_json,
                    direction,
                    allowed_classes_json,
                    forbidden_classes_json,
                    dwell_threshold_sec,
                    is_active,
                ),
            )
            return cur.lastrowid

    def update(
        self,
        rule_id: int,
        name: Optional[str] = None,
        rule_type: Optional[str] = None,
        geometry_json: Optional[str] = None,
        direction: Optional[str] = None,
        allowed_classes_json: Optional[str] = None,
        forbidden_classes_json: Optional[str] = None,
        dwell_threshold_sec: Optional[float] = None,
        is_active: Optional[int] = None,
    ) -> bool:
        fields = []
        values = []
        if name is not None:
            fields.append("name = ?")
            values.append(name)
        if rule_type is not None:
            fields.append("rule_type = ?")
            values.append(rule_type)
        if geometry_json is not None:
            fields.append("geometry_json = ?")
            values.append(geometry_json)
        if direction is not None:
            fields.append("direction = ?")
            values.append(direction)
        if allowed_classes_json is not None:
            fields.append("allowed_classes_json = ?")
            values.append(allowed_classes_json)
        if forbidden_classes_json is not None:
            fields.append("forbidden_classes_json = ?")
            values.append(forbidden_classes_json)
        if dwell_threshold_sec is not None:
            fields.append("dwell_threshold_sec = ?")
            values.append(dwell_threshold_sec)
        if is_active is not None:
            fields.append("is_active = ?")
            values.append(is_active)

        if not fields:
            return False

        fields.append("updated_at = CURRENT_TIMESTAMP")
        values.append(rule_id)
        sql = f"UPDATE rules SET {', '.join(fields)} WHERE id = ?"
        with db_session(self.db_path) as conn:
            cur = conn.execute(sql, tuple(values))
            return cur.rowcount > 0

    def delete(self, rule_id: int) -> bool:
        with db_session(self.db_path) as conn:
            cur = conn.execute("DELETE FROM rules WHERE id = ?", (rule_id,))
            return cur.rowcount > 0


class ViolationRepository:
    def __init__(self, db_path: str = "data/traffic.db"):
        self.db_path = db_path

    def create(
        self,
        camera_id: int,
        track_id: int,
        violation_type: str,
        vehicle_class_id: int,
        vehicle_class_name: str,
        detect_conf: float,
        bbox_norm_json: str,
        dwell_duration_sec: float,
        raw_image_path: str,
        annotated_image_path: str,
        crop_image_path: str,
        rule_id: Optional[int] = None,
        plate_number: Optional[str] = None,
        plate_image_path: Optional[str] = None,
        ocr_conf: float = 0.0,
        ocr_status: str = "pending",
        review_status: str = "pending",
        session_id: Optional[str] = None,
        session_status: str = "ACTIVE",
        repeat_count: int = 1,
    ) -> int:
        with db_session(self.db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO violations (
                    camera_id, rule_id, track_id, violation_type, vehicle_class_id,
                    vehicle_class_name, detect_conf, bbox_norm_json, dwell_duration_sec,
                    plate_number, ocr_conf, ocr_status, raw_image_path, annotated_image_path,
                    crop_image_path, plate_image_path, review_status, session_id,
                    session_status, repeat_count, last_seen_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (
                    camera_id,
                    rule_id,
                    track_id,
                    violation_type,
                    vehicle_class_id,
                    vehicle_class_name,
                    detect_conf,
                    bbox_norm_json,
                    dwell_duration_sec,
                    plate_number,
                    ocr_conf,
                    ocr_status,
                    raw_image_path,
                    annotated_image_path,
                    crop_image_path,
                    plate_image_path,
                    review_status,
                    session_id,
                    session_status,
                    repeat_count,
                ),
            )
            return cur.lastrowid

    def update_duration_and_last_seen(
        self,
        violation_id: int,
        dwell_duration_sec: float,
        repeat_count: int = 1,
    ) -> bool:
        sql = """
            UPDATE violations
            SET dwell_duration_sec = MAX(dwell_duration_sec, ?),
                repeat_count = ?,
                last_seen_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """
        with db_session(self.db_path) as conn:
            cur = conn.execute(sql, (dwell_duration_sec, repeat_count, violation_id))
            return cur.rowcount > 0

    def update_session_status(self, violation_id: int, session_status: str) -> bool:
        sql = "UPDATE violations SET session_status = ? WHERE id = ?"
        with db_session(self.db_path) as conn:
            cur = conn.execute(sql, (session_status, violation_id))
            return cur.rowcount > 0

    def get_by_id(self, violation_id: int) -> Optional[Dict[str, Any]]:
        with db_session(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM violations WHERE id = ?", (violation_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_all(
        self,
        page: int = 1,
        page_size: int = 50,
        camera_id: Optional[int] = None,
        violation_type: Optional[str] = None,
        vehicle_class_name: Optional[str] = None,
        review_status: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        clauses = []
        params: List[Any] = []

        if camera_id is not None:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if violation_type:
            clauses.append("violation_type = ?")
            params.append(violation_type)
        if vehicle_class_name:
            clauses.append("vehicle_class_name = ?")
            params.append(vehicle_class_name)
        if review_status:
            clauses.append("review_status = ?")
            params.append(review_status)
        if date_from:
            df = date_from.replace("T", " ").strip()
            if len(df) == 16:
                df += ":00"
            clauses.append("created_at >= ?")
            params.append(df)
        if date_to:
            dt = date_to.replace("T", " ").strip()
            if len(dt) == 16:
                dt += ":59"
            clauses.append("created_at <= ?")
            params.append(dt)

        where_clause = f" WHERE {' AND '.join(clauses)}" if clauses else ""

        with db_session(self.db_path) as conn:
            total_count = conn.execute(
                f"SELECT COUNT(*) FROM violations{where_clause}", tuple(params)
            ).fetchone()[0]

            offset = (max(page, 1) - 1) * page_size
            sql = f"""
                SELECT * FROM violations{where_clause}
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
            """
            rows = conn.execute(sql, tuple(params + [page_size, offset])).fetchall()
            return [dict(r) for r in rows], total_count

    def update_ocr(
        self,
        violation_id: int,
        plate_number: str,
        ocr_conf: float,
        ocr_status: str,
        plate_image_path: Optional[str] = None,
    ) -> bool:
        sql = """
            UPDATE violations
            SET plate_number = ?, ocr_conf = ?, ocr_status = ?,
                plate_image_path = COALESCE(?, plate_image_path)
            WHERE id = ?
        """
        with db_session(self.db_path) as conn:
            cur = conn.execute(sql, (plate_number, ocr_conf, ocr_status, plate_image_path, violation_id))
            return cur.rowcount > 0

    def update_review(
        self,
        violation_id: int,
        review_status: str,
        corrected_class_name: Optional[str] = None,
        corrected_plate: Optional[str] = None,
    ) -> bool:
        sql = """
            UPDATE violations
            SET review_status = ?,
                corrected_class_name = COALESCE(?, corrected_class_name),
                corrected_plate = COALESCE(?, corrected_plate)
            WHERE id = ?
        """
        with db_session(self.db_path) as conn:
            cur = conn.execute(
                sql, (review_status, corrected_class_name, corrected_plate, violation_id)
            )
            return cur.rowcount > 0

    def get_export_data(self, review_status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get violations for active learning feedback export."""
        query = "SELECT * FROM violations"
        params = []
        if review_status:
            query += " WHERE review_status = ?"
            params.append(review_status)
        query += " ORDER BY id ASC"

        with db_session(self.db_path) as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            return [dict(r) for r in rows]

    def delete_older_than(self, retention_days: int) -> int:
        """Delete violation records older than retention_days."""
        sql = """
            DELETE FROM violations
            WHERE created_at < datetime('now', '-' || ? || ' days')
        """
        with db_session(self.db_path) as conn:
            cur = conn.execute(sql, (retention_days,))
            return cur.rowcount
