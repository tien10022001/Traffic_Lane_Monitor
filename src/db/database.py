from contextlib import contextmanager
from pathlib import Path
import sqlite3
import threading
from typing import Generator
from src.core.logger import logger

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS cameras (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    source_url TEXT NOT NULL,
    source_type TEXT DEFAULT 'rtsp',
    is_active INTEGER DEFAULT 1,
    fps_limit INTEGER DEFAULT 15,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    rule_type TEXT NOT NULL,
    geometry_json TEXT NOT NULL,
    direction TEXT DEFAULT 'both',
    allowed_classes_json TEXT,
    forbidden_classes_json TEXT,
    dwell_threshold_sec REAL DEFAULT 10.0,
    is_active INTEGER DEFAULT 1,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id INTEGER NOT NULL,
    rule_id INTEGER,
    track_id INTEGER NOT NULL,
    violation_type TEXT NOT NULL,
    vehicle_class_id INTEGER NOT NULL,
    vehicle_class_name TEXT NOT NULL,
    detect_conf REAL NOT NULL,
    bbox_norm_json TEXT NOT NULL,
    dwell_duration_sec REAL DEFAULT 0.0,
    plate_number TEXT,
    ocr_conf REAL DEFAULT 0.0,
    ocr_status TEXT DEFAULT 'pending',
    raw_image_path TEXT NOT NULL,
    annotated_image_path TEXT NOT NULL,
    crop_image_path TEXT NOT NULL,
    plate_image_path TEXT,
    review_status TEXT DEFAULT 'pending',
    corrected_class_name TEXT,
    corrected_plate TEXT,
    session_id TEXT,
    session_status TEXT DEFAULT 'ACTIVE',
    last_seen_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    repeat_count INTEGER DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_violations_cam_time ON violations(camera_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_violations_review ON violations(review_status);
"""

_DB_LOCK = threading.Lock()


def get_db_connection(db_path: str = "data/traffic.db") -> sqlite3.Connection:
    """Create and return a configured SQLite connection with WAL mode."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn


@contextmanager
def db_session(db_path: str = "data/traffic.db") -> Generator[sqlite3.Connection, None, None]:
    """Context manager for thread-safe database operations with auto commit/rollback."""
    conn = get_db_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database transaction error: {e}")
        raise
    finally:
        conn.close()


def init_db(db_path: str = "data/traffic.db") -> None:
    """Initialize SQLite database with required tables, indices, and WAL mode."""
    with _DB_LOCK:
        logger.info(f"Initializing database at {db_path}...")
        with db_session(db_path) as conn:
            conn.executescript(SCHEMA_SQL)
            # Safe runtime migration for existing databases
            existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(violations)").fetchall()}
            if "session_id" not in existing_cols:
                conn.execute("ALTER TABLE violations ADD COLUMN session_id TEXT")
            if "session_status" not in existing_cols:
                conn.execute("ALTER TABLE violations ADD COLUMN session_status TEXT DEFAULT 'ACTIVE'")
            if "last_seen_at" not in existing_cols:
                conn.execute("ALTER TABLE violations ADD COLUMN last_seen_at DATETIME")
                conn.execute("UPDATE violations SET last_seen_at = created_at WHERE last_seen_at IS NULL")
            if "repeat_count" not in existing_cols:
                conn.execute("ALTER TABLE violations ADD COLUMN repeat_count INTEGER DEFAULT 1")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_violations_session ON violations(session_status, camera_id);")
        logger.info("Database schema initialized successfully.")


def wal_checkpoint(db_path: str = "data/traffic.db") -> None:
    """Perform WAL checkpoint TRUNCATE to reclaim disk space."""
    try:
        with db_session(db_path) as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        logger.info("SQLite WAL checkpoint (TRUNCATE) completed.")
    except Exception as e:
        logger.error(f"Error executing WAL checkpoint: {e}")
