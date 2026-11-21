from pathlib import Path
import os
import re
import shutil
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, HTTPException, Response, UploadFile
from pydantic import BaseModel, Field

from src.core.camera_manager import CameraManager
from src.core.config_loader import get_config
from src.core.logger import logger
from src.db.repositories import CameraRepository

router = APIRouter(prefix="/api/cameras", tags=["Cameras"])


class CameraCreateRequest(BaseModel):
    name: str
    source_url: str
    source_type: str = "rtsp"  # 'rtsp' or 'video_file'
    is_active: int = 1
    fps_limit: int = 15


class CameraUpdateRequest(BaseModel):
    name: Optional[str] = None
    source_url: Optional[str] = None
    source_type: Optional[str] = None
    is_active: Optional[int] = None
    fps_limit: Optional[int] = None


@router.get("", response_model=List[Dict[str, Any]])
def list_cameras(active_only: bool = False):
    db_path = get_config().paths.db_path
    repo = CameraRepository(db_path)
    return repo.get_all(active_only=active_only, mask_url=True)


@router.post("", status_code=201)
def create_camera(payload: CameraCreateRequest):
    db_path = get_config().paths.db_path
    repo = CameraRepository(db_path)
    cam_id = repo.create(
        name=payload.name,
        source_url=payload.source_url,
        source_type=payload.source_type,
        is_active=payload.is_active,
        fps_limit=payload.fps_limit,
    )
    # Hot-reload in camera manager
    manager = CameraManager.get_instance(db_path)
    manager.reload_camera(cam_id)
    return {"id": cam_id, "message": "Camera created successfully"}


@router.get("/list-videos")
def list_available_videos() -> List[Dict[str, Any]]:
    """Scan and return list of video files available in project root, data/videos, and data/test_videos."""
    video_extensions = {".mp4", ".avi", ".mkv", ".mov"}
    search_dirs = [Path("."), Path("data/videos"), Path("data/test_videos"), Path("data")]
    results = []
    seen_paths = set()

    for d in search_dirs:
        if d.is_dir():
            for root, dirs, files in os.walk(d):
                # Never traverse into virtualenvs, git, node_modules, logs or brain
                dirs[:] = [
                    sub for sub in dirs
                    if not sub.startswith(".") and sub.lower() not in (
                        "venv", "env", ".venv", "node_modules", "brain", "__pycache__", "logs", "docs", "tests"
                    )
                ]
                # If checking root, only inspect root directory itself, don't recurse through all deep subdirs
                if d == Path(".") and root != ".":
                    continue

                for f in files:
                    ext = Path(f).suffix.lower()
                    if ext in video_extensions:
                        full_p = Path(root) / f
                        norm_p = str(full_p).replace("\\", "/")
                        if norm_p not in seen_paths:
                            seen_paths.add(norm_p)
                            try:
                                sz_mb = round(os.path.getsize(full_p) / (1024 * 1024), 2)
                            except OSError:
                                sz_mb = 0.0
                            results.append({
                                "name": f,
                                "path": norm_p,
                                "size_mb": sz_mb,
                            })

    return results


@router.post("/browse-file")
def browse_file_dialog() -> Dict[str, Any]:
    """
    Open native Windows file dialog to pick video file.
    Returns the exact absolute path on Windows.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        # Bring dialog to front
        root.attributes("-topmost", True)
        file_path = filedialog.askopenfilename(
            title="Chọn file video MP4 cho Camera",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mkv *.mov"),
                ("MP4 Videos", "*.mp4"),
                ("All Files", "*.*"),
            ],
        )
        root.destroy()

        if file_path:
            # Normalize slashes
            clean_path = str(file_path).replace("\\", "/")
            return {"file_path": clean_path, "status": "selected"}
        return {"file_path": "", "status": "cancelled"}
    except Exception as e:
        logger.warning(f"Native file dialog unavailable or session 0: {e}")
        return {"file_path": "", "status": "error", "error": str(e)}


@router.post("/upload-video")
async def upload_video(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload video file from browser into data/videos/ and return its path."""
    dest_dir = Path("data/videos")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename
    clean_name = re.sub(r"[^\w\-.]", "_", file.filename)
    if not clean_name.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
        clean_name += ".mp4"

    dest_path = dest_dir / clean_name

    # Write file to disk in chunks
    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_size_mb = round(dest_path.stat().st_size / (1024 * 1024), 2)
        rel_path = str(dest_path).replace("\\", "/")
        logger.info(f"Uploaded video file: {rel_path} ({file_size_mb} MB)")
        return {
            "file_path": rel_path,
            "filename": clean_name,
            "size_mb": file_size_mb,
            "message": "Upload successful",
        }
    except Exception as e:
        logger.error(f"Failed to upload video file: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/{camera_id}")
def get_camera(camera_id: int):
    db_path = get_config().paths.db_path
    repo = CameraRepository(db_path)
    cam = repo.get_by_id(camera_id, mask_url=True)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    return cam


@router.put("/{camera_id}")
def update_camera(camera_id: int, payload: CameraUpdateRequest):
    db_path = get_config().paths.db_path
    repo = CameraRepository(db_path)
    source_url = payload.source_url
    if source_url and ("***" in source_url):
        # Do not overwrite real source_url if masked password was sent back
        source_url = None

    updated = repo.update(
        camera_id=camera_id,
        name=payload.name,
        source_url=source_url,
        source_type=payload.source_type,
        is_active=payload.is_active,
        fps_limit=payload.fps_limit,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Camera not found or no changes made")

    # Hot-reload in camera manager
    manager = CameraManager.get_instance(db_path)
    manager.reload_camera(camera_id)
    return {"message": "Camera updated successfully"}


@router.delete("/{camera_id}")
def delete_camera(camera_id: int):
    db_path = get_config().paths.db_path
    repo = CameraRepository(db_path)
    manager = CameraManager.get_instance(db_path)
    manager.remove_camera(camera_id)

    deleted = repo.delete(camera_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Camera not found")
    return {"message": "Camera deleted successfully"}


@router.get("/{camera_id}/snapshot")
def get_camera_snapshot(camera_id: int):
    """Return a single JPEG snapshot of current camera frame for canvas ROI background."""
    db_path = get_config().paths.db_path
    manager = CameraManager.get_instance(db_path)
    jpeg_bytes = manager.get_snapshot_jpeg(camera_id)
    if not jpeg_bytes:
        raise HTTPException(status_code=503, detail="Camera snapshot currently unavailable")
    return Response(content=jpeg_bytes, media_type="image/jpeg")
