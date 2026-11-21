import asyncio
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from src.core.camera_manager import CameraManager
from src.core.config_loader import get_config
from src.core.logger import logger
from src.core.violation_service import ViolationService
from src.monitoring.resource_monitor import ResourceMonitor

router = APIRouter(tags=["Streaming & WebSockets"])

# Connection managers for WebSockets
_METRICS_CLIENTS: List[WebSocket] = []
_EVENTS_CLIENTS: List[WebSocket] = []
_WS_LOCK = asyncio.Lock()


async def broadcast_metric_update(snapshot: Dict[str, Any]) -> None:
    """Broadcast system metrics to all connected /ws/metrics clients."""
    async with _WS_LOCK:
        clients = list(_METRICS_CLIENTS)

    for ws in clients:
        try:
            await ws.send_json(snapshot)
        except Exception:
            async with _WS_LOCK:
                if ws in _METRICS_CLIENTS:
                    _METRICS_CLIENTS.remove(ws)


async def broadcast_event_update(event_data: Dict[str, Any]) -> None:
    """Broadcast violation or OCR event to all connected /ws/events clients."""
    async with _WS_LOCK:
        clients = list(_EVENTS_CLIENTS)

    for ws in clients:
        try:
            await ws.send_json(event_data)
        except Exception:
            async with _WS_LOCK:
                if ws in _EVENTS_CLIENTS:
                    _EVENTS_CLIENTS.remove(ws)


@router.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    """WebSocket streaming real-time hardware & pipeline telemetry."""
    await websocket.accept()
    async with _WS_LOCK:
        _METRICS_CLIENTS.append(websocket)

    # Immediately send the latest snapshot upon connection
    monitor = ResourceMonitor.get_instance()
    try:
        await websocket.send_json(monitor.get_latest_snapshot())
    except Exception:
        pass

    try:
        while True:
            # Keep connection open and receive any ping/messages from client
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        async with _WS_LOCK:
            if websocket in _METRICS_CLIENTS:
                _METRICS_CLIENTS.remove(websocket)


@router.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """WebSocket streaming real-time violation triggers and OCR results."""
    await websocket.accept()
    async with _WS_LOCK:
        _EVENTS_CLIENTS.append(websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        async with _WS_LOCK:
            if websocket in _EVENTS_CLIENTS:
                _EVENTS_CLIENTS.remove(websocket)


@router.get("/api/stream/{camera_id}")
async def mjpeg_stream(camera_id: int):
    """
    MJPEG live stream with real-time HUD annotations (bboxes, ROIs, dwell clocks).
    Yields multipart/x-mixed-replace frame chunks.
    """
    db_path = get_config().paths.db_path
    manager = CameraManager.get_instance(db_path)

    async def frame_generator():
        target_delay = 1.0 / 18.0  # Smooth 18 FPS transmission
        last_sent_bytes = None
        while True:
            t0 = asyncio.get_event_loop().time()
            jpeg_bytes = manager.get_latest_annotated_jpeg(camera_id)

            if jpeg_bytes is not None and jpeg_bytes is not last_sent_bytes:
                last_sent_bytes = jpeg_bytes
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                )
            elif jpeg_bytes is None:
                # Fallback snapshot or wait
                snap = manager.get_snapshot_jpeg(camera_id)
                if snap is not None and snap is not last_sent_bytes:
                    last_sent_bytes = snap
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + snap + b"\r\n"
                    )

            elapsed = asyncio.get_event_loop().time() - t0
            sleep_time = max(0.025, target_delay - elapsed)
            await asyncio.sleep(sleep_time)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )
