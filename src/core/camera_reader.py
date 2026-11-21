from pathlib import Path
import queue
import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from src.core.logger import logger, mask_rtsp_url


class CameraReader:
    """
    Dedicated background reader thread for a single RTSP stream or MP4 video file.
    Uses maxsize=1 queue to eliminate latency (drops stale frames).
    Supports exponential backoff reconnect for RTSP and FPS pacing with loop for MP4.
    """

    def __init__(
        self,
        camera_id: int,
        name: str,
        source_url: str,
        source_type: str = "rtsp",
        fps_limit: int = 15,
        loop_video: bool = True,
    ):
        self.camera_id = camera_id
        self.name = name
        self.source_url = source_url
        self.source_type = source_type.lower()
        self.fps_limit = fps_limit
        self.loop_video = loop_video

        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Telemetry
        self.status = "offline"  # 'online', 'reconnecting', 'offline'
        self.current_fps = 0.0
        self.dropped_frames = 0
        self.total_frames = 0
        self.last_frame: Optional[np.ndarray] = None
        self.frame_shape: Tuple[int, int] = (720, 1280)  # Default (h, w)
        self.lock = threading.Lock()

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(
            target=self._read_loop,
            name=f"CameraReader-{self.camera_id}",
            daemon=True,
        )
        self.thread.start()
        logger.info(f"Started CameraReader for [{self.name}] (ID: {self.camera_id})")

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        self.status = "offline"
        logger.info(f"Stopped CameraReader for [{self.name}] (ID: {self.camera_id})")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Return the most recent frame read by this camera."""
        with self.lock:
            if self.last_frame is not None:
                return self.last_frame.copy()
            return None

    def _read_loop(self) -> None:
        backoff_delays = [2.0, 5.0, 10.0, 30.0]
        backoff_idx = 0

        while self.running:
            masked_url = mask_rtsp_url(self.source_url)
            logger.info(f"Connecting to camera [{self.name}] at {masked_url}...")
            
            # Open video capture
            cap = cv2.VideoCapture(self.source_url)
            if not cap.isOpened():
                self.status = "reconnecting"
                delay = backoff_delays[min(backoff_idx, len(backoff_delays) - 1)]
                logger.warning(
                    f"Camera [{self.name}] failed to open. Retrying in {delay}s (attempt {backoff_idx + 1})..."
                )
                backoff_idx += 1
                time.sleep(delay)
                continue

            # Connected successfully
            self.status = "online"
            backoff_idx = 0
            logger.info(f"Camera [{self.name}] connected successfully.")

            # Get video properties
            native_fps = cap.get(cv2.CAP_PROP_FPS)
            target_fps = self.fps_limit if self.fps_limit > 0 else (native_fps if native_fps > 0 else 15.0)
            target_frame_time = 1.0 / max(1.0, target_fps)

            fps_calc_time = time.time()
            fps_frame_counter = 0

            while self.running and cap.isOpened():
                loop_start = time.perf_counter()

                ret, frame = cap.read()
                if not ret:
                    if self.source_type == "video_file" and self.loop_video:
                        # Loop video
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        logger.warning(f"Camera [{self.name}] stream ended or disconnected.")
                        break

                fps_frame_counter += 1
                self.total_frames += 1

                # Update frame shape & cache
                with self.lock:
                    self.last_frame = frame
                    self.frame_shape = frame.shape[:2]

                # Maxsize=1 queue: drop older frame to ensure 0-lag real-time AI
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                        self.dropped_frames += 1
                    except queue.Empty:
                        pass

                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    self.dropped_frames += 1

                # Calculate current FPS every 1 second
                now = time.time()
                if now - fps_calc_time >= 1.0:
                    self.current_fps = round(fps_frame_counter / (now - fps_calc_time), 1)
                    fps_frame_counter = 0
                    fps_calc_time = now

                # Pacing for video file test mode
                if self.source_type == "video_file":
                    elapsed = time.perf_counter() - loop_start
                    sleep_time = target_frame_time - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            cap.release()
            if self.running:
                self.status = "reconnecting"
                delay = backoff_delays[min(backoff_idx, len(backoff_delays) - 1)]
                backoff_idx += 1
                time.sleep(delay)

        self.status = "offline"
