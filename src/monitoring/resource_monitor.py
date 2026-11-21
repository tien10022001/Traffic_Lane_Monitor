from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional
import psutil

from src.core.config_loader import MonitoringConfig, get_config
from src.core.logger import logger

# Safe import of pynvml (only when GPU is enabled, skips entirely on CPU)
_HAS_NVML = False
try:
    if get_config().ai.device.lower() != "cpu":
        import pynvml
        pynvml.nvmlInit()
        _HAS_NVML = True
except Exception:
    _HAS_NVML = False


class ResourceMonitor:
    """
    Background system resource monitor collecting host, process, GPU, and pipeline metrics.
    Stores historical metrics in a thread-safe ring buffer.
    """

    _instance: Optional["ResourceMonitor"] = None
    _singleton_lock = threading.Lock()

    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or get_config().monitoring
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None

        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())
        # First call to cpu_percent initializes calculation
        self.process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)

        self.history = deque(maxlen=self.config.history_size)
        self.latest_snapshot: Dict[str, Any] = {}

        # Cache for evidence directory size (expensive to calculate every 2s)
        self.last_evidence_size_time = 0.0
        self.cached_evidence_size_mb = 0.0

        # Hooks to fetch dynamic pipeline telemetry from CameraManager and OCRWorker
        self.pipeline_provider: Optional[Callable[[], Dict[str, Any]]] = None
        self.listeners: List[Callable[[Dict[str, Any]], None]] = []

        # GPU initialization check
        self.nvml_initialized = _HAS_NVML
        self.gpu_handle = None
        if self.nvml_initialized:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                else:
                    self.nvml_initialized = False
            except Exception as e:
                logger.warning(f"Error accessing NVML device: {e}")
                self.nvml_initialized = False

    @classmethod
    def get_instance(cls) -> "ResourceMonitor":
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def register_pipeline_provider(self, provider: Callable[[], Dict[str, Any]]) -> None:
        """Register callback that returns live pipeline stats (camera statuses, FPS, queue)."""
        self.pipeline_provider = provider

    def add_listener(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        """Add WebSocket or external listener to receive metric snapshots."""
        with self.lock:
            if listener not in self.listeners:
                self.listeners.append(listener)

    def remove_listener(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        with self.lock:
            if listener in self.listeners:
                self.listeners.remove(listener)

    def start(self) -> None:
        """Start the background monitoring thread."""
        with self.lock:
            if self.running:
                return
            self.running = True
            self.thread = threading.Thread(target=self._monitor_loop, name="ResourceMonitorThread", daemon=True)
            self.thread.start()
            logger.info("ResourceMonitor started.")

    def stop(self) -> None:
        """Stop the background monitoring thread."""
        with self.lock:
            self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
            logger.info("ResourceMonitor stopped.")

    def _get_evidence_size_mb(self) -> float:
        now = time.time()
        if now - self.last_evidence_size_time > 60.0:  # Calculate once per minute
            self.last_evidence_size_time = now
            evidence_dir = Path(get_config().paths.evidence_dir)
            total_bytes = 0
            if evidence_dir.exists():
                try:
                    for root, _, files in os.walk(evidence_dir):
                        for f in files:
                            fp = os.path.join(root, f)
                            try:
                                total_bytes += os.path.getsize(fp)
                            except OSError:
                                pass
                except Exception as e:
                    logger.debug(f"Error calculating evidence directory size: {e}")
            self.cached_evidence_size_mb = round(total_bytes / (1024 * 1024), 2)
        return self.cached_evidence_size_mb

    def _collect_gpu_stats(self) -> Dict[str, Any]:
        """Collect NVIDIA GPU stats or safe fallback."""
        active_provider = "CPUExecutionProvider"
        from src.ai.detector import SharedDetector
        try:
            detector = SharedDetector._instance
            if detector:
                active_provider = detector.active_provider
        except Exception:
            pass

        if not self.nvml_initialized or not self.gpu_handle:
            return {
                "available": False,
                "name": "N/A (CPU / DirectML Fallback)",
                "provider": active_provider,
                "util_percent": 0.0,
                "vram_used_mb": 0,
                "vram_total_mb": 0,
                "vram_percent": 0.0,
                "temp_c": 0,
            }

        try:
            gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode("utf-8")
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)

            vram_used_mb = int(mem_info.used / (1024 * 1024))
            vram_total_mb = int(mem_info.total / (1024 * 1024))
            vram_percent = round((mem_info.used / mem_info.total) * 100.0, 1)

            return {
                "available": True,
                "name": gpu_name,
                "provider": active_provider,
                "util_percent": float(util.gpu),
                "vram_used_mb": vram_used_mb,
                "vram_total_mb": vram_total_mb,
                "vram_percent": vram_percent,
                "temp_c": int(temp),
            }
        except Exception as e:
            logger.debug(f"Error querying NVML stats: {e}")
            return {
                "available": False,
                "name": "Error querying GPU",
                "provider": active_provider,
                "util_percent": 0.0,
                "vram_used_mb": 0,
                "vram_total_mb": 0,
                "vram_percent": 0.0,
                "temp_c": 0,
            }

    def collect_snapshot(self) -> Dict[str, Any]:
        """Collect a full snapshot of system metrics."""
        now_iso = datetime.now().astimezone().isoformat()
        uptime_sec = int(time.time() - self.start_time)

        # Host metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        ram_used_gb = round((mem.total - mem.available) / (1024 ** 3), 2)
        ram_total_gb = round(mem.total / (1024 ** 3), 2)
        ram_percent = mem.percent

        data_path = Path("data").resolve()
        data_path.mkdir(parents=True, exist_ok=True)
        disk_usage = psutil.disk_usage(str(data_path))
        disk_free_gb = round(disk_usage.free / (1024 ** 3), 2)
        disk_percent = disk_usage.percent
        evidence_size_mb = self._get_evidence_size_mb()

        # Service process metrics
        try:
            svc_cpu = self.process.cpu_percent(interval=None)
            svc_mem_info = self.process.memory_info()
            svc_ram_mb = round(svc_mem_info.rss / (1024 * 1024), 1)
            svc_threads = self.process.num_threads()
        except Exception:
            svc_cpu, svc_ram_mb, svc_threads = 0.0, 0.0, 0

        # GPU metrics
        gpu_stats = self._collect_gpu_stats()

        # Pipeline stats
        if self.pipeline_provider:
            try:
                pipeline_stats = self.pipeline_provider()
            except Exception as e:
                logger.error(f"Error in pipeline provider callback: {e}")
                pipeline_stats = self._default_pipeline_stats()
        else:
            pipeline_stats = self._default_pipeline_stats()

        # Alerts calculation
        alerts: List[str] = []
        if cpu_percent >= self.config.warn_cpu_percent:
            alerts.append(f"High host CPU usage: {cpu_percent}% >= {self.config.warn_cpu_percent}%")
        if ram_percent >= self.config.warn_ram_percent:
            alerts.append(f"High host RAM usage: {ram_percent}% >= {self.config.warn_ram_percent}%")
        if gpu_stats.get("available") and gpu_stats.get("util_percent", 0) >= self.config.warn_gpu_percent:
            alerts.append(f"High GPU utilization: {gpu_stats['util_percent']}% >= {self.config.warn_gpu_percent}%")
        if disk_free_gb <= self.config.warn_disk_free_gb:
            alerts.append(f"Low free disk space: {disk_free_gb} GB <= {self.config.warn_disk_free_gb} GB")

        snapshot = {
            "timestamp": now_iso,
            "uptime_sec": uptime_sec,
            "host": {
                "cpu_percent": cpu_percent,
                "ram_used_gb": ram_used_gb,
                "ram_total_gb": ram_total_gb,
                "ram_percent": ram_percent,
                "disk_free_gb": disk_free_gb,
                "disk_percent": disk_percent,
                "evidence_dir_size_mb": evidence_size_mb,
            },
            "service_process": {
                "cpu_percent": svc_cpu,
                "ram_mb": svc_ram_mb,
                "threads": svc_threads,
            },
            "gpu": gpu_stats,
            "pipeline": pipeline_stats,
            "alerts": alerts,
        }

        with self.lock:
            self.latest_snapshot = snapshot
            self.history.append(snapshot)

        return snapshot

    def _default_pipeline_stats(self) -> Dict[str, Any]:
        return {
            "active_cameras": 0,
            "total_cameras": 0,
            "total_fps": 0.0,
            "ocr_queue_size": 0,
            "ocr_queue_maxsize": get_config().ocr.queue_maxsize,
            "cameras": [],
        }

    def get_latest_snapshot(self) -> Dict[str, Any]:
        with self.lock:
            if not self.latest_snapshot:
                return self.collect_snapshot()
            return self.latest_snapshot

    def get_history(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.history)

    def _monitor_loop(self) -> None:
        interval = self.config.interval_sec
        while self.running:
            try:
                snap = self.collect_snapshot()
                # Notify listeners
                with self.lock:
                    current_listeners = list(self.listeners)
                for listener in current_listeners:
                    try:
                        listener(snap)
                    except Exception as e:
                        logger.debug(f"Error in metric listener: {e}")
            except Exception as e:
                logger.error(f"Error in resource monitor loop: {e}")
            time.sleep(interval)
