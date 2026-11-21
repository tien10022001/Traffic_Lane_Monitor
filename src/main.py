import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
import os
import shutil
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.ai.detector import SharedDetector
from src.ai.ocr_worker import AsyncOCRWorkerPool
from src.api.routes_cameras import router as cameras_router
from src.api.routes_rules import router as rules_router
from src.api.routes_stream import (
    broadcast_event_update,
    broadcast_metric_update,
    router as stream_router,
)
from src.api.routes_system import router as system_router
from src.api.routes_violations import router as violations_router
from src.core.camera_manager import CameraManager
from src.core.config_loader import get_config
from src.core.logger import logger
from src.core.violation_service import ViolationService
from src.db.database import init_db, wal_checkpoint
from src.db.repositories import ViolationRepository
from src.monitoring.resource_monitor import ResourceMonitor


async def retention_worker_loop():
    """Daily retention worker running at configured hour (e.g. 02:00) to purge old evidence."""
    cfg = get_config().retention
    if not cfg.enabled:
        return

    logger.info(f"Retention scheduler initialized (runs at {cfg.run_at_hour:02d}:00 daily, keeps {cfg.retention_days} days).")
    while True:
        try:
            now = datetime.now()
            # Calculate next run time
            target = now.replace(hour=cfg.run_at_hour, minute=0, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            wait_seconds = (target - now).total_seconds()
            await asyncio.sleep(wait_seconds)

            logger.info("Executing scheduled retention cleanup...")
            # 1. Purge database records older than retention_days
            db_path = get_config().paths.db_path
            repo = ViolationRepository(db_path)
            deleted_rows = repo.delete_older_than(cfg.retention_days)
            logger.info(f"Retention: purged {deleted_rows} old violation records from database.")

            # 2. Purge old evidence folders
            evidence_dir = Path(get_config().paths.evidence_dir)
            cutoff_date = (datetime.now() - timedelta(days=cfg.retention_days)).strftime("%Y-%m-%d")
            if evidence_dir.exists():
                for date_folder in evidence_dir.iterdir():
                    if date_folder.is_dir() and date_folder.name < cutoff_date:
                        try:
                            shutil.rmtree(date_folder)
                            logger.info(f"Retention: deleted expired evidence directory {date_folder.name}")
                        except Exception as e:
                            logger.error(f"Error removing old evidence folder {date_folder}: {e}")

            # 3. WAL Checkpoint TRUNCATE
            wal_checkpoint(db_path)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in retention worker loop: {e}")
            await asyncio.sleep(60.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    logger.info("=====================================================")
    logger.info("Starting CCTV Traffic AI Service...")
    logger.info("=====================================================")

    cfg = get_config()
    loop = asyncio.get_running_loop()

    # 1. Initialize SQLite database
    init_db(cfg.paths.db_path)

    # 2. Pre-warm SharedDetector
    try:
        SharedDetector.get_instance()
    except Exception as e:
        logger.error(f"Failed to load AI Detector: {e}")

    # 3. Initialize & Start OCR Worker Pool
    ocr_pool = AsyncOCRWorkerPool.get_instance(cfg.paths.db_path)
    ocr_pool.start()

    # 4. Initialize ViolationService
    violation_service = ViolationService.get_instance(cfg.paths.db_path)

    # 5. Initialize CameraManager & hook violation callback
    camera_manager = CameraManager.get_instance(cfg.paths.db_path)
    camera_manager.set_violation_callback(violation_service.process_violation)

    # 6. Initialize ResourceMonitor & hook pipeline stats
    resource_monitor = ResourceMonitor.get_instance()
    resource_monitor.register_pipeline_provider(camera_manager.get_pipeline_stats)

    # 7. Bridge event listeners to WebSocket broadcasters
    def on_resource_snapshot(snap):
        asyncio.run_coroutine_threadsafe(broadcast_metric_update(snap), loop)

    def on_violation_or_ocr_event(event):
        asyncio.run_coroutine_threadsafe(broadcast_event_update(event), loop)

    resource_monitor.add_listener(on_resource_snapshot)
    resource_monitor.start()

    violation_service.add_event_listener(on_violation_or_ocr_event)
    ocr_pool.add_event_listener(on_violation_or_ocr_event)

    # 8. Start all cameras
    camera_manager.start_all()

    # 9. Start Retention Background Task
    retention_task = asyncio.create_task(retention_worker_loop())

    logger.info("CCTV Traffic AI Service is READY and accepting requests.")
    yield

    # --- SHUTDOWN (Graceful Shutdown) ---
    logger.info("Initiating graceful shutdown of CCTV Traffic AI Service...")

    retention_task.cancel()
    camera_manager.stop_all()
    ocr_pool.stop()
    resource_monitor.stop()

    wal_checkpoint(cfg.paths.db_path)
    logger.info("Graceful shutdown completed successfully. Process exiting.")


def create_app() -> FastAPI:
    cfg = get_config()
    app = FastAPI(
        title="CCTV Traffic AI Service",
        description="Multi-Camera Real-time Traffic Surveillance & Active Learning Service",
        version="2.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API Routers
    app.include_router(system_router)
    app.include_router(cameras_router)
    app.include_router(rules_router)
    app.include_router(violations_router)
    app.include_router(stream_router)

    # Serve Web Dashboard
    web_dir = Path("web").resolve()
    if web_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

        @app.get("/", include_in_schema=False)
        def serve_index():
            index_path = web_dir / "index.html"
            if index_path.is_file():
                return FileResponse(
                    str(index_path),
                    headers={
                        "Cache-Control": "no-cache, no-store, must-revalidate",
                        "Pragma": "no-cache",
                        "Expires": "0",
                    },
                )
            return {"message": "CCTV Traffic AI Service Running. Web dashboard index.html not found."}

    return app


app = create_app()
