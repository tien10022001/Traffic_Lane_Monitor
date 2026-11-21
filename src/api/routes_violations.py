from io import BytesIO
from pathlib import Path
import json
from typing import Any, Dict, List, Optional
import zipfile
from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from src.core.config_loader import get_config
from src.db.repositories import ViolationRepository

router = APIRouter(prefix="/api/violations", tags=["Violations & Active Learning"])


class ViolationReviewRequest(BaseModel):
    review_status: str  # 'pending', 'confirmed', 'false_positive'
    corrected_class_name: Optional[str] = None
    corrected_plate: Optional[str] = None


@router.get("")
def list_violations(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    camera_id: Optional[int] = None,
    violation_type: Optional[str] = None,
    vehicle_class_name: Optional[str] = None,
    review_status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
):
    db_path = get_config().paths.db_path
    repo = ViolationRepository(db_path)
    items, total = repo.get_all(
        page=page,
        page_size=page_size,
        camera_id=camera_id,
        violation_type=violation_type,
        vehicle_class_name=vehicle_class_name,
        review_status=review_status,
        date_from=date_from,
        date_to=date_to,
    )
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size if total > 0 else 1,
    }


@router.get("/{violation_id}")
def get_violation(violation_id: int):
    db_path = get_config().paths.db_path
    repo = ViolationRepository(db_path)
    item = repo.get_by_id(violation_id)
    if not item:
        raise HTTPException(status_code=404, detail="Violation not found")
    return item


@router.patch("/{violation_id}/review")
def review_violation(violation_id: int, payload: ViolationReviewRequest):
    db_path = get_config().paths.db_path
    repo = ViolationRepository(db_path)
    updated = repo.update_review(
        violation_id=violation_id,
        review_status=payload.review_status,
        corrected_class_name=payload.corrected_class_name,
        corrected_plate=payload.corrected_plate,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Violation not found")

    # Layer 3: Update deduplication manager (assigns false_positive_cooldown_sec if marked false_positive)
    try:
        from src.core.deduplication import DeduplicationManager
        DeduplicationManager.get_instance(db_path).mark_review_status(violation_id, payload.review_status)
    except Exception:
        pass

    return {"message": "Violation review updated successfully"}


@router.get("/{violation_id}/image/{img_type}")
def get_violation_image(violation_id: int, img_type: str):
    """
    Serve raw, annotated, crop, or plate image for a violation.
    img_type: 'raw', 'annotated', 'crop', 'plate'
    """
    db_path = get_config().paths.db_path
    repo = ViolationRepository(db_path)
    item = repo.get_by_id(violation_id)
    if not item:
        raise HTTPException(status_code=404, detail="Violation not found")

    path_map = {
        "raw": item.get("raw_image_path"),
        "annotated": item.get("annotated_image_path"),
        "crop": item.get("crop_image_path"),
        "plate": item.get("plate_image_path"),
    }

    file_path = path_map.get(img_type.lower())
    if not file_path or not Path(file_path).is_file():
        raise HTTPException(status_code=404, detail=f"Image type '{img_type}' not available")

    return FileResponse(file_path, media_type="image/jpeg")


@router.get("/export-feedback/download")
def export_feedback(review_status: Optional[str] = "confirmed"):
    """
    Export Active Learning ZIP dataset:
    Includes clean raw images + YOLO format .txt annotations + classes.txt.
    """
    db_path = get_config().paths.db_path
    repo = ViolationRepository(db_path)
    violations = repo.get_export_data(review_status=review_status)

    classes_map = get_config().ai.classes
    # Invert class map for name -> id lookup
    name_to_id = {name.lower(): cid for cid, name in classes_map.items()}

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Write classes.txt
        classes_txt = "\n".join([classes_map[i] for i in sorted(classes_map.keys())])
        zf.writestr("classes.txt", classes_txt)

        count = 0
        for v in violations:
            raw_path = v.get("raw_image_path")
            if not raw_path or not Path(raw_path).is_file():
                continue

            # Determine final class id
            cid = v["vehicle_class_id"]
            if v.get("corrected_class_name"):
                corr_name = v["corrected_class_name"].lower()
                if corr_name in name_to_id:
                    cid = name_to_id[corr_name]

            # Parse bounding box
            try:
                cx, cy, w, h = json.loads(v["bbox_norm_json"])
            except Exception:
                continue

            # YOLO annotation line: <class_id> <cx> <cy> <w> <h>
            label_line = f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"

            base_name = f"sample_{v['id']:06d}"
            # Add image to zip
            zf.write(raw_path, arcname=f"images/{base_name}.jpg")
            # Add label to zip
            zf.writestr(f"labels/{base_name}.txt", label_line)
            count += 1

    zip_buffer.seek(0)
    filename = f"active_learning_dataset_{count}_samples.zip"

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/reset-feed")
def reset_violation_feed() -> Dict[str, Any]:
    """Reset active deduplication sessions and cache so new violations can trigger cleanly."""
    from src.core.deduplication import DeduplicationManager
    db_path = get_config().paths.db_path
    dedup = DeduplicationManager.get_instance(db_path)
    dedup.clear_cache()
    return {"message": "Violation feed and active deduplication cache reset successfully"}
