"""Local, human-in-the-loop image-picker review page."""
from pathlib import Path
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()
_PAGE = Path(__file__).with_name("review.html")


@router.get("/review", include_in_schema=False)
async def review_page():
    return FileResponse(_PAGE, media_type="text/html")


@router.get("/review/data", include_in_schema=False)
async def review_data():
    """Serve an explicitly configured local JSON/JSONL evaluation dataset."""
    path = os.getenv("IMAGE_PROCESSOR_REVIEW_FILE")
    if not path or not Path(path).is_file():
        raise HTTPException(status_code=404, detail="No review dataset is configured")
    media_type = "application/x-ndjson" if path.lower().endswith(".jsonl") else "application/json"
    return FileResponse(path, media_type=media_type)
