"""Local, human-in-the-loop image-picker review page."""
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()
_PAGE = Path(__file__).with_name("review.html")


@router.get("/review", include_in_schema=False)
async def review_page():
    return FileResponse(_PAGE, media_type="text/html")
