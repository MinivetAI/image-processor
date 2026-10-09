"""HTTP adapter for the existing image-picker contract."""
import json
import logging
import os
from pathlib import Path
import re
import tempfile
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request

from ..image_picker import ImagePickerInput, run_image_picker

router = APIRouter()


def _save_attempt_diagnostics(product_id: str, attempts: list[dict]) -> None:
    """Keep rejected model outputs in a local, non-repository diagnostics directory."""
    if not attempts:
        return
    directory = Path(os.getenv(
        "IMAGE_PROCESSOR_DIAGNOSTICS_DIR",
        str(Path(tempfile.gettempdir()) / "image-processor-diagnostics"),
    ))
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", product_id)[:80] or "product"
    path = directory / f"{safe_id}-{uuid4().hex}.json"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"product_id": product_id, "attempts": attempts}, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    except OSError:
        logging.exception("Unable to save image-picker model diagnostics")


@router.post("/v1/image-processor/pick")
async def pick_image_references(payload: ImagePickerInput, request: Request):
    try:
        result = await run_image_picker(
            payload.model_dump(), request.app.state.server, extra=request.app.state.extra,
        )
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Image processor inference failed: {error}") from error
    _save_attempt_diagnostics(result["product_id"], result.pop("attempt_diagnostics", []))
    result.pop("raw_response", None)
    return result
