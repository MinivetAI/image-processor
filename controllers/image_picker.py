"""HTTP adapter for the existing image-picker contract."""
from fastapi import APIRouter, HTTPException, Request

from image_picker import ImagePickerInput, run_image_picker

router = APIRouter()


@router.post("/v1/image-processor/pick")
async def pick_image_references(payload: ImagePickerInput, request: Request):
    try:
        result = await run_image_picker(
            payload.model_dump(), request.app.state.server, extra=request.app.state.extra,
        )
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Image processor inference failed: {error}") from error
    result.pop("raw_response", None)
    return result
