"""Health endpoint; no model calls."""
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/healthz")
async def healthz(request: Request):
    return {"status": "ok", "model": request.app.state.server.model}
