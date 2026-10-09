"""HTTP service for one-call product-image assessment and reference selection."""
import argparse
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from alfred import LLMServer
from image_picker import ImagePickerInput, run_image_picker


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8071
DEFAULT_MAX_CONCURRENT = 64


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    return int(value) if value is not None else default


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-url", default=_env("IMAGE_PICKER_LLM_URL"),
                        help="OpenAI-compatible LLM endpoint; defaults to IMAGE_PICKER_LLM_URL")
    parser.add_argument("--model", default=_env("IMAGE_PICKER_LLM_MODEL"),
                        help="Model name; defaults to IMAGE_PICKER_LLM_MODEL")
    parser.add_argument("--host", default=_env("IMAGE_PROCESSOR_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=_env_int("IMAGE_PROCESSOR_PORT", DEFAULT_PORT))
    parser.add_argument("--max-concurrent", type=int,
                        default=_env_int("IMAGE_PROCESSOR_MAX_CONCURRENT", DEFAULT_MAX_CONCURRENT),
                        help="Shared concurrent LLM request limit")
    parser.add_argument("--timeout", type=float,
                        default=float(_env("IMAGE_PROCESSOR_TIMEOUT", "200")))
    parser.add_argument("--disable-thinking", action="store_true",
                        help="Send Qwen/vLLM's enable_thinking=false request option")
    parser.add_argument("--reload", action="store_true", help="Reload on source changes")
    return parser.parse_args()


def build_server(base_url: str, model: str, *, timeout: float, max_concurrent: int) -> LLMServer:
    if not base_url or not model:
        raise ValueError("--vllm-url and --model are required (or set IMAGE_PICKER_LLM_URL/MODEL)")
    return LLMServer(
        base_url=base_url,
        model=model,
        api_key=_env("IMAGE_PICKER_LLM_API_KEY"),
        timeout=timeout,
        max_concurrent=max_concurrent,
        retries=0,
    )


def create_app(server: LLMServer, *, disable_thinking: bool = False) -> FastAPI:
    """Create the API around one shared Alfred client."""

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            await server.close()

    app = FastAPI(
        title="Image Processor",
        description="Assess product images and select evidence-backed reference images in one LLM call.",
        version="1.0.0",
        lifespan=lifespan,
    )
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if disable_thinking else None

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "model": server.model}

    @app.post("/v1/image-processor/pick")
    async def pick_image_references(payload: ImagePickerInput):
        try:
            result = await run_image_picker(payload.model_dump(), server, extra=extra)
        except Exception as error:
            raise HTTPException(status_code=502, detail=f"Image processor inference failed: {error}") from error
        result.pop("raw_response", None)
        return result

    return app


def main():
    args = parse_args()
    server = build_server(args.vllm_url, args.model, timeout=args.timeout,
                          max_concurrent=args.max_concurrent)
    app = create_app(server, disable_thinking=args.disable_thinking)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
