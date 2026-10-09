"""Alfred client setup, cleanup, and one-call image requests."""
from contextlib import asynccontextmanager
import json
import os

from alfred import LLMServer
from pydantic import BaseModel


def build_server(base_url: str, model: str, *, timeout: float, max_concurrent: int = 64) -> LLMServer:
    if not base_url or not model:
        raise ValueError("--vllm-url and --model are required (or set IMAGE_PICKER_LLM_URL/MODEL)")
    return LLMServer(
        base_url=base_url, model=model,
        api_key=os.getenv("IMAGE_PICKER_LLM_API_KEY") or None,
        timeout=timeout, max_concurrent=max_concurrent, retries=0,
    )


@asynccontextmanager
async def client_session(client: LLMServer):
    try:
        yield client
    finally:
        await client.close()


async def request_assessment(
    client: LLMServer, *, context: dict, images: list[tuple[str, str]],
    instruction: str, guide: type[BaseModel], max_tokens: int,
    extra: dict | None = None,
) -> dict:
    if client.retries != 0:
        raise ValueError("Image picking requires an Alfred server configured with retries=0")
    protected = {"messages", "response_format", "stream", "tools", "tool_choice", "n"}
    if protected & (client.default_extra.keys() | (extra or {}).keys()):
        raise ValueError("extra cannot override messages, output schema, or response mode")
    content = [{"type": "text", "text": json.dumps(context, ensure_ascii=False)}]
    for image_id, source in images:
        blocks = client.build_content(f"Image ID: {image_id}", media=[source])
        if any(block["type"] not in {"text", "image_url"} for block in blocks):
            raise ValueError("Image picking accepts image media only")
        content.extend(blocks)
    return await client.respond(
        [{"role": "system", "content": instruction}, {"role": "user", "content": content}],
        guide=guide, max_tokens=max_tokens, extra=extra,
        return_envelope=True, allow_redirects=False,
    )
