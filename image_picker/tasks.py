"""One Alfred call. Configuration and session ownership belong to the caller."""
import json

from alfred import LLMServer

from .inputs import ImagePickerInput
from .instructions import build_instruction
from .outputs import build_output_model


async def run_image_picker(payload: dict, server: LLMServer, *, extra: dict | None = None,
                           max_tokens: int | None = None) -> dict:
    if server.retries != 0:
        raise ValueError("Image picking requires an Alfred server configured with retries=0")
    protected = {"messages", "response_format", "stream", "tools", "tool_choice", "n"}
    if protected & (server.default_extra.keys() | (extra or {}).keys()):
        raise ValueError("extra cannot override messages, output schema, or response mode")
    inp = ImagePickerInput.model_validate(payload)
    output = build_output_model(inp)
    budget = max_tokens if max_tokens is not None else 1024 + len(inp.images) * (200 + 40 * len(inp.assessment_definition.fields))
    if budget < 1:
        raise ValueError("max_tokens must be positive")
    context = inp.model_dump(exclude={"images", "assessment_instructions"})
    context["images"] = [{"image_id": image.image_id, "metadata": image.metadata} for image in inp.images]
    content = [{"type": "text", "text": json.dumps(context, ensure_ascii=False)}]
    for image in inp.images:
        blocks = server.build_content(f"Image ID: {image.image_id}", media=[image.source])
        if any(block["type"] not in {"text", "image_url"} for block in blocks):
            raise ValueError("Image picking accepts image media only")
        content.extend(blocks)
    envelope = await server.respond(
        [{"role": "system", "content": build_instruction(inp.assessment_instructions)}, {"role": "user", "content": content}],
        guide=output, max_tokens=budget, extra=extra, return_envelope=True, allow_redirects=False,
    )
    choice = envelope["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError(f"Incomplete model response: {choice.get('finish_reason')!r}")
    raw = choice["message"].get("content")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Model returned empty content")
    result = output.model_validate_json(raw)
    return {"product_id": inp.product_id, "result": result.model_dump(), "raw_response": envelope,
            "inference": {"model": envelope.get("model", server.model), "usage": envelope.get("usage", {})}}
