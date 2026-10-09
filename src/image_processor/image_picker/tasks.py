"""One Alfred call. Configuration and session ownership belong to the caller."""
from ..server.alfred import LLMServer, request_assessment

from .inputs import ImagePickerInput
from .instructions import build_instruction
from .outputs import build_output_model


async def run_image_picker(payload: dict, server: LLMServer, *, extra: dict | None = None,
                           max_tokens: int | None = None) -> dict:
    inp = ImagePickerInput.model_validate(payload)
    output = build_output_model(inp)
    budget = max_tokens if max_tokens is not None else 1024 + len(inp.images) * (200 + 40 * len(inp.assessment_definition.fields))
    if budget < 1:
        raise ValueError("max_tokens must be positive")
    context = inp.model_dump(exclude={"images", "assessment_instructions"})
    context["images"] = [{"image_id": image.image_id, "metadata": image.metadata} for image in inp.images]
    envelope = await request_assessment(
        server, context=context, images=[(image.image_id, image.source) for image in inp.images],
        instruction=build_instruction(inp.assessment_instructions),
        guide=output, max_tokens=budget, extra=extra,
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
