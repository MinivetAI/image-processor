"""One Alfred call. Configuration and session ownership belong to the caller."""
from ..server.alfred import LLMServer, request_assessment

from .inputs import ImagePickerInput
from .instructions import build_instruction
from .media import prepare_images
from .outputs import build_output_model, to_public_result


class ModelResponseError(ValueError):
    """A failed model response plus its original envelope for local diagnosis."""

    def __init__(self, message, envelope, content=None):
        super().__init__(message)
        self.envelope = envelope
        self.content = content


async def run_image_picker(payload: dict, server: LLMServer, *, extra: dict | None = None,
                           max_tokens: int | None = None) -> dict:
    inp = ImagePickerInput.model_validate(payload)
    output = build_output_model(inp)
    budget = max_tokens if max_tokens is not None else 1024 + len(inp.images) * (200 + 40 * len(inp.assessment_definition.fields))
    if budget < 1:
        raise ValueError("max_tokens must be positive")
    compressed_images = await prepare_images([image.source for image in inp.images])
    context = inp.model_dump(exclude={"product_id", "images", "assessment_instructions"})
    context["images"] = [{"image_number": number} for number in range(1, len(inp.images) + 1)]
    envelope = await request_assessment(
        server, context=context, images=[(f"Image {number}", source)
                                         for number, source in enumerate(compressed_images, 1)],
        instruction=build_instruction(inp.assessment_instructions),
        guide=output, max_tokens=budget, extra=extra,
    )
    choice = envelope["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ModelResponseError(f"Incomplete model response: {choice.get('finish_reason')!r}", envelope,
                                 choice.get("message", {}).get("content"))
    raw = choice["message"].get("content")
    if not isinstance(raw, str) or not raw.strip():
        raise ModelResponseError("Model returned empty content", envelope, raw)
    try:
        result = to_public_result(inp, output.model_validate_json(raw))
    except Exception as error:
        raise ModelResponseError(f"Model response validation failed: {error}", envelope, raw) from error
    return {"product_id": inp.product_id, "result": result.model_dump(), "raw_response": envelope,
            "inference": {"model": envelope.get("model", server.model), "usage": envelope.get("usage", {})}}
