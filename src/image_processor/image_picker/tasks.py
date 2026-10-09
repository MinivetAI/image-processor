"""One Alfred call. Configuration and session ownership belong to the caller."""

from ..server.alfred import LLMServer, request_assessment

from .inputs import ImagePickerInput
from .instructions import build_instruction
from .media import prepare_images
from .outputs import build_output_model, build_public_output_model, to_public_result


def _validation_reason(error: Exception) -> str:
    """Keep correction prompts and client diagnostics concise and input-free."""
    errors = getattr(error, "errors", None)
    if callable(errors):
        try:
            details = errors(include_input=False, include_context=False, include_url=False)
            messages = []
            for item in details[:4]:
                location = ".".join(map(str, item.get("loc", ())))
                messages.append(f"{location}: {item['msg']}" if location else item["msg"])
            if messages:
                return "; ".join(messages)[:700]
        except TypeError:
            pass
    return str(error).splitlines()[0][:700]


def _degraded_result(inp: ImagePickerInput, reason: str) -> dict:
    """Return a structurally valid, deliberately non-selecting result."""
    result = {
        "assessments": [{
            "image_id": image.image_id,
            "variant_ids": [],
            "product_match": "uncertain",
            "quality": "unusable",
            "duplicate_of": None,
            "tags": {field.name: None for field in inp.assessment_definition.fields},
            "evidence": [],
            "limitations": ["Automated assessment was incomplete; observations are unknown."],
        } for image in inp.images],
        "variants": [],
        "selected_variant_id": None,
        "selected_references": [],
        "unresolved": [reason],
    }
    return build_public_output_model(inp).model_validate(result).model_dump()


async def run_image_picker(payload: dict, server: LLMServer, *, extra: dict | None = None,
                           max_tokens: int | None = None) -> dict:
    inp = ImagePickerInput.model_validate(payload)
    output = build_output_model(inp)
    budget = max_tokens if max_tokens is not None else 1024 + len(inp.images) * (200 + 40 * len(inp.assessment_definition.fields))
    if budget < 1:
        raise ValueError("max_tokens must be positive")
    try:
        compressed_images = await prepare_images([image.source for image in inp.images])
    except ValueError as error:
        reason = "Image preparation could not be completed: " + _validation_reason(error)
        return {
            "product_id": inp.product_id,
            "result": _degraded_result(inp, reason),
            "processing_status": "degraded",
            "raw_response": {},
            "attempt_diagnostics": [],
            "inference": {"model": server.model, "usage": {}},
        }
    context = inp.model_dump(exclude={"product_id", "images", "assessment_instructions"})
    context["images"] = [{"image_number": number} for number in range(1, len(inp.images) + 1)]
    images = [(f"Image {number}", source)
              for number, source in enumerate(compressed_images, 1)]
    instruction = build_instruction(inp.assessment_instructions)
    failures = []
    envelope = {}
    result = None
    for attempt in range(2):
        current_instruction = instruction
        if attempt:
            current_instruction += (
                "\n\nCORRECTION REQUIRED: Your previous response failed validation: "
                f"{failures[-1]['reason']} Reassess every numbered image and return the "
                "complete response again. Include exactly one assessment for every image, "
                "in input order. Ensure all variant labels, evidence, and selected "
                "references agree with one another."
            )
        envelope = await request_assessment(
            server, context=context, images=images, instruction=current_instruction,
            guide=output, max_tokens=budget, extra=extra,
        )
        raw = None
        try:
            choices = envelope.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise ValueError("Model response has no valid choice")
            choice = choices[0]
            message = choice.get("message")
            raw = message.get("content") if isinstance(message, dict) else None
            if choice.get("finish_reason") != "stop":
                raise ValueError(f"Response ended with finish_reason={choice.get('finish_reason')!r}")
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError("Model returned empty content")
            result = to_public_result(inp, output.model_validate_json(raw))
            break
        except Exception as error:
            failures.append({
                "attempt": attempt + 1,
                "reason": _validation_reason(error),
                "envelope": envelope,
            })

    model_name = envelope.get("model", server.model)
    usage = envelope.get("usage", {})
    diagnostics = failures if failures else []
    if result is None:
        reason = "Model response remained invalid after one corrective retry: " + failures[-1]["reason"]
        return {
            "product_id": inp.product_id,
            "result": _degraded_result(inp, reason),
            "processing_status": "degraded",
            "raw_response": envelope,
            "attempt_diagnostics": diagnostics,
            "inference": {"model": model_name, "usage": usage},
        }
    return {
        "product_id": inp.product_id,
        "result": result.model_dump(),
        "processing_status": "complete",
        "raw_response": envelope,
        "attempt_diagnostics": diagnostics,
        "inference": {"model": model_name, "usage": usage},
    }
