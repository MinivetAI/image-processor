"""Prepare, validate saved responses, or make one real inference call."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time

from .image_picker import ImagePickerInput, build_output_model, run_image_picker
from .image_picker.instructions import build_instruction
from .server.alfred import build_server, client_session


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "validate", "run"])
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--response", type=Path, help="Raw response JSON for validate mode")
    parser.add_argument("--base-url", default=os.environ.get("IMAGE_PICKER_LLM_URL"))
    parser.add_argument("--model", default=os.environ.get("IMAGE_PICKER_LLM_MODEL"))
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--timeout", type=float, default=200)
    parser.add_argument("--disable-thinking", action="store_true", help="Qwen/vLLM-specific request option")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    # Relative local image paths are relative to the input file, not the caller's cwd.
    for image in payload.get("images", []):
        source = image["source"]
        if not source.startswith(("http://", "https://", "data:")):
            image["source"] = str((args.input.resolve().parent / source).resolve())
    inp = ImagePickerInput.model_validate(payload)
    output = build_output_model(inp)
    kwargs = {"max_tokens": args.max_tokens}
    if args.max_tokens is not None and args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "input.json", inp.model_dump())
    write_json(args.output / "schema.json", output.model_json_schema())
    (args.output / "instruction.txt").write_text(build_instruction(inp.assessment_instructions), encoding="utf-8")
    started = time.monotonic()
    calls = 0
    try:
        if args.mode == "prepare":
            write_json(args.output / "run.json", {"mode": "prepare", "llm_calls": 0,
                       "image_count": len(inp.images), "max_tokens": args.max_tokens})
            print(f"Prepared {inp.business_unit}/{inp.cms_vertical}; no LLM call")
            return
        if args.mode == "validate":
            if args.response is None:
                raise ValueError("validate requires --response")
            raw = json.loads(args.response.read_text(encoding="utf-8"))
            result = output.model_validate(raw)
            value = {"product_id": inp.product_id, "result": result.model_dump(), "inference": {"model": "saved-response", "usage": {}}}
        else:
            if not args.base_url or not args.model:
                raise ValueError("run requires --base-url and --model (or IMAGE_PICKER_LLM_URL/MODEL)")
            server = build_server(args.base_url, args.model, timeout=args.timeout)
            extra = {"chat_template_kwargs": {"enable_thinking": False}} if args.disable_thinking else None

            async def execute():
                async with client_session(server):
                    return await run_image_picker(payload, server, extra=extra, **kwargs)

            value = asyncio.run(execute())
            calls = 1
            envelope = value.pop("raw_response")
            write_json(args.output / "response-envelope.json", envelope)
            write_json(args.output / "raw-response.json", json.loads(envelope["choices"][0]["message"]["content"]))
            write_json(args.output / "usage.json", value["inference"])

        write_json(args.output / "result.json", value)
        write_json(args.output / "run.json", {"mode": args.mode, "status": "validated",
                   "llm_calls": calls, "seconds": round(time.monotonic() - started, 3)})
        print(f"Validated {len(value['result']['assessments'])} images; selected "
              f"{len(value['result']['selected_references'])} references; LLM calls: {calls}")
    except Exception as error:
        write_json(args.output / "run.json", {"mode": args.mode, "status": "failed",
                   "llm_calls": None if args.mode == "run" else 0, "seconds": round(time.monotonic() - started, 3),
                   "error_type": type(error).__name__, "error": str(error)})
        raise SystemExit(f"{type(error).__name__}: {error}") from None


if __name__ == "__main__":
    main()
