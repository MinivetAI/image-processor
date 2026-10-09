"""Contract and one-call orchestration tests; no real model or network required."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import unittest
import asyncio
from unittest.mock import AsyncMock, patch
from alfred import LLMServer

from pydantic import ValidationError

from generate_examples import examples
from image_picker import ImagePickerInput, build_output_model, run_image_picker
from types import SimpleNamespace

def prepare(payload):
    inp = ImagePickerInput.model_validate(payload)
    return SimpleNamespace(input=inp, output_model=build_output_model(inp))

def validate_result(prepared, raw):
    return prepared.output_model.model_validate(raw)


class PickerTests(unittest.TestCase):
    def setUp(self):
        self.cases = {name: (payload, raw) for name, payload, raw in examples()}
        self.payload, self.raw = deepcopy(self.cases["backpack"])
        for image in self.payload["images"]:
            image["source"] = "data:image/png;base64,AA=="
        self.prepared = prepare(self.payload)

    def reject(self, raw):
        with self.assertRaises((ValueError, ValidationError)):
            validate_result(self.prepared, raw)

    def test_one_inference_call_and_no_registry_import(self):
        import sys
        server = LLMServer("http://unused/v1", "fixture", retries=0)
        envelope = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(self.raw)}}]}
        with patch.object(server, "respond", new=AsyncMock(return_value=envelope)) as respond:
            result = asyncio.run(run_image_picker(self.payload, server))
            self.assertEqual(respond.await_count, 1)
            self.assertEqual(respond.call_args.kwargs["guide"].model_json_schema(), self.prepared.output_model.model_json_schema())
            self.assertEqual(len(result["result"]["selected_references"]), 3)
        self.assertNotIn("tasks.registry", sys.modules)

    def test_failure_does_not_make_a_repair_call(self):
        raw = deepcopy(self.raw)
        raw["assessments"].pop()
        server = LLMServer("http://unused/v1", "fixture", retries=0)
        envelope = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(raw)}}]}
        with patch.object(server, "respond", new=AsyncMock(return_value=envelope)) as respond:
            with self.assertRaises(ValueError):
                asyncio.run(run_image_picker(self.payload, server))
            self.assertEqual(respond.await_count, 1)

    def test_retry_configuration_is_rejected_before_inference(self):
        server = LLMServer("http://unused/v1", "fixture", retries=1)
        with patch.object(server, "respond", new=AsyncMock()) as respond:
            with self.assertRaisesRegex(ValueError, "retries=0"):
                asyncio.run(run_image_picker(self.payload, server))
            respond.assert_not_called()

    def test_bpc_and_lifestyle_have_different_required_fields(self):
        first = self.prepared.output_model.model_json_schema()["$defs"]["ProductTags"]
        second = prepare(self.cases["lipstick"][0]).output_model.model_json_schema()["$defs"]["ProductTags"]
        self.assertIn("carried_or_worn", first["required"])
        self.assertNotIn("carried_or_worn", second["properties"])
        self.assertIn("applied_on_human", second["required"])
        self.assertNotIn("applied_on_human", first["properties"])

    def test_all_319_definitions_compile_without_a_model(self):
        DEFAULT_DEFINITIONS = Path(__file__).parent / "reference/image_tags"
        files = list(DEFAULT_DEFINITIONS.rglob("*.json"))
        self.assertEqual(len(files), 319)
        for path in files:
            payload = {**self.payload, "business_unit": "Lifestyle" if path.parent.name == "LifeStyle" else path.parent.name,
                       "cms_vertical": path.stem}
            data = json.loads(path.read_text())
            payload["assessment_definition"] = {"fields": [{"name": t["tag"], "description": t["description"]} for t in data["tags"]]}
            prepared = prepare(payload)
            self.assertEqual(len(prepared.output_model.model_json_schema()["$defs"]["ProductTags"]["required"]), len(json.loads(path.read_text())["tags"]))

    def test_custom_category_has_no_repository_dependency(self):
        output = prepare({**self.payload, "business_unit": "Custom", "cms_vertical": "New Category"})
        self.assertEqual(output.output_model.model_validate(self.raw).model_dump(), self.raw)

    def test_missing_tags_and_extra_tags_are_rejected(self):
        raw = deepcopy(self.raw)
        del raw["assessments"][0]["tags"]["logo"]
        self.reject(raw)
        raw = deepcopy(self.raw)
        raw["assessments"][0]["tags"]["invented"] = "present"
        self.reject(raw)

    def test_original_ids_order_and_coverage_are_enforced(self):
        raw = deepcopy(self.raw)
        raw["assessments"].reverse()
        self.reject(raw)
        raw = deepcopy(self.raw)
        raw["selected_references"][0]["image_id"] = "invented"
        self.reject(raw)

    def test_uncertain_or_absent_role_is_not_selectable(self):
        raw = deepcopy(self.raw)
        raw["selected_references"][0]["roles"].append("interior_visible")
        self.reject(raw)

    def test_uncertain_identity_and_unusable_images_cannot_be_selected(self):
        for key, value in (("product_match", "uncertain"), ("quality", "unusable")):
            raw = deepcopy(self.raw)
            raw["assessments"][0][key] = value
            self.reject(raw)

    def test_duplicate_input_ids_and_forward_duplicate_links_are_rejected(self):
        payload = deepcopy(self.payload)
        payload["images"][1]["image_id"] = payload["images"][0]["image_id"]
        with self.assertRaises(ValueError):
            prepare(payload)
        raw = deepcopy(self.raw)
        raw["assessments"][0]["duplicate_of"] = "img-04"
        self.reject(raw)

    def test_duplicate_images_cannot_be_selected(self):
        raw = deepcopy(self.raw)
        raw["selected_references"].append({"image_id": "img-04", "roles": ["front_view"], "reason": "bad duplicate"})
        self.reject(raw)

    def test_variant_memberships_must_agree(self):
        raw = deepcopy(self.raw)
        raw["variants"][0]["image_ids"].pop()
        self.reject(raw)

    def test_empty_selection_has_no_false_winner(self):
        raw = deepcopy(self.raw)
        raw["selected_references"] = []
        self.reject(raw)
        raw["selected_variant_id"] = None
        raw["unresolved"] = ["No suitable references for requested purpose."]
        self.assertIsNone(validate_result(self.prepared, raw).selected_variant_id)

    def test_reference_limit_is_explicit_and_enforced(self):
        prepared = prepare({**self.payload, "max_references": 2})
        with self.assertRaisesRegex(ValueError, "max_references"):
            validate_result(prepared, self.raw)

    def test_every_positive_observation_needs_evidence(self):
        raw = deepcopy(self.raw)
        raw["assessments"][0]["evidence"].pop()
        self.reject(raw)

    def test_request_definition_changes_schema_but_not_policy(self):
        payload = deepcopy(self.payload)
        payload["assessment_definition"]["fields"].append({"name": "new_detail", "description": "New visible detail."})
        schema = prepare(payload).output_model.model_json_schema()
        self.assertIn("new_detail", schema["$defs"]["ProductTags"]["required"])
        self.assertIn("New visible detail.", schema["$defs"]["ProductTags"]["properties"]["new_detail"]["description"])

    def test_custom_types_and_nulls_are_strict(self):
        fields = [
            {"name": "lid", "type": "enum", "values": ["screw", "flip"], "description": "Visible lid design"},
            {"name": "printed_capacity", "type": "integer", "description": "Printed capacity in ml"},
            {"name": "aspect", "type": "number", "description": "Visible height/width ratio"},
            {"name": "label", "type": "string", "description": "Exact visible label text"},
            {"name": "mouth_visible", "type": "boolean", "description": "Actual opening visible"},
        ]
        payload = {**self.payload, "cms_vertical": "custom bottle", "assessment_definition": {"fields": fields}}
        tags = prepare(payload).output_model.model_fields["assessments"].annotation.__args__[0].model_fields["tags"].annotation
        value = {"lid": "screw", "printed_capacity": 500, "aspect": 2.5, "label": "", "mouth_visible": None}
        self.assertEqual(tags.model_validate(value).model_dump(), value)
        for name, wrong in [("lid", "unknown"), ("printed_capacity", "500"), ("printed_capacity", True), ("aspect", "2.5"), ("label", 3), ("mouth_visible", 1)]:
            with self.assertRaises(ValueError):
                tags.model_validate({**value, name: wrong})
        self.assertEqual(tags.model_validate(dict.fromkeys(value)).model_dump(), dict.fromkeys(value))

    def test_invalid_definitions_fail_before_call(self):
        bad_fields = [[], [{"name": "x", "description": "x", "type": "object"}],
            [{"name": "x", "description": "x", "type": "enum"}],
            [{"name": "x", "description": "x", "values": ["a"]}],
            [{"name": "model_dump", "description": "x"}],
            [{"name": "x", "description": " "}],
            [{"name": "x", "description": "x"}] * 2,
            [{"name": "x", "description": "x", "type": "enum", "values": ["a", "a"]}]]
        server = LLMServer("http://unused/v1", "fixture", retries=0)
        for fields in bad_fields:
            with patch.object(server, "respond", new=AsyncMock()) as respond:
                with self.assertRaises(ValueError):
                    asyncio.run(run_image_picker({**self.payload, "assessment_definition": {"fields": fields}}, server))
                respond.assert_not_called()

    def test_extra_cannot_disable_contract_or_request_multiple_responses(self):
        for key in ("messages", "response_format", "stream", "tools", "tool_choice", "n"):
            for default in (False, True):
                server = LLMServer("http://unused/v1", "fixture", retries=0,
                                   default_extra={key: None} if default else None)
                with patch.object(server, "respond", new=AsyncMock()) as respond:
                    with self.assertRaisesRegex(ValueError, "extra cannot override"):
                        asyncio.run(run_image_picker(self.payload, server, extra=None if default else {key: None}))
                    respond.assert_not_called()

    def test_custom_bottle_output_and_zero_numeric_evidence(self):
        payload, raw = deepcopy(self.cases["custom-bottle"])
        output = prepare(payload).output_model
        self.assertEqual(output.model_validate_json(json.dumps(raw)).model_dump(), raw)
        row = raw["assessments"][0]
        row["tags"]["printed_capacity_ml"] = 0
        raw["selected_references"][0]["roles"].append("printed_capacity_ml")
        self.assertEqual(output.model_validate(raw).assessments[0].tags.printed_capacity_ml, 0)
        row["evidence"] = [item for item in row["evidence"] if item["tag"] != "printed_capacity_ml"]
        with self.assertRaises(ValueError):
            output.model_validate(raw)

    def test_missing_definition_is_rejected(self):
        payload = deepcopy(self.payload)
        del payload["assessment_definition"]
        with self.assertRaises(ValueError):
            prepare(payload)

    def test_loaded_instructions_reach_the_same_single_alfred_call(self):
        from image_picker.instructions import IMAGE_PICKER, build_instruction
        guidance = "Prioritize lid design and drinking opening."
        payload = {**self.payload, "assessment_instructions": guidance}
        self.assertEqual(build_instruction(), IMAGE_PICKER)
        self.assertEqual(prepare(payload).output_model.model_json_schema(), self.prepared.output_model.model_json_schema())
        server = LLMServer("http://unused/v1", "fixture", retries=0)
        envelope = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(self.raw)}}]}
        with patch.object(server, "respond", new=AsyncMock(return_value=envelope)) as respond:
            result = asyncio.run(run_image_picker(payload, server))
            self.assertEqual(respond.await_count, 1)
            messages = respond.call_args.args[0]
            self.assertEqual(messages[0]["content"], build_instruction(guidance))
            self.assertTrue(messages[0]["content"].startswith(IMAGE_PICKER))
            context = json.loads(messages[1]["content"][0]["text"])
            self.assertNotIn("assessment_instructions", context)
            self.assertEqual(result["result"], self.raw)

    def test_invalid_loaded_instructions_fail_before_inference(self):
        server = LLMServer("http://unused/v1", "fixture", retries=0)
        for guidance in ("", " \n\t", 42, "x" * 16001):
            with patch.object(server, "respond", new=AsyncMock()) as respond:
                with self.assertRaises(ValueError):
                    asyncio.run(run_image_picker({**self.payload, "assessment_instructions": guidance}, server))
                respond.assert_not_called()

    def test_prepare_saves_the_composed_loaded_instruction(self):
        from image_picker.instructions import build_instruction
        payload, _ = self.cases["custom-bottle"]
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.json"
            output_path = Path(directory) / "prepared"
            input_path.write_text(json.dumps(payload))
            completed = subprocess.run([sys.executable, str(Path(__file__).with_name("run.py")),
                "prepare", str(input_path), "--output", str(output_path)], capture_output=True, text=True, timeout=15)
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            self.assertEqual((output_path / "instruction.txt").read_text(), build_instruction(payload["assessment_instructions"]))
            self.assertEqual(json.loads((output_path / "run.json").read_text())["llm_calls"], 0)

    def test_transport_makes_one_http_request_even_on_incomplete_response(self):
        owner = self
        class Handler(BaseHTTPRequestHandler):
            calls = 0
            finish_reason = "stop"
            status_code = 200
            def do_POST(self):
                Handler.calls += 1
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.assertEqual(body["response_format"]["type"], "json_schema")
                owner.assertEqual(sum(part["type"] == "image_url" for part in body["messages"][-1]["content"]), 4)
                envelope = {"choices": [{"finish_reason": Handler.finish_reason, "message": {"content": json.dumps(owner.raw)}}], "model": "mock", "usage": {"completion_tokens": 123}}
                self.send_response(Handler.status_code)
                if Handler.status_code == 307:
                    self.send_header("Location", "/redirected")
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(envelope).encode())
            def log_message(self, *args): pass
        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        async def execute():
            backend = LLMServer(f"http://127.0.0.1:{server.server_port}/v1", "mock", retries=0)
            try:
                return await run_image_picker(self.payload, backend)
            finally:
                await backend.close()
        try:
            with patch.object(LLMServer, "respond", autospec=True, side_effect=LLMServer.respond) as respond:
                response = asyncio.run(execute())
                self.assertEqual(respond.call_count, 1)
            self.assertEqual(response["result"], self.raw)
            self.assertEqual(response["inference"]["usage"], {"completion_tokens": 123})
            self.assertEqual(Handler.calls, 1)
            Handler.finish_reason = "length"
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                asyncio.run(execute())
            self.assertEqual(Handler.calls, 2)
            Handler.status_code = 307
            with self.assertRaisesRegex(RuntimeError, "307"):
                asyncio.run(execute())
            self.assertEqual(Handler.calls, 3)  # No automatic redirect or retry.
            Handler.status_code = 503
            with self.assertRaisesRegex(RuntimeError, "503"):
                asyncio.run(execute())
            self.assertEqual(Handler.calls, 4)  # Alfred retries are disabled too.

            Handler.status_code = 200
            Handler.finish_reason = "stop"
            async def legacy_call():
                backend = LLMServer(f"http://127.0.0.1:{server.server_port}/v1", "mock", retries=0)
                try:
                    content = backend.build_content("Images", media=[image["source"] for image in self.payload["images"]])
                    return await backend.respond([{"role": "user", "content": content}], guide=self.prepared.output_model)
                finally:
                    await backend.close()
            message = asyncio.run(legacy_call())
            self.assertEqual(json.loads(message["content"]), self.raw)
            self.assertNotIn("choices", message)

            with tempfile.TemporaryDirectory() as directory:
                input_path = Path(directory) / "input.json"
                output = Path(directory) / "run"
                input_path.write_text(json.dumps(self.payload))
                completed = subprocess.run([
                    sys.executable, str(Path(__file__).with_name("run.py")), "run", str(input_path),
                    "--output", str(output), "--base-url", f"http://127.0.0.1:{server.server_port}/v1",
                    "--model", "mock"], capture_output=True, text=True, timeout=15)
                self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
                self.assertEqual(json.loads((output / "run.json").read_text())["llm_calls"], 1)
                self.assertEqual(json.loads((output / "raw-response.json").read_text()), self.raw)
                self.assertEqual(json.loads((output / "response-envelope.json").read_text())["model"], "mock")
                self.assertEqual(Handler.calls, 6)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
