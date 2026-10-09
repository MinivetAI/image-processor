import unittest
from copy import deepcopy
import base64
import json
import os
import tempfile
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from pathlib import Path
from PIL import Image

from fastapi.testclient import TestClient

from image_processor.app import create_app
from image_processor.server.alfred import build_server
from scripts.generate_examples import examples, model_response
from image_processor.image_picker.inputs import ImagePickerInput
from image_processor.image_picker.outputs import to_public_result


def sample_image_uri():
    image = Image.new("RGB", (32, 48), "navy")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class AppTests(unittest.TestCase):
    def test_review_page_can_load_explicit_local_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "cases.jsonl"
            dataset.write_text('{"case_id":"sample"}\n', encoding="utf-8")
            with patch.dict(os.environ, {"IMAGE_PROCESSOR_REVIEW_FILE": str(dataset)}):
                server = build_server("http://unused/v1", "test-model", timeout=1)
                with TestClient(create_app(server)) as client:
                    page = client.get("/review?load=1")
                    self.assertEqual(page.status_code, 200)
                    self.assertIn("review/data", page.text)
                    response = client.get("/review/data")
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.text, '{"case_id":"sample"}\n')

        with patch.dict(os.environ, {}, clear=True):
            server = build_server("http://unused/v1", "test-model", timeout=1)
            with TestClient(create_app(server)) as client:
                self.assertEqual(client.get("/review/data").status_code, 404)

    def test_service_exposes_health_and_picker_endpoints(self):
        app = create_app(SimpleNamespace(model="test-model"))
        paths = set(app.openapi()["paths"])
        self.assertIn("/healthz", paths)
        self.assertIn("/v1/image-processor/pick", paths)

    def test_http_examples_share_one_client_and_preserve_contracts(self):
        server = build_server("http://unused/v1", "fixture", timeout=1)
        cases = list(examples())
        envelopes = [
            {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(model_response(payload, result))}}],
             "model": "fixture", "usage": {"completion_tokens": 123}}
            for _, payload, result in cases
        ]
        with patch.object(server, "respond", new=AsyncMock(side_effect=envelopes)) as respond:
            with patch.object(server, "close", new=AsyncMock()) as close:
                with TestClient(create_app(server, disable_thinking=True)) as client:
                    self.assertEqual(client.get("/healthz").json(), {"status": "ok", "model": "fixture"})
                    respond.assert_not_called()
                    for name, payload, result in cases:
                        payload = deepcopy(payload)
                        for image in payload["images"]:
                            image["source"] = sample_image_uri()
                        response = client.post("/v1/image-processor/pick", json=payload)
                        self.assertEqual(response.status_code, 200, name + response.text)
                        public = to_public_result(ImagePickerInput.model_validate(payload), model_response(payload, result)).model_dump()
                        self.assertEqual(response.json(), {
                            "product_id": payload["product_id"], "result": public,
                            "processing_status": "complete",
                            "inference": {"model": "fixture", "usage": {"completion_tokens": 123}},
                        })
                    self.assertEqual(respond.await_count, 3)
                    for call in respond.await_args_list:
                        self.assertEqual(call.kwargs["extra"], {"chat_template_kwargs": {"enable_thinking": False}})
                        self.assertFalse(call.kwargs["allow_redirects"])
                close.assert_awaited_once()

    def test_http_request_resolves_omitted_reference_definition(self):
        _, payload, result = next(examples())
        del payload["assessment_definition"]
        for image in payload["images"]:
            image["source"] = sample_image_uri()
        server = build_server("http://unused/v1", "fixture", timeout=1)
        envelope = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(model_response(payload, result))}}]}
        with patch.object(server, "respond", new=AsyncMock(return_value=envelope)) as respond:
            with TestClient(create_app(server)) as client:
                response = client.post("/v1/image-processor/pick", json=payload)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["result"], to_public_result(ImagePickerInput.model_validate(payload), model_response(payload, result)).model_dump())
                respond.assert_awaited_once()
                fields = respond.call_args.kwargs["guide"].model_json_schema()["$defs"]["ProductTags"]["properties"]
                self.assertIn("carried_or_worn", fields)

    def test_invalid_http_requests_never_call_the_model(self):
        _, payload, _ = next(examples())
        unknown = deepcopy(payload)
        del unknown["assessment_definition"]
        unknown["cms_vertical"] = "unknown-vertical"
        duplicate = deepcopy(payload)
        duplicate["assessment_definition"]["fields"].append(duplicate["assessment_definition"]["fields"][0])
        server = build_server("http://unused/v1", "fixture", timeout=1)
        with patch.object(server, "respond", new=AsyncMock()) as respond:
            with TestClient(create_app(server)) as client:
                for invalid in (unknown, duplicate):
                    self.assertEqual(client.post("/v1/image-processor/pick", json=invalid).status_code, 422)
                respond.assert_not_called()

    def test_http_retry_returns_degraded_and_saves_raw_validation_diagnostics(self):
        _, payload, result = next(examples())
        payload = deepcopy(payload)
        for image in payload["images"]:
            image["source"] = sample_image_uri()
        invalid = deepcopy(model_response(payload, result))
        invalid["assessments"].pop()
        failed = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(invalid)}}],
                  "model": "fixture", "usage": {"completion_tokens": 30}}
        server = build_server("http://unused/v1", "fixture", timeout=1)
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"IMAGE_PROCESSOR_DIAGNOSTICS_DIR": directory}):
                with patch.object(server, "respond", new=AsyncMock(return_value=failed)) as respond:
                    with TestClient(create_app(server)) as client:
                        response = client.post("/v1/image-processor/pick", json=payload)
            self.assertEqual(response.status_code, 200, response.text)
            body = response.json()
            self.assertEqual(body["processing_status"], "degraded")
            self.assertEqual(len(body["result"]["assessments"]), len(payload["images"]))
            self.assertEqual(body["result"]["selected_references"], [])
            self.assertNotIn("raw_response", body)
            self.assertEqual(respond.await_count, 2)
            files = list(Path(directory).glob("*.json"))
            self.assertEqual(len(files), 1)
            diagnostic = json.loads(files[0].read_text())
            self.assertEqual(len(diagnostic["attempts"]), 2)
            self.assertEqual(diagnostic["attempts"][0]["envelope"], failed)

    def test_http_oversized_image_returns_degraded_without_model_call(self):
        _, payload, _ = next(examples())
        payload = deepcopy(payload)
        server = build_server("http://unused/v1", "fixture", timeout=1)
        with patch("image_processor.image_picker.tasks.prepare_images",
                   new=AsyncMock(side_effect=ValueError("Image dimensions exceed the 40 megapixel limit"))):
            with patch.object(server, "respond", new=AsyncMock()) as respond:
                with TestClient(create_app(server)) as client:
                    response = client.post("/v1/image-processor/pick", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["processing_status"], "degraded")
        self.assertEqual(response.json()["result"]["selected_references"], [])
        respond.assert_not_called()


if __name__ == "__main__":
    unittest.main()
