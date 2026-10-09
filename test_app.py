import unittest
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app import create_app
from generate_examples import examples
from server.alfred import build_server


class AppTests(unittest.TestCase):
    def test_service_exposes_health_and_picker_endpoints(self):
        app = create_app(SimpleNamespace(model="test-model"))
        paths = set(app.openapi()["paths"])
        self.assertIn("/healthz", paths)
        self.assertIn("/v1/image-processor/pick", paths)

    def test_http_examples_share_one_client_and_preserve_contracts(self):
        server = build_server("http://unused/v1", "fixture", timeout=1)
        cases = list(examples())
        envelopes = [
            {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(result)}}],
             "model": "fixture", "usage": {"completion_tokens": 123}}
            for _, _, result in cases
        ]
        with patch.object(server, "respond", new=AsyncMock(side_effect=envelopes)) as respond:
            with patch.object(server, "close", new=AsyncMock()) as close:
                with TestClient(create_app(server, disable_thinking=True)) as client:
                    self.assertEqual(client.get("/healthz").json(), {"status": "ok", "model": "fixture"})
                    respond.assert_not_called()
                    for name, payload, result in cases:
                        payload = deepcopy(payload)
                        for image in payload["images"]:
                            image["source"] = "data:image/png;base64,AA=="
                        response = client.post("/v1/image-processor/pick", json=payload)
                        self.assertEqual(response.status_code, 200, name + response.text)
                        self.assertEqual(response.json(), {
                            "product_id": payload["product_id"], "result": result,
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
            image["source"] = "data:image/png;base64,AA=="
        server = build_server("http://unused/v1", "fixture", timeout=1)
        envelope = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(result)}}]}
        with patch.object(server, "respond", new=AsyncMock(return_value=envelope)) as respond:
            with TestClient(create_app(server)) as client:
                response = client.post("/v1/image-processor/pick", json=payload)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["result"], result)
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


if __name__ == "__main__":
    unittest.main()
