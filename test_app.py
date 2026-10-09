import unittest
from types import SimpleNamespace

from app import create_app


class AppTests(unittest.TestCase):
    def test_service_exposes_health_and_picker_endpoints(self):
        app = create_app(SimpleNamespace(model="test-model"))
        paths = {route.path for route in app.routes}
        self.assertIn("/healthz", paths)
        self.assertIn("/v1/image-processor/pick", paths)


if __name__ == "__main__":
    unittest.main()
