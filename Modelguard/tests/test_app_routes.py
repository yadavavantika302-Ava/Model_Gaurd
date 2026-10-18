import io
import unittest

from src.app import app


class ExistingRouteRegressionTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_existing_workspaces_still_render(self):
        for path in ("/", "/check", "/train"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)

    def test_csv_training_preview_still_accepts_csv_upload(self):
        response = self.client.post(
            "/train/preview",
            data={"dataset": (io.BytesIO(b"feature,target\n1,0\n2,1\n3,0\n"), "sample.csv")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["rows"], 3)

    def test_model_check_route_still_returns_validation_message(self):
        response = self.client.post("/check", data={})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Could not check model", response.data)


if __name__ == "__main__":
    unittest.main()
