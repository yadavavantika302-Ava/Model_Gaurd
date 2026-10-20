import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import src.app as app_module
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

    def test_training_estimator_validates_only_selected_algorithm_parameters(self):
        regression = app_module.make_training_estimator(
            "regression", "random_forest", {"learning_rate": "invalid"})
        classification = app_module.make_training_estimator(
            "classification", "random_forest", {"learning_rate": "2"})
        self.assertIsInstance(regression, app_module.RandomForestRegressor)
        self.assertIsInstance(classification, app_module.RandomForestClassifier)

        boosting = app_module.make_training_estimator(
            "regression", "gradient_boosting", {"learning_rate": ".2"})
        self.assertIsInstance(boosting, app_module.GradientBoostingRegressor)
        self.assertEqual(boosting.learning_rate, .2)
        with self.assertRaisesRegex(ValueError, "Learning rate must be between"):
            app_module.make_training_estimator(
                "regression", "gradient_boosting", {"learning_rate": "2"})

    def test_training_route_uses_beginner_defaults_and_advanced_algorithm_settings(self):
        regression_rows = ["feature,target"] + [f"{index},{index * 2 + 1}" for index in range(1, 31)]
        classification_rows = ["feature,target"] + [
            f"{index},{'yes' if index % 2 else 'no'}" for index in range(1, 31)]
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory, "models")
            report_dir = Path(directory, "reports")
            with patch.object(app_module, "TRAINING_MODEL_DIR", model_dir), \
                 patch.object(app_module, "TRAINING_RUN_DIR", report_dir):
                requests = [
                    ("rf-regression.csv", regression_rows, "regression", "random_forest", "advanced",
                     {"learning_rate": "invalid", "n_estimators": "20"}),
                    ("rf-classification.csv", classification_rows, "classification", "random_forest", "advanced",
                     {"learning_rate": "2", "n_estimators": "20"}),
                    ("rf-beginner.csv", regression_rows, "regression", "random_forest", "beginner",
                     {"learning_rate": "invalid", "n_estimators": "bad"}),
                    ("gb-valid.csv", regression_rows, "regression", "gradient_boosting", "advanced",
                     {"learning_rate": ".2", "n_estimators": "20"}),
                ]
                for filename, rows, task, algorithm, mode, parameters in requests:
                    with self.subTest(filename=filename):
                        response = self.client.post(
                            "/train",
                            data={"dataset": (io.BytesIO(("\n".join(rows) + "\n").encode()), filename),
                                  "task": task, "target": "target", "algorithm": algorithm,
                                  "mode": mode, **parameters},
                            content_type="multipart/form-data")
                        self.assertEqual(response.status_code, 200)
                        self.assertIn(b"TRAINING COMPLETE", response.data)

                rejected = self.client.post(
                    "/train",
                    data={"dataset": (io.BytesIO(("\n".join(regression_rows) + "\n").encode()), "gb-invalid.csv"),
                          "task": "regression", "target": "target", "algorithm": "gradient_boosting",
                          "mode": "advanced", "learning_rate": "2"},
                    content_type="multipart/form-data")
                self.assertIn(b"Learning rate must be between", rejected.data)

    def test_model_check_route_still_returns_validation_message(self):
        response = self.client.post("/check", data={})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Could not check model", response.data)


if __name__ == "__main__":
    unittest.main()
