import io
import os
import unittest
from hashlib import sha256
from unittest.mock import patch

import numpy as np
from PIL import Image
from flask import render_template
from src.app import app

from src import image_testing


class Upload:
    def __init__(self, filename, payload):
        self.filename = filename
        self.stream = io.BytesIO(payload)


class FakeInterpreter:
    def __init__(self, input_detail=None, output_detail=None, fail=False):
        self.input_detail = input_detail or {
            "index": 0,
            "shape": np.asarray([1, 4, 4, 3]),
            "dtype": np.uint8,
            "quantization": (1.0, 0),
            "quantization_parameters": {
                "scales": np.asarray([1.0]), "zero_points": np.asarray([0]),
                "quantized_dimension": 0,
            },
        }
        self.output_detail = output_detail or {
            "index": 1,
            "shape": np.asarray([1, 3]),
            "dtype": np.uint8,
            "quantization": (1 / 256, 0),
            "quantization_parameters": {
                "scales": np.asarray([1 / 256]), "zero_points": np.asarray([0]),
                "quantized_dimension": 0,
            },
        }
        self.fail = fail
        self.tensor = None

    def get_input_details(self):
        return [self.input_detail]

    def get_output_details(self):
        return [self.output_detail]

    def set_tensor(self, _index, tensor):
        self.tensor = tensor

    def invoke(self):
        if self.fail:
            raise RuntimeError("mock inference failure")

    def get_tensor(self, _index):
        mean = float(self.tensor.astype(np.float32).mean()) / 255.0
        values = [0.65 + 0.25 * mean, 0.25 - 0.20 * mean, 0.10 - 0.05 * mean]
        return np.rint(np.asarray(values) * 256).astype(np.uint8)[None, :]


def png_bytes(size=(8, 6)):
    image = Image.new("RGB", size)
    for y in range(size[1]):
        for x in range(size[0]):
            image.putpixel((x, y), (20 + x * 10, 70 + y * 8, 130))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def tflite_stub():
    return b"xxxxTFL3test-model"


class ImageTestingTests(unittest.TestCase):
    def call_test(self, interpreter=None, **overrides):
        values = {
            "model_upload": Upload("sample.tflite", tflite_stub()),
            "image_upload": Upload("scene.png", png_bytes()),
            "labels_text": "barren,water,grass",
            "resize_strategy": "stretch",
            "preprocessing": "auto",
            "output_interpretation": "auto",
            "color_order": "rgb",
            "true_label": "barren",
        }
        values.update(overrides)
        with patch.object(image_testing, "_interpreter", return_value=interpreter or FakeInterpreter()):
            return image_testing.run_image_test(**values)

    def test_valid_image_runs_and_returns_descriptors_and_explanation(self):
        result = self.call_test()
        self.assertEqual(result["predicted_class"], "barren")
        self.assertTrue(result["truth_match"])
        self.assertEqual(result["image_analysis"]["format"], "PNG")
        self.assertIn("glcm_contrast", result["image_analysis"])
        self.assertIn("channel_mins", result["image_analysis"])
        self.assertEqual(set(result["image_analysis"]["quadrant_luminance"]),
                         {"top_left", "top_right", "bottom_left", "bottom_right"})
        self.assertTrue(result["explanation_preview"].startswith("data:image/jpeg;base64,"))
        self.assertIn("grayscale_entropy_bits", result["image_analysis"])
        self.assertIn("mean_gradient_magnitude", result["image_analysis"])
        self.assertEqual(len(result["controlled_experiments"]), 4)
        self.assertEqual(result["controlled_experiments"][0]["category"], "Colour")
        self.assertAlmostEqual(
            result["controlled_experiments"][0]["original_score"], result["scores"][0]["value"]
        )
        self.assertIn("not feature-importance percentages", result["experiments_note"])

    def test_raw_score_experiments_are_not_mislabeled_as_confidence(self):
        result = self.call_test(output_interpretation="scores")
        self.assertIsNone(result["confidence"])
        self.assertIn("Raw model score", result["target_score_label"])
        self.assertTrue(all("delta" in item for item in result["controlled_experiments"]))

    def test_exact_model_profile_corrects_unit_pixels_before_quantization(self):
        profile = next(iter(image_testing.MODEL_PROFILES.values()))
        with patch.object(image_testing, "_model_profile", return_value=profile):
            result = self.call_test(preprocessing="unit", labels_text="")
        self.assertEqual(result["preprocessing_used"], "raw")
        self.assertEqual(result["diagnostics"]["applied"]["pixel_range"], "raw")
        self.assertEqual(result["diagnostics"]["input_tensor_dtype"], "uint8")
        self.assertGreater(result["diagnostics"]["input_tensor_range"][1], 1)
        self.assertIn("verified", result["label_source"].lower())
        self.assertEqual(result["diagnostics"]["class_mapping"][2]["label"], "grass")
        self.assertTrue(result["diagnostics"]["profile_warnings"])

    def test_exact_profile_is_only_used_for_matching_model_bytes_and_tensor_metadata(self):
        profile = next(iter(image_testing.MODEL_PROFILES.values()))
        model_bytes = tflite_stub()
        profile_key = sha256(model_bytes).hexdigest()
        input_detail = {
            "shape": np.asarray(profile["input_shape"]), "dtype": np.uint8,
            "quantization": (profile["input_scale"], profile["input_zero_point"]),
            "quantization_parameters": {
                "scales": np.asarray([profile["input_scale"]]),
                "zero_points": np.asarray([profile["input_zero_point"]]),
                "quantized_dimension": 0,
            },
        }
        output_detail = {
            "shape": np.asarray(profile["output_shape"]), "dtype": np.uint8,
            "quantization": (profile["output_scale"], profile["output_zero_point"]),
            "quantization_parameters": {
                "scales": np.asarray([profile["output_scale"]]),
                "zero_points": np.asarray([profile["output_zero_point"]]),
                "quantized_dimension": 0,
            },
        }
        with patch.dict(image_testing.MODEL_PROFILES, {profile_key: profile}):
            self.assertIs(image_testing._model_profile(model_bytes, input_detail, output_detail), profile)
            wrong_input = dict(input_detail, quantization=(0.5, 0), quantization_parameters={
                "scales": np.asarray([0.5]), "zero_points": np.asarray([0]),
                "quantized_dimension": 0,
            })
            with self.assertRaisesRegex(ValueError, "unexpected input quantization"):
                image_testing._model_profile(model_bytes, wrong_input, output_detail)

    def test_user_label_mapping_preserves_supplied_output_order(self):
        result = self.call_test(labels_text="water,vegetation,grass", true_label="water")
        mapping = result["diagnostics"]["class_mapping"]
        self.assertEqual([item["label"] for item in mapping], ["water", "vegetation", "grass"])
        self.assertEqual(result["diagnostics"]["class_mapping_source"],
                         "User-provided labels (training order not independently verified)")

    def test_diagnostics_retain_raw_output_before_interpretation(self):
        result = self.call_test()
        diagnostics = result["diagnostics"]
        self.assertEqual(diagnostics["raw_model_output_dtype"], "uint8")
        self.assertEqual(diagnostics["raw_model_output_shape"], [1, 3])
        expected_raw = np.rint(np.asarray(diagnostics["interpreted_output_values"]) * 256).astype(int)
        self.assertEqual(diagnostics["raw_model_output"], [expected_raw.tolist()])
        self.assertEqual(diagnostics["output_quantization"]["scales"], [1 / 256])
        self.assertEqual(diagnostics["output_interpretation"], "probabilities")

    def test_tiny_image_descriptors_handle_single_pixel_dimension(self):
        image = Image.new("RGB", (1, 4), (10, 20, 30))
        descriptors = image_testing._image_descriptors(image)
        self.assertEqual(descriptors["width"], 1)
        self.assertEqual(descriptors["edge_density"], 0.0)

    def test_image_results_template_renders_experiments_and_reliability_sections(self):
        result = self.call_test()
        with app.test_request_context("/image-test"):
            html = render_template(
                "image_test.html", result=result, error=None,
                form_values={"labels": "", "resize_strategy": "stretch", "preprocessing": "auto",
                             "color_order": "rgb", "output_interpretation": "auto", "true_label": ""},
            )
        self.assertIn("Controlled experiments", html)
        self.assertIn("Reliability report", html)
        self.assertIn("Grayscale entropy", html)
        self.assertIn("Inference diagnostics", html)
        self.assertIn("Raw RGB pixel range", html)

    @unittest.skipUnless(
        os.environ.get("MODELGUARD_CAN_SAT_MODEL") and os.environ.get("MODELGUARD_MEADOW_IMAGE"),
        "Set MODELGUARD_CAN_SAT_MODEL and MODELGUARD_MEADOW_IMAGE to run the real terrain-model regression.",
    )
    def test_real_cansat_meadow_image_uses_verified_raw_rgb_profile(self):
        with open(os.environ["MODELGUARD_CAN_SAT_MODEL"], "rb") as model_file:
            model_bytes = model_file.read()
        with open(os.environ["MODELGUARD_MEADOW_IMAGE"], "rb") as image_file:
            image_bytes = image_file.read()
        result = image_testing.run_image_test(
            Upload("terrain_model_int8.tflite", model_bytes),
            Upload("meadow_016.jpg", image_bytes),
            labels_text="", resize_strategy="stretch", preprocessing="unit",
            output_interpretation="auto", color_order="rgb", true_label="grass",
        )
        self.assertEqual(result["predicted_class"], "grass")
        self.assertTrue(result["truth_match"])
        self.assertEqual(result["preprocessing_used"], "raw")
        self.assertEqual(result["diagnostics"]["input_tensor_dtype"], "uint8")
        self.assertEqual(result["diagnostics"]["raw_model_output"], [[0, 0, 255, 0, 0, 0]])
        self.assertAlmostEqual(result["confidence"], 255 / 256, places=6)
        self.assertEqual(result["output_interpretation_used"], "probabilities")
        self.assertIn("SOFTMAX", result["diagnostics"]["output_interpretation_source"])
        self.assertAlmostEqual(result["confidence"], 255 / 256, places=6)
        self.assertEqual(result["output_interpretation_used"], "probabilities")
        self.assertIn("SOFTMAX", result["diagnostics"]["output_interpretation_source"])

    def test_label_count_must_match_model_output(self):
        with self.assertRaisesRegex(ValueError, "model returns 3 class values"):
            self.call_test(labels_text="barren,water")

    def test_blank_labels_are_reported_as_generic_indices(self):
        result = self.call_test(labels_text="", true_label="")
        self.assertEqual(result["label_source"], "Generic output indices; class mapping unknown")
        self.assertTrue(result["predicted_class"].startswith("Class "))

    def test_true_label_must_be_configured(self):
        with self.assertRaisesRegex(ValueError, "must exactly match"):
            self.call_test(true_label="desert")

    def test_input_dimensions_are_validated(self):
        detail = FakeInterpreter().input_detail.copy()
        detail["shape"] = np.asarray([1, -1, 4, 3])
        with self.assertRaisesRegex(ValueError, "dynamic image dimensions"):
            self.call_test(interpreter=FakeInterpreter(input_detail=detail))

    def test_bgr_selection_reverses_rgb_channels(self):
        detail = FakeInterpreter().input_detail
        image = Image.new("RGB", (4, 4), (10, 20, 30))
        rgb, _, _ = image_testing._preprocess(image, detail, "raw", "stretch", "rgb")
        bgr, _, _ = image_testing._preprocess(image, detail, "raw", "stretch", "bgr")
        np.testing.assert_array_equal(rgb[0, 0, 0], [10, 20, 30])
        np.testing.assert_array_equal(bgr[0, 0, 0], [30, 20, 10])

    def test_unit_normalization_with_raw_uint8_quantization_collapses_pixels(self):
        detail = FakeInterpreter().input_detail
        image = Image.new("RGB", (4, 4), (120, 180, 240))
        raw, _, _ = image_testing._preprocess(image, detail, "raw", "stretch", "rgb")
        unit, _, _ = image_testing._preprocess(image, detail, "unit", "stretch", "rgb")
        np.testing.assert_array_equal(raw[0, 0, 0], [120, 180, 240])
        np.testing.assert_array_equal(unit[0, 0, 0], [0, 1, 1])

    def test_quantization_auto_detects_unit_domain_and_applies_int8_affine_mapping(self):
        detail = {
            "index": 0, "shape": np.asarray([1, 2, 2, 1]), "dtype": np.int8,
            "quantization": (1 / 255, -128),
            "quantization_parameters": {
                "scales": np.asarray([1 / 255]), "zero_points": np.asarray([-128]),
                "quantized_dimension": 0,
            },
        }
        mode, _, mismatch = image_testing._resolve_preprocessing(detail, "auto")
        self.assertEqual(mode, "unit")
        self.assertFalse(mismatch)
        image = Image.new("L", (2, 2), 255)
        tensor, _, _ = image_testing._preprocess(image, detail, "unit", "stretch", "rgb")
        self.assertTrue(np.all(tensor == 127))

    def test_probability_output_is_not_softmaxed_again(self):
        scores, interpretation, _, _ = image_testing._model_scores(
            FakeInterpreter(), FakeInterpreter().input_detail, FakeInterpreter().output_detail,
            np.zeros((1, 4, 4, 3), dtype=np.uint8), "auto"
        )
        self.assertEqual(interpretation, "probabilities")
        self.assertGreater(float(scores[0]), 0.64)
        self.assertAlmostEqual(float(scores.sum()), 1.0)

    def test_logits_override_does_not_softmax_probability_shaped_output(self):
        fake = FakeInterpreter()
        tensor = np.zeros((1, 4, 4, 3), dtype=np.uint8)
        auto, _, _, _ = image_testing._model_scores(fake, fake.input_detail, fake.output_detail, tensor, "auto")
        forced, interpretation, label, _ = image_testing._model_scores(
            fake, fake.input_detail, fake.output_detail, tensor, "logits"
        )
        self.assertEqual(interpretation, "probabilities")
        self.assertIn("Softmax skipped", label)
        np.testing.assert_array_equal(auto, forced)

    def test_corrupt_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "valid TensorFlow Lite model identifier"):
            self.call_test(model_upload=Upload("bad.tflite", b"badmodel"))

    def test_invalid_image_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Could not decode this image"):
            self.call_test(image_upload=Upload("scene.png", b"not an image"))

    def test_runtime_failure_is_reported(self):
        with self.assertRaisesRegex(ValueError, "mock inference failure"):
            self.call_test(interpreter=FakeInterpreter(fail=True))

    def test_float_model_requires_explicit_normalization(self):
        detail = {
            "dtype": np.float32,
            "quantization": (0.0, 0),
            "quantization_parameters": {"scales": np.asarray([]), "zero_points": np.asarray([])},
        }
        with self.assertRaisesRegex(ValueError, "does not identify a standard pixel range"):
            image_testing._resolve_preprocessing(detail, "auto")

    def test_rank_three_model_reports_explanation_unavailable(self):
        detail = FakeInterpreter().input_detail.copy()
        detail["shape"] = np.asarray([4, 4, 3])
        result = self.call_test(interpreter=FakeInterpreter(input_detail=detail))
        self.assertIsNone(result["explanation_preview"])
        self.assertIn("could not be generated", result["explanation_note"])


if __name__ == "__main__":
    unittest.main()
