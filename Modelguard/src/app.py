from pathlib import Path
from datetime import datetime
from time import perf_counter
from importlib import metadata as package_metadata
import json
import os
import re
import tempfile
import uuid

import numpy as np
import pandas as pd
from flask import abort, request, render_template, send_file, jsonify, url_for
from flask import Flask
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score
)
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.base import clone
import joblib

ROOT = Path(__file__).resolve().parents[1]

WORKFLOW_REPORT_DIR = ROOT / "reports" / "workflow_runs"
WORKFLOW_PREDICTION_DIR = ROOT / "reports" / "workflow_predictions"
TRAINING_RUN_DIR = ROOT / "reports" / "training_runs"
TRAINING_MODEL_DIR = ROOT / "models" / "training_runs"

app = Flask(__name__)



def read_csv_upload(upload, label):
    if not upload or not upload.filename:
        raise ValueError(f"{label} is required.")

    if not upload.filename.lower().endswith(".csv"):
        raise ValueError(f"{label} must be a CSV file.")

    return pd.read_csv(upload, nrows=100000)


def training_profile(data):
    suggestions = []
    for column in data.columns:
        series = data[column]
        unique = int(series.nunique(dropna=True))
        if unique > 1:
            suggestions.append({"name": str(column), "unique": unique,
                                "kind": "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical",
                                "suggested": column == data.columns[-1]})
    return {"rows": len(data), "columns": len(data.columns), "missing": int(data.isna().sum().sum()),
            "duplicates": int(data.duplicated().sum()), "schema": suggestions,
            "preview": data.head(8).replace({np.nan: None}).to_dict(orient="records")}


def infer_training_task(target):
    values = target.dropna()
    if not pd.api.types.is_numeric_dtype(values) or values.nunique() <= min(20, max(2, int(len(values) * .05))):
        return "classification"
    return "regression"


def make_training_estimator(task, algorithm, options):
    trees = bounded_int(options, "n_estimators", 150, 10, 500)
    depth = bounded_int(options, "max_depth", 1, 1, 100) if options.get("max_depth") else None
    seed = 42
    if task == "classification":
        choices = {
            "auto": RandomForestClassifier(n_estimators=trees, max_depth=depth, random_state=seed, n_jobs=1),
            "random_forest": RandomForestClassifier(n_estimators=trees, max_depth=depth, random_state=seed, n_jobs=1),
            "logistic": LogisticRegression(max_iter=1000),
            "decision_tree": DecisionTreeClassifier(max_depth=depth, random_state=seed),
            "gradient_boosting": GradientBoostingClassifier(n_estimators=min(trees, 300), max_depth=depth or 3, random_state=seed),
            "svm": SVC(probability=True), "knn": KNeighborsClassifier(n_neighbors=5),
            "neural_network": MLPClassifier(hidden_layer_sizes=parse_hidden_layers(options.get("layers", "64,32")),
                activation=options.get("activation", "relu"), learning_rate_init=bounded_float(options, "learning_rate", .001, .00001, 1),
                batch_size=bounded_int(options, "batch_size", 32, 1, 4096), max_iter=bounded_int(options, "epochs", 200, 10, 1000),
                alpha=bounded_float(options, "dropout", .0001, 0, 1), random_state=seed, early_stopping=True),
        }
    else:
        choices = {
            "auto": RandomForestRegressor(n_estimators=trees, max_depth=depth, random_state=seed, n_jobs=1),
            "random_forest": RandomForestRegressor(n_estimators=trees, max_depth=depth, random_state=seed, n_jobs=1),
            "linear": LinearRegression(), "ridge": Ridge(), "lasso": Lasso(),
            "decision_tree": DecisionTreeRegressor(max_depth=depth, random_state=seed),
            "gradient_boosting": GradientBoostingRegressor(n_estimators=min(trees, 300), max_depth=depth or 3, random_state=seed),
            "neural_network": MLPRegressor(hidden_layer_sizes=parse_hidden_layers(options.get("layers", "64,32")),
                activation=options.get("activation", "relu"), learning_rate_init=bounded_float(options, "learning_rate", .001, .00001, 1),
                batch_size=bounded_int(options, "batch_size", 32, 1, 4096), max_iter=bounded_int(options, "epochs", 200, 10, 1000),
                alpha=bounded_float(options, "dropout", .0001, 0, 1), random_state=seed, early_stopping=True),
        }
    if algorithm not in choices:
        raise ValueError("Choose a supported algorithm for the selected task.")
    return choices[algorithm]


def parse_hidden_layers(value):
    try:
        layers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except (AttributeError, ValueError):
        raise ValueError("Neural network layers must be comma-separated positive integers.")
    if not layers or len(layers) > 8 or any(size < 1 or size > 2048 for size in layers):
        raise ValueError("Use 1–8 neural network layers, each with 1–2048 neurons.")
    return layers


def bounded_int(options, key, default, minimum, maximum):
    try:
        value = int(options.get(key, default))
    except (TypeError, ValueError):
        raise ValueError(f"{key.replace('_', ' ').title()} must be a whole number.")
    if value < minimum or value > maximum:
        raise ValueError(f"{key.replace('_', ' ').title()} must be between {minimum} and {maximum}.")
    return value


def bounded_float(options, key, default, minimum, maximum):
    try:
        value = float(options.get(key, default))
    except (TypeError, ValueError):
        raise ValueError(f"{key.replace('_', ' ').title()} must be a number.")
    if not np.isfinite(value) or value < minimum or value > maximum:
        raise ValueError(f"{key.replace('_', ' ').title()} must be between {minimum} and {maximum}.")
    return value


def train_dataset(data, target_column, task, algorithm, options):
    if len(data) < 10:
        raise ValueError("Upload at least 10 data rows to train and evaluate a model.")
    if len(data) > 50000 or len(data.columns) > 100:
        raise ValueError("Training is limited to 50,000 rows and 100 columns per run.")
    if task == "clustering":
        if len(data) > 20000 and algorithm == "dbscan":
            raise ValueError("DBSCAN is limited to 20,000 rows in this local workspace.")
        features = data.copy()
        y = None
    else:
        if target_column not in data.columns:
            raise ValueError("Select a target column for supervised training.")
        clean = data.dropna(subset=[target_column]).copy()
        if len(clean) < 10:
            raise ValueError("The selected target has too few non-missing rows.")
        y = clean.pop(target_column)
        features = clean
        if task == "auto":
            task = infer_training_task(y)
        if task not in {"classification", "regression"}:
            raise ValueError("Choose classification, regression, or clustering.")
        if task == "classification" and y.nunique() < 2:
            raise ValueError("Classification needs at least two target classes.")
    if features.shape[1] == 0:
        raise ValueError("There are no input features after selecting the target.")
    features = features.loc[:, features.nunique(dropna=True) > 1]
    if features.shape[1] == 0:
        raise ValueError("All input columns are constant or empty.")

    numeric = list(features.select_dtypes(include=["number", "bool"]).columns)
    categorical = [column for column in features.columns if column not in numeric]
    transforms = []
    if numeric:
        transforms.append(("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric))
    if categorical:
        transforms.append(("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", max_categories=100))]), categorical))
    preprocessor = ColumnTransformer(transforms, remainder="drop")
    test_size = .2
    if task == "clustering":
        if algorithm not in {"kmeans", "dbscan"}:
            algorithm = "kmeans"
        estimator = KMeans(n_clusters=max(2, min(bounded_int(options, "n_clusters", 3, 2, 100), max(2, len(features) - 1))), random_state=42, n_init=10) if algorithm == "kmeans" else DBSCAN(eps=bounded_float(options, "eps", .5, .01, 1000), min_samples=bounded_int(options, "min_samples", 5, 2, 1000))
        pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        pipeline.fit(features)
        labels = pipeline.predict(features) if algorithm == "kmeans" else pipeline.named_steps["model"].labels_
        score = None
        try:
            transformed = pipeline.named_steps["preprocess"].transform(features)
            if len(set(labels)) > 1 and len(set(labels)) < len(labels):
                score = round(float(silhouette_score(transformed, labels, sample_size=min(2000, len(labels)), random_state=42)), 4)
        except Exception:
            pass
        report = {"task": task, "algorithm": algorithm, "rows": len(features), "features": list(features.columns),
                  "metrics": {"clusters_found": int(len(set(labels)) - (1 if -1 in labels else 0)), "noise_rows": int(np.sum(np.asarray(labels) == -1)), "silhouette_score": score}}
        predictions = pd.DataFrame({"cluster": labels})
        return pipeline, report, predictions

    chronological = options.get("split") == "chronological"
    if chronological:
        split_column = options.get("time_column")
        if split_column not in features:
            raise ValueError("Choose a valid time column for chronological evaluation.")
        order = pd.to_datetime(features[split_column], errors="coerce")
        if order.notna().mean() < .8:
            raise ValueError("The selected chronological split column must contain parseable dates.")
        idx = order.sort_values(kind="stable").index
        cutoff = max(1, int(len(idx) * .8))
        train_idx, test_idx = idx[:cutoff], idx[cutoff:]
        if len(test_idx) == 0:
            raise ValueError("Not enough rows for a chronological holdout.")
    else:
        stratify = y if task == "classification" and y.value_counts().min() >= 2 and y.nunique() < len(y) * .5 else None
        train_idx, test_idx = train_test_split(features.index, test_size=test_size, random_state=42, stratify=stratify)
    model = make_training_estimator(task, algorithm, options)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(features.loc[train_idx], y.loc[train_idx])
    predicted = pipeline.predict(features.loc[test_idx])
    actual = y.loc[test_idx]
    if task == "classification":
        metrics = {"accuracy": round(float(accuracy_score(actual, predicted)), 4),
                   "precision_weighted": round(float(precision_score(actual, predicted, average="weighted", zero_division=0)), 4),
                   "recall_weighted": round(float(recall_score(actual, predicted, average="weighted", zero_division=0)), 4),
                   "f1_weighted": round(float(f1_score(actual, predicted, average="weighted", zero_division=0)), 4)}
    else:
        metrics = {"mae": round(float(mean_absolute_error(actual, predicted)), 4),
                   "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 4),
                   "r2": round(float(r2_score(actual, predicted)), 4) if len(actual) > 1 else None}
    report = {"task": task, "algorithm": algorithm, "target": target_column, "rows": len(data),
              "train_rows": len(train_idx), "test_rows": len(test_idx), "split": "chronological" if chronological else "random holdout",
              "features": list(features.columns), "metrics": metrics}
    predictions = pd.DataFrame({"actual": actual, "predicted": predicted}, index=test_idx).sort_index()
    return pipeline, report, predictions


def read_optional_csv_upload(upload):
    if not upload or not upload.filename:
        return None

    if not upload.filename.lower().endswith(".csv"):
        raise ValueError("Reference data must be a CSV file.")

    return pd.read_csv(upload, nrows=100000)


def parse_feature_columns(value):
    if not value:
        return None

    columns = [item.strip() for item in value.split(",") if item.strip()]
    return columns or None


def profile_dataset(data):
    numeric_columns = [
        column for column in data.columns
        if pd.api.types.is_numeric_dtype(data[column])
    ]
    missing_cells = int(data.isna().sum().sum())

    return {
        "row_count": int(len(data)),
        "column_count": int(len(data.columns)),
        "numeric_count": int(len(numeric_columns)),
        "missing_cells": missing_cells,
        "columns": list(data.columns[:30])
    }


def json_safe(value):
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
            if key != "clean_features"
        }

    if isinstance(value, list):
        return [json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [json_safe(item) for item in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (pd.DataFrame, pd.Series)):
        return None

    return value


def public_error_message(error):
    message = str(error).splitlines()[0][:600]
    for local_path in (str(ROOT), str(Path.cwd())):
        message = message.replace(local_path, "[local project]")
    message = re.sub(r"[A-Za-z]:\\[^\s,;]+", "[local path]", message)
    return message


def build_text_report(result):
    lines = [
        "MODELGUARD OFFLINE RELIABILITY REPORT",
        "=" * 40,
        f"Run ID: {result['run_id']}",
        f"Created: {result['created_at']}",
        f"Mode: {result['mode_label']}",
        f"Model: {result['model_name']}",
        f"Detected type: {result['model_profile']['task_label']}",
        f"Verdict: {result['verdict']}",
        f"Risk score: {result['risk_score']}",
        "",
        "ASSESSMENT",
        "-" * 10,
        result["assessment"],
        "",
        "FEATURES",
        "-" * 8,
        ", ".join(result["features"]),
        "",
        "MODEL INFORMATION",
        "-" * 17,
        f"Framework: {result['model_information']['framework']} {result['model_information']['framework_version']}",
        f"Family: {result['model_information']['family']}",
        f"Input dimensions: {result['model_information']['input_dimensions']}",
        f"Output dimensions: {result['model_information']['output_dimensions']}",
        f"Preprocessing: {result['model_information']['preprocessing']}",
        "",
        "DATA QUALITY",
        "-" * 12,
        f"Issues: {result['quality']['issue_count']}",
        f"Missing features: {result['quality']['missing_columns']}",
        f"Invalid values: {result['quality']['invalid_values']}",
        f"Duplicate rows: {result['quality']['duplicates']}",
        "",
        "PERFORMANCE",
        "-" * 11
    ]

    if result["performance"]:
        for key, value in result["performance"]["metrics"].items():
            lines.append(f"{key}: {value}")
    else:
        lines.append("No labeled metrics were calculated.")

    lines.extend([
        "",
        "MODEL EXPLANATION",
        "-" * 17
    ])
    explainability = result.get("explainability")
    if explainability:
        lines.append(f"Global explanation method: {explainability['method']}")
        lines.append(f"Individual explanation method: {explainability['local_method']}")
        lines.append(f"Selected CSV row {explainability['sample_row']} prediction: {explainability['sample_prediction']}")
        if explainability.get("simple_explanation"):
            lines.append(explainability["simple_explanation"])
        if explainability.get("global"):
            lines.append("Global parameter sensitivity:")
            lines.extend([
                f"- {item['feature']}: {item['importance']} ({item['help']})"
                for item in explainability["global"]
            ])
        if explainability.get("local"):
            lines.append("First row sensitivity (reset one feature to a typical reference value):")
            lines.extend([
                f"- {item['feature']}={item['value']}: {item['direction']} output by {item['impact']}"
                for item in explainability["local"]
            ])
        lines.append(explainability["baseline_note"])
    else:
        lines.append("Feature-level explanation was unavailable for this model.")

    lines.extend([
        "",
        "RECOMMENDATIONS",
        "-" * 15
    ])
    lines.extend([f"- {item}" for item in result["recommendations"]])
    return "\n".join(lines)


def save_workflow_artifacts(result, predictions_table=None):
    WORKFLOW_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    WORKFLOW_PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    run_id = result["run_id"]
    json_file = WORKFLOW_REPORT_DIR / f"{run_id}.json"
    text_file = WORKFLOW_REPORT_DIR / f"{run_id}.txt"

    if predictions_table is not None and not predictions_table.empty:
        prediction_file = WORKFLOW_PREDICTION_DIR / f"{run_id}.csv"
        predictions_table.to_csv(prediction_file, index=False)
        result["predictions_file"] = str(prediction_file)
    else:
        result["predictions_file"] = None

    result["report_file"] = str(json_file)
    result["text_report_file"] = str(text_file)
    safe_result = json_safe(result)
    json_file.write_text(json.dumps(safe_result, indent=4), encoding="utf-8")
    text_file.write_text(build_text_report(safe_result), encoding="utf-8")


class TFLiteModelAdapter:
    """Expose a tabular TFLite interpreter through the app's predict interface."""

    def __init__(self, model_bytes):
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            try:
                try:
                    from ai_edge_litert.interpreter import Interpreter
                except ImportError:
                    try:
                        from tensorflow.lite import Interpreter
                    except ImportError:
                        from tensorflow.lite.python.interpreter import Interpreter
            except ImportError as exc:
                raise ValueError(
                    "TFLite model detected, but no Lite runtime is installed. "
                    "Install ai-edge-litert, tflite-runtime, or TensorFlow in the environment that "
                    "runs ModelGuard, then restart the app."
                ) from exc

        try:
            self.interpreter = Interpreter(model_content=model_bytes)
            self.interpreter.allocate_tensors()
        except Exception as exc:
            raise ValueError(f"Could not initialize this TFLite model: {exc}") from exc

        self.inputs = self.interpreter.get_input_details()
        self.outputs = self.interpreter.get_output_details()
        if len(self.inputs) != 1 or len(self.outputs) != 1:
            raise ValueError(
                "This TFLite checker currently supports models with one input tensor "
                "and one output tensor."
            )

        input_shape = list(self.inputs[0]["shape"])
        feature_shape = input_shape[1:] if len(input_shape) > 1 else input_shape
        self.n_features_in_ = int(np.prod(feature_shape))
        output_shape = list(self.outputs[0]["shape"])
        output_shape = output_shape[1:] if len(output_shape) > 1 else output_shape
        self.output_width = int(np.prod(output_shape)) if output_shape else 1
        self.classes_ = None
        self.predict_proba = None

    @staticmethod
    def _dequantize(values, detail):
        scale, zero_point = detail.get("quantization", (0.0, 0))
        if scale:
            return (values.astype(np.float32) - zero_point) * scale
        return values

    def _run(self, data):
        try:
            values = data.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("TFLite input columns must contain numeric values.") from exc
        if values.shape[1] != self.n_features_in_:
            raise ValueError(
                f"TFLite model expects {self.n_features_in_} input values per row, "
                f"but the CSV provides {values.shape[1]} selected features."
            )

        input_detail = self.inputs[0]
        output_detail = self.outputs[0]
        input_shape = list(input_detail["shape"])
        if len(input_shape) > 1 and input_shape[0] not in (1, -1):
            raise ValueError(
                "This TFLite model requires a fixed batch size greater than one; "
                "the checker currently evaluates one CSV row at a time."
            )
        feature_shape = input_shape[1:] if len(input_shape) > 1 else input_shape
        if any(dimension <= 0 for dimension in feature_shape):
            feature_shape = [self.n_features_in_]
        expected_shape = ([1] + feature_shape) if len(input_shape) > 1 else feature_shape
        predictions = []
        for row in values:
            tensor = row.reshape(expected_shape)
            input_dtype = input_detail["dtype"]
            scale, zero_point = input_detail.get("quantization", (0.0, 0))
            if np.issubdtype(input_dtype, np.integer):
                if not scale:
                    raise ValueError("TFLite model uses integer inputs without quantization details.")
                limits = np.iinfo(input_dtype)
                tensor = np.clip(np.round(tensor / scale + zero_point), limits.min, limits.max)
            tensor = tensor.astype(input_dtype, copy=False)
            self.interpreter.set_tensor(input_detail["index"], tensor)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(output_detail["index"])
            output = self._dequantize(output, output_detail).reshape(-1)
            predictions.append(output)
        return np.vstack(predictions) if predictions else np.empty((0, self.output_width))

    def predict(self, data):
        outputs = self._run(data)
        return outputs[:, 0] if outputs.shape[1] == 1 else outputs


class KerasModelAdapter:
    """Tabular adapter for Keras models loaded with safe mode."""

    def __init__(self, model_bytes, suffix):
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ValueError("TensorFlow is required to load .keras models.") from exc
        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as model_file:
                model_file.write(model_bytes)
                path = model_file.name
            try:
                self.model = tf.keras.models.load_model(path, compile=False, safe_mode=True)
            except TypeError as exc:
                raise ValueError(
                    "This TensorFlow version does not provide safe Keras loading. "
                    "Use a compatible runtime that supports safe_mode."
                ) from exc
        except Exception as exc:
            raise ValueError(
                "Could not load this Keras model. Custom layers or objects may need "
                f"their source code registered before loading. Details: {exc}"
            ) from exc
        finally:
            if path and os.path.exists(path):
                os.unlink(path)

        if len(self.model.inputs) != 1 or len(self.model.outputs) != 1:
            raise ValueError("This Keras adapter currently supports one model input and one output.")
        shape = self.model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        self.input_shape = tuple(shape)
        self.n_features_in_ = int(np.prod([dimension for dimension in self.input_shape[1:] if dimension]))
        output_shape = self.model.output_shape
        if isinstance(output_shape, list):
            output_shape = output_shape[0]
        self.output_shape = tuple(output_shape)
        self.output_width = int(np.prod([dimension for dimension in self.output_shape[1:] if dimension])) or 1
        self.classes_ = None
        self.layers = self.model.layers
        self.predict_proba = None

    def count_params(self):
        return self.model.count_params()

    def _run(self, data):
        try:
            values = data.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("Keras tabular input columns must contain numeric values.") from exc
        if values.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Keras model expects {self.n_features_in_} input values per row, "
                f"but {values.shape[1]} were selected."
            )
        tensor_shape = [int(value) for value in self.input_shape[1:] if value]
        values = values.reshape((len(values), *tensor_shape))
        return np.asarray(self.model.predict(values, verbose=0)).reshape(len(values), -1)

    def predict(self, data):
        outputs = self._run(data)
        return outputs[:, 0] if outputs.shape[1] == 1 else outputs


class OnnxModelAdapter:
    """Single-input tabular adapter for ONNX models when onnxruntime is installed."""

    def __init__(self, model_bytes):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ValueError("Install onnxruntime to inspect .onnx model files.") from exc
        try:
            self.session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
        except Exception as exc:
            raise ValueError(f"Could not initialize this ONNX model: {str(exc)[:240]}") from exc
        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()
        if len(self.inputs) != 1 or len(self.outputs) != 1:
            raise ValueError("This ONNX adapter currently supports one input and one output tensor.")
        shape = list(self.inputs[0].shape)
        feature_shape = shape[1:] if len(shape) > 1 else shape
        known_dims = [int(value) for value in feature_shape if isinstance(value, (int, np.integer)) and value > 0]
        if len(known_dims) != len(feature_shape):
            raise ValueError("ONNX input feature dimensions are dynamic; specify a fixed-shape model export.")
        self.n_features_in_ = int(np.prod(known_dims))
        output_shape = list(self.outputs[0].shape)
        output_shape = output_shape[1:] if len(output_shape) > 1 else output_shape
        self.output_width = int(np.prod([value for value in output_shape if isinstance(value, int) and value > 0])) or 1
        self.classes_ = None
        self.predict_proba = None

    def _run(self, data):
        try:
            values = data.apply(pd.to_numeric, errors="raise").to_numpy()
        except (TypeError, ValueError) as exc:
            raise ValueError("ONNX tabular inputs must be numeric.") from exc
        if values.shape[1] != self.n_features_in_:
            raise ValueError(f"ONNX model expects {self.n_features_in_} features; got {values.shape[1]}.")
        type_name = self.inputs[0].type
        dtype = {
            "tensor(float)": np.float32,
            "tensor(double)": np.float64,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32
        }.get(type_name)
        if dtype is None:
            raise ValueError(f"ONNX input type '{type_name}' is not supported by the tabular adapter.")
        result = self.session.run([self.outputs[0].name], {self.inputs[0].name: values.astype(dtype)})[0]
        return np.asarray(result).reshape(len(values), -1)

    def predict(self, data):
        outputs = self._run(data)
        return outputs[:, 0] if outputs.shape[1] == 1 else outputs


def load_existing_model(upload):
    if upload and upload.filename:
        model_bytes = upload.read()
        filename = upload.filename.lower()
        if filename.endswith(".tflite") or model_bytes[4:8] == b"TFL3":
            model = TFLiteModelAdapter(model_bytes)
            metadata = {
                "model_type": "TensorFlow Lite model",
                "target": "",
                "source": upload.filename,
                "features": None
            }
            return model, metadata
        if filename.endswith(".keras"):
            model = KerasModelAdapter(model_bytes, Path(filename).suffix)
            metadata = {
                "model_type": "Keras model", "target": "",
                "source": upload.filename, "features": None
            }
            return model, metadata
        if filename.endswith(".onnx"):
            model = OnnxModelAdapter(model_bytes)
            metadata = {
                "model_type": "ONNX model", "target": "",
                "source": upload.filename, "features": None
            }
            return model, metadata
        if filename.endswith((".pkl", ".pickle", ".joblib", ".sav", ".h5", ".hdf5", ".pt", ".pth")):
            raise ValueError(
                "This serialized format can execute model-defined or pickle code when loaded. "
                "ModelGuard will not deserialize it in the web process. Export the model to ONNX, "
                "TFLite, or safe-mode .keras, or use a separately sandboxed model worker."
            )
        if filename.endswith((".json", ".ubj", ".txt", ".cbm", ".model")):
            raise ValueError(
                "This native booster format does not have a configured safe adapter in this build. "
                "Export it to ONNX or use a dedicated restricted model worker."
            )
        raise ValueError(
            "Unsupported model format. This build accepts .onnx, .tflite, and safe-mode .keras files."
        )

    raise ValueError("Upload a supported existing model file to continue.")


def model_methods(model):
    methods = [
        name for name in [
            "predict",
            "predict_proba",
            "decision_function",
            "transform",
            "fit_predict",
            "score_samples"
        ]
        if callable(getattr(model, name, None))
    ]
    return methods


def model_predict(model, data):
    """Call a tabular model with a DataFrame, then a numeric matrix fallback."""
    predict = getattr(model, "predict", None)
    if not callable(predict):
        raise ValueError("The uploaded model does not expose a callable predict() method.")
    try:
        return np.asarray(predict(data))
    except (TypeError, ValueError, AttributeError) as dataframe_error:
        try:
            numeric = data.apply(pd.to_numeric, errors="raise").to_numpy()
            return np.asarray(predict(numeric))
        except (TypeError, ValueError, AttributeError) as array_error:
            raise ValueError(
                "The model could not predict from these CSV columns. Check the "
                "feature names, order, data types, and expected input shape. "
                f"DataFrame error: {dataframe_error}; array error: {array_error}"
            ) from array_error


def model_predict_proba(model, data):
    predict_proba = getattr(model, "predict_proba", None)
    if not callable(predict_proba):
        raise ValueError("This model does not expose predict_proba().")
    try:
        return np.asarray(predict_proba(data))
    except (TypeError, ValueError, AttributeError) as dataframe_error:
        try:
            numeric = data.apply(pd.to_numeric, errors="raise").to_numpy()
            return np.asarray(predict_proba(numeric))
        except (TypeError, ValueError, AttributeError) as array_error:
            raise ValueError(
                "The model could not calculate probabilities from these CSV columns. "
                f"DataFrame error: {dataframe_error}; array error: {array_error}"
            ) from array_error


def detect_model_profile(model, metadata=None):
    metadata = metadata or {}
    estimator_type = getattr(model, "_estimator_type", None)
    class_name = type(model).__name__
    name = class_name.lower()
    methods = model_methods(model)
    task_type = metadata.get("task_type")

    if not task_type:
        if estimator_type in {"classifier", "regressor", "clusterer"}:
            task_type = {
                "classifier": "classification",
                "regressor": "regression",
                "clusterer": "clustering"
            }[estimator_type]
        elif "classifier" in name or "svc" in name or "logistic" in name:
            task_type = "classification"
        elif "regressor" in name or "regression" in name:
            task_type = "regression"
        elif "cluster" in name or "kmeans" in name or "dbscan" in name:
            task_type = "clustering"
        elif "isolationforest" in name or "oneclass" in name or "outlier" in name:
            task_type = "anomaly"
        elif "predict_proba" in methods:
            task_type = "classification"
        elif "predict" in methods:
            task_type = "prediction"
        elif "transform" in methods:
            task_type = "transformer"
        else:
            task_type = "unsupported"

    task_labels = {
        "classification": "Classification",
        "regression": "Regression",
        "clustering": "Clustering",
        "anomaly": "Anomaly / outlier detection",
        "prediction": "Generic prediction model",
        "transformer": "Transformer / feature extractor",
        "unsupported": "Unsupported model interface"
    }

    return {
        "class_name": class_name,
        "task_type": task_type,
        "task_label": task_labels.get(task_type, task_type.title()),
        "methods": methods,
        "interface_label": ", ".join(methods) if methods else "No runnable method detected",
        "supports_predict": "predict" in methods,
        "supports_proba": "predict_proba" in methods,
        "supports_uncertainty": hasattr(model, "estimators_")
    }


FEATURE_HELP = {
    "hr": "Hour of day (0–23).",
    "season": "Season category (1–4).",
    "yr": "Year indicator used by the dataset.",
    "mnth": "Month of year (1–12).",
    "holiday": "Whether the date is a holiday.",
    "weekday": "Day of week (0–6).",
    "workingday": "Whether this is a working day.",
    "weathersit": "Weather condition category.",
    "temp": "Normalized temperature.",
    "atemp": "Normalized feels-like temperature.",
    "hum": "Normalized humidity.",
    "windspeed": "Normalized wind speed.",
    "lat": "Latitude coordinate.",
    "latitude": "Latitude coordinate.",
    "lon": "Longitude coordinate.",
    "longitude": "Longitude coordinate.",
    "elevation": "Height above sea level.",
    "slope": "Terrain steepness.",
    "aspect": "Direction a terrain face points.",
    "traffic_volume": "Observed or estimated traffic volume."
}


def explain_predictions(model, features, feature_data, predictions, reference, selected_row=0, task_type="prediction"):
    """Provide model-level importance and a readable local sensitivity view."""
    if not len(predictions) or feature_data.empty:
        return None

    estimator = model
    importances_features = list(features)
    if isinstance(model, Pipeline):
        estimator = model.named_steps.get("estimator", model)
    importances = getattr(estimator, "feature_importances_", None)
    ranked = []
    if importances is not None and len(importances) != len(features):
        try:
            encoded_names = model.named_steps["preprocessor"].get_feature_names_out()
            aggregated = {feature: 0.0 for feature in features}
            for encoded_name, importance in zip(encoded_names, importances):
                encoded_text = str(encoded_name)
                matched = next((feature for feature in features
                               if encoded_text == feature or encoded_text.endswith(f"__{feature}")
                               or encoded_text.startswith(f"numeric__{feature}")
                               or encoded_text.startswith(f"categorical__{feature}_")), None)
                if matched is not None:
                    aggregated[matched] += float(importance)
            importances_features = list(aggregated)
            importances = np.array(list(aggregated.values()))
        except (AttributeError, KeyError, ValueError):
            importances = None
    if importances is not None and len(importances) == len(importances_features):
        ranked = sorted(
            zip(importances_features, importances), key=lambda item: float(item[1]), reverse=True
        )
        ranked = [
            {"feature": name, "importance": round(float(score), 4),
             "help": FEATURE_HELP.get(str(name).lower(), "Input parameter used by the model.")}
            for name, score in ranked[:10]
        ]

    baseline_source = reference if reference is not None and not reference.empty else feature_data
    explanation_method = "Built-in model feature importance"
    shap_global = None
    shap_local = None
    output_component = None
    if (callable(getattr(model, "predict", None)) and len(features) <= 40
            and not (task_type == "classification" and not callable(getattr(model, "predict_proba", None)))):
        try:
            import shap

            background = baseline_source[features].head(24).copy()
            sampled_rows = feature_data[features].head(32).copy()

            def numeric_model_output(values):
                values = pd.DataFrame(values, columns=features)
                if callable(getattr(model, "predict_proba", None)):
                    output = model_predict_proba(model, values)
                else:
                    output = model_predict(model, values)
                output = np.asarray(output)
                if output.ndim == 1:
                    output = output.reshape(-1, 1)
                return output.astype(float)

            explainer = shap.Explainer(
                numeric_model_output,
                background,
                algorithm="permutation",
                feature_names=features
            )
            max_evals = max(2 * len(features) + 1, 21)
            global_values = np.asarray(explainer(sampled_rows, max_evals=max_evals).values)
            if global_values.ndim == 2:
                global_values = global_values[:, :, np.newaxis]
            if global_values.ndim == 3 and global_values.shape[1] == len(features):
                importance = np.mean(np.abs(global_values), axis=(0, 2))
                total = float(importance.sum())
                if total > 0:
                    shap_global = [
                        {"feature": features[index], "importance": float(score / total),
                         "help": FEATURE_HELP.get(str(features[index]).lower(), "Input parameter used by the model.")}
                        for index, score in sorted(enumerate(importance), key=lambda pair: pair[1], reverse=True)[:10]
                    ]

                selected_row_data = feature_data[features].iloc[[selected_row]]
                local_values = np.asarray(explainer(selected_row_data, max_evals=max_evals).values)
                if local_values.ndim == 2:
                    local_values = local_values[:, :, np.newaxis]
                if local_values.ndim == 3 and local_values.shape[1] == len(features):
                    predicted_class = predictions[selected_row] if len(predictions) > selected_row else None
                    estimator_for_classes = model
                    if isinstance(model, Pipeline) and model.steps:
                        estimator_for_classes = model.steps[-1][1]
                    classes = list(getattr(estimator_for_classes, "classes_", []))
                    output_index = classes.index(predicted_class) if predicted_class in classes else 0
                    output_index = min(output_index, local_values.shape[2] - 1)
                    if not classes and local_values.shape[2] > 1:
                        output_component = output_index
                    signed_values = local_values[0, :, output_index]
                    shap_local = [
                        {"feature": name, "value": json_safe(selected_row_data.iloc[0][name]),
                         "impact": round(float(signed_values[index]), 6),
                         "direction": "raises" if signed_values[index] > 0 else "lowers" if signed_values[index] < 0 else "does not change",
                         "help": FEATURE_HELP.get(str(name).lower(), "Input parameter used by the model.")}
                        for index, name in sorted(enumerate(features), key=lambda pair: abs(signed_values[pair[0]]), reverse=True)[:8]
                    ]
                    explanation_method = "SHAP permutation contributions"
        except Exception:
            shap_global = None
            shap_local = None

    if shap_global is not None:
        ranked = shap_global
    elif not ranked and callable(getattr(model, "predict", None)):
        # Black-box fallback: replace one column at a time with a typical value
        # and measure how much the model output changes over a small row sample.
        explanation_method = "Model-agnostic one-feature-at-a-time sensitivity"
        sample = feature_data.head(128).copy()
        try:
            baseline_predictions = model_predict(model, sample)
            sensitivities = []
            for feature in features:
                if pd.api.types.is_numeric_dtype(baseline_source[feature]):
                    typical = pd.to_numeric(baseline_source[feature], errors="coerce").median()
                    if pd.isna(typical):
                        continue
                else:
                    modes = baseline_source[feature].dropna().mode()
                    if modes.empty:
                        continue
                    typical = modes.iloc[0]
                changed = sample.copy()
                changed.loc[:, feature] = typical
                changed_predictions = model_predict(model, changed)
                try:
                    before = np.asarray(baseline_predictions, dtype=float)
                    after = np.asarray(changed_predictions, dtype=float)
                    if before.shape != after.shape:
                        continue
                    sensitivity = float(np.nanmean(np.abs(before - after)))
                except (TypeError, ValueError):
                    before = np.asarray(baseline_predictions, dtype=object)
                    after = np.asarray(changed_predictions, dtype=object)
                    if before.shape != after.shape:
                        continue
                    changed_flags = before != after
                    if changed_flags.ndim > 1:
                        changed_flags = np.any(changed_flags, axis=tuple(range(1, changed_flags.ndim)))
                    sensitivity = float(np.mean(changed_flags))
                sensitivities.append((feature, sensitivity))
            total_sensitivity = sum(score for _, score in sensitivities)
            if total_sensitivity > 0:
                ranked = sorted(sensitivities, key=lambda item: item[1], reverse=True)
                ranked = [
                    {"feature": name, "importance": score / total_sensitivity,
                     "help": FEATURE_HELP.get(str(name).lower(), "Input parameter used by the model.")}
                    for name, score in ranked[:10]
                ]
        except Exception:
            ranked = []

    row = feature_data.iloc[[selected_row]].copy()
    local = []
    local_explanation_method = "One-feature-at-a-time sensitivity"
    score_label = "predicted output"
    try:
        original_prediction = model_predict(model, row)[0]
        classes = getattr(model, "classes_", None)
        class_index = None
        if classes is not None and callable(getattr(model, "predict_proba", None)):
            class_index = list(classes).index(original_prediction)
            original_score = float(model_predict_proba(model, row)[0][class_index])
            score_label = f"probability of class {original_prediction!s}"
        else:
            if task_type == "classification" and np.isscalar(original_prediction):
                original_score = original_prediction
                score_label = "predicted class"
            else:
                try:
                    original_score = float(original_prediction)
                    score_label = "predicted value"
                except (TypeError, ValueError):
                    if np.isscalar(original_prediction):
                        original_score = original_prediction
                        score_label = "predicted class"
                    else:
                        original_score = np.asarray(original_prediction, dtype=float)
                        score_label = "output vector"

        for feature in features if shap_local is None else []:
            if pd.api.types.is_numeric_dtype(baseline_source[feature]):
                typical = pd.to_numeric(baseline_source[feature], errors="coerce").median()
                if pd.isna(typical):
                    continue
            else:
                modes = baseline_source[feature].dropna().mode()
                if modes.empty:
                    continue
                typical = modes.iloc[0]
            changed = row.copy()
            changed.loc[:, feature] = typical
            if class_index is not None:
                changed_score = float(model_predict_proba(model, changed)[0][class_index])
                delta = float(original_score) - changed_score
            elif score_label == "predicted class":
                changed_prediction = model_predict(model, changed)[0]
                changed_score = 0.0 if changed_prediction == original_score else 1.0
                delta = changed_score
            elif score_label == "output vector":
                changed_score = np.asarray(model_predict(model, changed)[0], dtype=float)
                delta = float(np.linalg.norm(original_score - changed_score))
            else:
                changed_score = float(model_predict(model, changed)[0])
                delta = float(original_score) - changed_score
            local.append({
                "feature": feature,
                "value": json_safe(row.iloc[0][feature]),
                "typical": round(float(typical), 4) if isinstance(typical, (int, float, np.number)) else str(typical),
                "impact": round(float(delta), 4),
                "direction": (
                    "changes" if score_label == "predicted class" and delta
                    else "does not change" if score_label == "predicted class"
                    else "changes" if score_label == "output vector" and delta
                    else "does not change" if score_label == "output vector"
                    else "raises" if delta > 0 else "lowers" if delta < 0 else "does not change"
                ),
                "help": FEATURE_HELP.get(str(feature).lower(), "Input parameter used by the model.")
            })
    except Exception:
        local = []

    if shap_local is not None:
        local = shap_local
        local_explanation_method = "SHAP permutation contributions"
    local.sort(key=lambda item: abs(item["impact"]), reverse=True)
    if shap_local:
        positive = [item["feature"] for item in shap_local if item["impact"] > 0][:3]
        negative = [item["feature"] for item in shap_local if item["impact"] < 0][:3]
        component_note = f" for output component {output_component}" if output_component is not None else ""
        simple_explanation = (
            f"The model predicted {json_safe(predictions[selected_row])}. "
            f"For the explained score{component_note}, {', '.join(positive) if positive else 'no features'} contributed positively and "
            f"{', '.join(negative) if negative else 'no features'} contributed negatively, according to SHAP."
        )
    elif local:
        strongest = ", ".join(item["feature"] for item in local[:3])
        simple_explanation = (
            f"The model predicted {json_safe(predictions[selected_row])}. "
            f"Changing {strongest} to typical reference values changed this row's output most in the one-feature sensitivity check."
        )
    else:
        simple_explanation = "The model returned an output, but no supported feature-level explanation was available."
    return {
        "method": explanation_method,
        "local_method": local_explanation_method if local else "Unavailable",
        "global": ranked,
        "local": local[:8],
        "sample_prediction": json_safe(predictions[selected_row]) if len(predictions) > selected_row else None,
        "score_label": score_label,
        "output_component": output_component,
        "sample_row": selected_row + 1,
        "sample_values": json_safe(row.iloc[0].to_dict()),
        "simple_explanation": simple_explanation,
        "baseline_note": (
            "SHAP contributions show the selected output's change relative to the explainer's background baseline; they describe model behavior, not causation."
            if shap_local is not None else
            "Local sensitivity compares the selected row with typical reference values; these changes describe model behavior, not causation."
        )
    }


def check_quality(data, features):
    missing_columns = [
        feature for feature in features
        if feature not in data.columns
    ]
    available = [
        feature for feature in features
        if feature in data.columns
    ]

    issues = []
    warnings = []

    if missing_columns:
        issues.append(f"Missing required features: {missing_columns}")

    if not available:
        issues.append("No required model features are available.")
        return {
            "missing_columns": ", ".join(missing_columns) or "None",
            "invalid_values": 0,
            "duplicates": int(data.duplicated().sum()),
            "non_numeric": "None",
            "issues": issues,
            "warnings": warnings,
            "issue_count": len(issues),
            "clean_features": None
        }

    numeric_data = pd.DataFrame(index=data.index)
    for column in available:
        numeric_data[column] = pd.to_numeric(data[column], errors="coerce")
    non_numeric = []
    invalid_values = int(data[available].isna().sum().sum())
    duplicates = int(data.duplicated().sum())
    infinite_values = int(
        numeric_data.isin([np.inf, -np.inf]).sum().sum()
    )

    if invalid_values:
        issues.append(f"Found {invalid_values} invalid or missing feature values.")

    if infinite_values:
        issues.append(f"Found {infinite_values} infinite feature values.")

    if duplicates:
        warnings.append(f"Found {duplicates} duplicate rows.")

    range_limits = {
        "hr": (0, 23),
        "season": (1, 4),
        "yr": (0, 1),
        "mnth": (1, 12),
        "holiday": (0, 1),
        "weekday": (0, 6),
        "workingday": (0, 1),
        "weathersit": (1, 4),
        "temp": (0, 1),
        "atemp": (0, 1),
        "hum": (0, 1),
        "windspeed": (0, 1)
    }

    for column, (minimum, maximum) in range_limits.items():
        if column in numeric_data:
            invalid_range = int(
                ((numeric_data[column] < minimum) |
                 (numeric_data[column] > maximum)).sum()
            )
            if invalid_range:
                issues.append(
                    f"{column}: {invalid_range} values outside "
                    f"the expected range [{minimum}, {maximum}]."
                )

    clean_features = None
    if not missing_columns and not issues:
        clean_features = data[features].copy()

    return {
        "missing_columns": ", ".join(missing_columns) or "None",
        "invalid_values": invalid_values + infinite_values,
        "duplicates": duplicates,
        "non_numeric": ", ".join(non_numeric) or "None",
        "issues": issues,
        "warnings": warnings,
        "issue_count": len(issues),
        "clean_features": clean_features
    }


def calculate_metrics(actual, predictions, task_type):
    actual = pd.Series(actual).reset_index(drop=True)
    predictions = pd.Series(predictions).reset_index(drop=True)
    valid = actual.notna() & predictions.notna()

    if not valid.all():
        actual = actual[valid]
        predictions = predictions[valid]

    if actual.empty:
        return {
            "task_type": task_type,
            "metrics": {},
            "primary_metric": None,
            "primary_metric_name": "Unavailable"
        }

    if task_type == "prediction":
        numeric_actual = pd.to_numeric(actual, errors="coerce")
        numeric_predictions = pd.to_numeric(predictions, errors="coerce")
        if numeric_actual.notna().all() and numeric_predictions.notna().all():
            task_type = "regression"
        else:
            task_type = "classification"

    if task_type == "classification":
        average = "weighted"
        metrics = {
            "Accuracy": round(float(accuracy_score(actual, predictions)), 4),
            "Precision": round(float(precision_score(
                actual, predictions, average=average, zero_division=0
            )), 4),
            "Recall": round(float(recall_score(
                actual, predictions, average=average, zero_division=0
            )), 4),
            "F1": round(float(f1_score(
                actual, predictions, average=average, zero_division=0
            )), 4)
        }
        return {
            "task_type": task_type,
            "metrics": metrics,
            "primary_metric": metrics["F1"],
            "primary_metric_name": "F1"
        }

    if task_type in {"clustering", "anomaly", "transformer", "unsupported"}:
        return {
            "task_type": task_type,
            "metrics": {},
            "primary_metric": None,
            "primary_metric_name": "Not applicable without a task-specific evaluation method"
        }

    numeric_actual = pd.to_numeric(actual, errors="coerce")
    numeric_predictions = pd.to_numeric(predictions, errors="coerce")
    valid = numeric_actual.notna() & numeric_predictions.notna()

    if not valid.any():
        return {
            "task_type": task_type,
            "metrics": {},
            "primary_metric": None,
            "primary_metric_name": "Unavailable"
        }

    mae = mean_absolute_error(numeric_actual[valid], numeric_predictions[valid])
    mse = mean_squared_error(numeric_actual[valid], numeric_predictions[valid])
    rmse = mse ** 0.5
    r2 = r2_score(numeric_actual[valid], numeric_predictions[valid])
    metrics = {
        "MAE": round(float(mae), 3),
        "RMSE": round(float(rmse), 3),
        "R2": round(float(r2), 4)
    }

    return {
        "task_type": "regression",
        "metrics": metrics,
        "primary_metric": metrics["R2"],
        "primary_metric_name": "R2"
    }


def summarize_predictions(predictions, task_type):
    if len(predictions) == 0:
        return None

    prediction_array = np.asarray(predictions)
    if prediction_array.ndim > 1:
        return f"Raw vector output ({prediction_array.shape[1]} values per row); class or score meanings are not exposed."
    values = pd.Series(predictions)
    if task_type == "regression":
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if numeric.empty:
            return "Predictions produced, but they are not numeric."
        return (
            f"mean {numeric.mean():.3f}, min {numeric.min():.3f}, "
            f"max {numeric.max():.3f}"
        )

    counts = values.value_counts(dropna=False).head(5)
    return ", ".join([f"{label}: {count}" for label, count in counts.items()])


def load_reference_data(fallback, features):
    return fallback


def detect_drift(reference, current, features, threshold=0.50):
    if reference is None:
        return {
            "flagged_count": "Unavailable",
            "rows": [{
                "feature": "Reference data",
                "reference_mean": "Unavailable",
                "current_mean": "Unavailable",
                "status": "REFERENCE DATA MISSING"
            }],
            "flagged_ratio": 0
        }

    rows = []
    flagged = 0

    for feature in features:
        if feature not in reference.columns:
            rows.append({
                "feature": feature,
                "reference_mean": "Unavailable",
                "current_mean": "Unavailable",
                "status": "REFERENCE FEATURE MISSING"
            })
            continue

        if feature not in current.columns:
            rows.append({
                "feature": feature,
                "reference_mean": "Unavailable",
                "current_mean": "Unavailable",
                "status": "EVALUATION FEATURE MISSING"
            })
            continue

        ref = pd.to_numeric(reference[feature], errors="coerce").dropna()
        cur = pd.to_numeric(current[feature], errors="coerce").dropna()

        if len(ref) < 2 or len(cur) < 2:
            status = "INSUFFICIENT DATA"
            score = 0
        else:
            ref_std = ref.std()
            score = abs(cur.mean() - ref.mean()) if ref_std == 0 else (
                abs(cur.mean() - ref.mean()) / ref_std
            )
            status = (
                "REVIEW POSSIBLE DRIFT"
                if score > threshold
                else "NO LARGE SHIFT"
            )

        if status == "REVIEW POSSIBLE DRIFT":
            flagged += 1

        rows.append({
            "feature": feature,
            "reference_mean": (
                round(float(ref.mean()), 3) if len(ref) else "Unavailable"
            ),
            "current_mean": (
                round(float(cur.mean()), 3) if len(cur) else "Unavailable"
            ),
            "status": status
        })

    return {
        "flagged_count": flagged,
        "rows": rows,
        "flagged_ratio": flagged / len(features) if features else 0
    }


def detect_ood(reference, current, features):
    if reference is None or current.empty:
        return {
            "row_rate": "Unavailable",
            "rows": [{
                "feature": "Reference data",
                "count": "Unavailable",
                "band": "Unavailable"
            }],
            "row_ratio": 0
        }

    row_flags = pd.Series(False, index=current.index)
    rows = []

    for feature in features:
        if feature not in reference.columns:
            rows.append({
                "feature": feature,
                "count": "Unavailable",
                "band": "Reference feature missing"
            })
            continue

        if feature not in current.columns:
            rows.append({
                "feature": feature,
                "count": "Unavailable",
                "band": "Evaluation feature missing"
            })
            continue

        ref = pd.to_numeric(reference[feature], errors="coerce").dropna()
        cur = pd.to_numeric(current[feature], errors="coerce")

        if ref.empty:
            rows.append({
                "feature": feature,
                "count": "Unavailable",
                "band": "Unavailable"
            })
            continue

        q1 = ref.quantile(0.25)
        q3 = ref.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        flags = (cur < lower) | (cur > upper)
        row_flags = row_flags | flags.fillna(False)

        rows.append({
            "feature": feature,
            "count": int(flags.sum()),
            "band": f"{lower:.3f} to {upper:.3f}"
        })

    row_ratio = float(row_flags.mean()) if len(row_flags) else 0
    return {
        "row_rate": round(row_ratio * 100, 1),
        "rows": rows,
        "row_ratio": row_ratio
    }


def analyze_confidence(model, features_data, predictions, reference, target, profile):
    mean_prediction = "Unavailable"
    if len(predictions):
        prediction_array = np.asarray(predictions)
        if prediction_array.ndim > 1:
            mean_prediction = f"Vector output with {prediction_array.shape[1]} values per row"
        else:
            numeric_predictions = pd.to_numeric(
                pd.Series(predictions), errors="coerce"
            ).dropna()
            if not numeric_predictions.empty:
                mean_prediction = round(float(numeric_predictions.mean()), 3)
            else:
                mean_prediction = "Non-numeric predictions"

    if len(predictions) == 0 or features_data.empty:
        return {
            "status": "Prediction uncertainty unavailable until data validation passes.",
            "mean_prediction": mean_prediction,
            "mean_uncertainty": "Unavailable",
            "high_uncertainty_rate": 0,
            "high_uncertainty_ratio": 0
        }

    if np.asarray(predictions).ndim > 1:
        return {
            "status": "Output vector semantics are not exposed; confidence was not inferred.",
            "mean_prediction": mean_prediction,
            "mean_uncertainty": "Unavailable",
            "high_uncertainty_rate": 0,
            "high_uncertainty_ratio": 0
        }

    if profile["supports_proba"]:
        probabilities = model_predict_proba(model, features_data)
        max_probabilities = np.max(probabilities, axis=1)
        low_confidence_ratio = float((max_probabilities < 0.60).mean())

        return {
            "status": "Estimated from class probabilities.",
            "mean_prediction": mean_prediction,
            "mean_uncertainty": round(float(1 - np.mean(max_probabilities)), 3),
            "high_uncertainty_rate": round(low_confidence_ratio * 100, 1),
            "high_uncertainty_ratio": low_confidence_ratio
        }

    if not hasattr(model, "estimators_"):
        return {
            "status": "Prediction uncertainty unavailable for this model type.",
            "mean_prediction": mean_prediction,
            "mean_uncertainty": "Unavailable",
            "high_uncertainty_rate": 0,
            "high_uncertainty_ratio": 0
        }

    if profile["task_type"] != "regression":
        return {
            "status": "Prediction spread is only enabled for regression ensembles without probabilities.",
            "mean_prediction": mean_prediction,
            "mean_uncertainty": "Unavailable",
            "high_uncertainty_rate": 0,
            "high_uncertainty_ratio": 0
        }

    model_input = features_data.to_numpy()
    tree_predictions = np.array([
        estimator.predict(model_input)
        for estimator in model.estimators_
    ])
    uncertainty = tree_predictions.std(axis=0)
    mean_uncertainty = float(np.mean(uncertainty))

    target_std = None
    if reference is not None and target in reference.columns:
        reference_target = pd.to_numeric(
            reference[target], errors="coerce"
        ).dropna()
        if not reference_target.empty:
            target_std = float(reference_target.std())

    threshold = (
        target_std * 0.30
        if target_std and target_std > 0
        else np.percentile(uncertainty, 75)
    )
    high_ratio = float((uncertainty > threshold).mean())

    return {
        "status": "Estimated from Random Forest prediction spread.",
        "mean_prediction": mean_prediction,
        "mean_uncertainty": round(mean_uncertainty, 3),
        "high_uncertainty_rate": round(high_ratio * 100, 1),
        "high_uncertainty_ratio": high_ratio
    }


def assess_reliability(quality, performance, drift, ood, confidence):
    risk_score = 0
    reasons = []
    recommendations = []

    if quality["issue_count"]:
        risk_score += 4
        reasons.append("the evaluation data has blocking quality issues")
        recommendations.append(
            "Fix missing, invalid, or out-of-range feature values before trusting predictions."
        )

    if performance and performance.get("task_type") == "clustering":
        reasons.append("unsupervised clustering checks were used; labeled metrics do not apply")
        recommendations.append("Review cluster separation and usefulness with domain context.")
    elif performance is None:
        risk_score += 1
        reasons.append("ground-truth labels were not available for performance checks")
        recommendations.append(
            "Add the target column to evaluation data when possible so accuracy can be measured."
        )
    elif performance.get("primary_metric") is None:
        risk_score += 2
        reasons.append("labeled performance could not be interpreted for this model type")
        recommendations.append(
            "Check that the target column uses the same labels or numeric scale as the model output."
        )
    else:
        task_type = performance.get("task_type")
        primary = performance["primary_metric"]
        primary_name = performance["primary_metric_name"]

        if task_type == "regression" and primary < 0.60:
            risk_score += 3
            reasons.append("labeled regression performance is below the reliability threshold")
            recommendations.append(
                "Review model fit and feature relevance because evaluation R2 is below 0.60."
            )
        elif task_type == "classification" and primary < 0.70:
            risk_score += 3
            reasons.append("labeled classification performance is below the reliability threshold")
            recommendations.append(
                "Review class errors and feature relevance because evaluation F1 is below 0.70."
            )
        elif task_type not in {"regression", "classification"} and primary < 0.70:
            risk_score += 2
            reasons.append(f"{primary_name} is below the review threshold")
            recommendations.append(
                "Review the model output mapping before using these predictions in decisions."
            )

    if isinstance(drift["flagged_count"], int) and drift["flagged_count"] > 0:
        risk_score += 2 if drift["flagged_ratio"] < 0.25 else 3
        reasons.append("feature drift was detected")
        recommendations.append(
            "Investigate drifted features and confirm whether the new input distribution is expected."
        )

    if isinstance(ood["row_rate"], (int, float)) and ood["row_ratio"] > 0.10:
        risk_score += 3
        reasons.append("many rows are outside the reference feature ranges")
        recommendations.append(
            "Review OOD rows separately before using their predictions in decisions."
        )

    if confidence["high_uncertainty_ratio"] > 0.20:
        risk_score += 2
        reasons.append("prediction uncertainty is elevated")
        recommendations.append(
            "Route high-uncertainty predictions for manual review or collect more labels."
        )

    if not reasons:
        reasons.append("quality, performance, drift, OOD, and confidence checks are acceptable")
        recommendations.append(
            "Continue monitoring the model with fresh labeled evaluation data."
        )

    if quality["issue_count"]:
        verdict = "INCOMPLETE - Fix Data Issues"
        status_class = "incomplete"
    elif performance is None or performance.get("task_type") == "clustering":
        verdict = "INCOMPLETE - Evaluation Limited"
        status_class = "incomplete"
    elif risk_score >= 4:
        verdict = "RISKY - Review Before Use"
        status_class = "risky"
    else:
        verdict = "RELIABLE - Checks Passed"
        status_class = "reliable"

    assessment = "ModelGuard assessment: " + "; ".join(reasons) + "."

    return verdict, status_class, risk_score, assessment, recommendations


def inspect_model(model, metadata, profile, features=None):
    estimator = model
    if isinstance(model, Pipeline) and model.steps:
        estimator = model.steps[-1][1]
    module_name = type(estimator).__module__.split(".")[0]
    family_name = type(estimator).__name__
    if isinstance(model, TFLiteModelAdapter):
        module_name = "tensorflow-lite"
        family_name = "TensorFlow Lite model"
    elif isinstance(model, KerasModelAdapter):
        module_name = "tensorflow.keras"
        family_name = "Keras model"
    elif isinstance(model, OnnxModelAdapter):
        module_name = "onnxruntime"
        family_name = "ONNX model"
    distribution_aliases = {"sklearn": "scikit-learn", "tensorflow.keras": "tensorflow"}
    version_package = distribution_aliases.get(module_name, module_name)
    if isinstance(model, (TFLiteModelAdapter, KerasModelAdapter)):
        version_package = "tensorflow"
    try:
        framework_version = package_metadata.version(version_package)
    except package_metadata.PackageNotFoundError:
        framework_version = "Unavailable"

    parameters = {}
    get_params = getattr(estimator, "get_params", None)
    if callable(get_params):
        try:
            for key, value in list(get_params(deep=False).items())[:60]:
                if isinstance(value, (str, int, float, bool)) or value is None:
                    display = value
                else:
                    display = repr(value)[:180]
                parameters[str(key)] = display
        except Exception:
            parameters = {}

    preprocessing = []
    if isinstance(model, Pipeline):
        preprocessing = [
            {"step": name, "component": type(step).__name__}
            for name, step in model.steps[:-1]
        ]
    estimator_count = getattr(estimator, "n_estimators", None)
    layer_count = len(getattr(model, "layers", [])) if hasattr(model, "layers") else None
    parameter_count = None
    count_params = getattr(model, "count_params", None)
    if callable(count_params):
        try:
            parameter_count = int(count_params())
        except Exception:
            pass
    architecture = repr(model)[:1800]
    if isinstance(model, KerasModelAdapter):
        try:
            architecture = model.model.to_json()[:1800]
        except Exception:
            pass

    input_details = getattr(model, "inputs", None)
    output_details = getattr(model, "outputs", None)
    if isinstance(model, OnnxModelAdapter):
        input_shapes = [str(detail.shape) for detail in input_details]
        output_shapes = [str(detail.shape) for detail in output_details]
    else:
        input_shapes = [str(detail.get("shape")) for detail in input_details] if input_details else []
        output_shapes = [str(detail.get("shape")) for detail in output_details] if output_details else []
    if not input_shapes:
        shape = getattr(model, "input_shape", None)
        input_shapes = [str(shape)] if shape is not None else []
    if not output_shapes:
        shape = getattr(model, "output_shape", None)
        output_shapes = [str(shape)] if shape is not None else []

    return {
        "family": family_name,
        "framework": module_name,
        "framework_version": framework_version,
        "model_type": profile["task_label"],
        "task_source": metadata.get("task_source", "Inferred from exposed model interface"),
        "interface": profile["interface_label"],
        "feature_count": len(features) if features else getattr(model, "n_features_in_", "Unavailable"),
        "feature_names": list(features or getattr(model, "feature_names_in_", [])),
        "output_classes": [str(value) for value in getattr(model, "classes_", [])] or "Not exposed by this model",
        "output_dimensions": output_shapes or "Not exposed by this model",
        "input_dimensions": input_shapes or "Not exposed by this model",
        "hyperparameters": parameters or "Not exposed by this model",
        "preprocessing": preprocessing or "No saved preprocessing pipeline detected",
        "estimator_count": estimator_count if estimator_count is not None else "Not applicable / unavailable",
        "layer_count": layer_count if layer_count is not None else "Not exposed by this model",
        "parameter_count": parameter_count if parameter_count is not None else "Not exposed by this model",
        "architecture": architecture,
        "metadata_limitations": [
            "Training history, data lineage, and hidden framework-specific metadata cannot be recovered unless they were saved with the model."
        ]
    }


def infer_features(model, metadata, data, target="", configured_features=None):
    candidates = configured_features or metadata.get("features") or getattr(model, "feature_names_in_", None)
    if candidates is not None:
        features = [str(name) for name in candidates]
        missing = [name for name in features if name not in data.columns]
        if missing:
            raise ValueError(f"Input CSV is missing expected model features: {', '.join(missing[:12])}.")
    else:
        features = [column for column in data.columns if column != target]
        expected = getattr(model, "n_features_in_", None)
        if expected is None and isinstance(model, Pipeline):
            expected = getattr(model.steps[0][1], "n_features_in_", None)
        if expected is not None and len(features) != int(expected):
            if len(features) < int(expected):
                raise ValueError(f"Model expects {expected} input features, but the CSV has only {len(features)} after excluding the outcome. Specify the exact features.")
            features = features[:int(expected)]
    if target and target in features:
        features.remove(target)
    if not features:
        raise ValueError("No model input features were found in the CSV.")
    return features


def run_workflow(files, target_column=None, feature_columns=None, selected_row=0, task_override="", trusted_training_id=""):
    reference_data = None
    requested_target = (target_column or "").strip()
    configured_features = parse_feature_columns(feature_columns)

    if trusted_training_id:
        if not re.fullmatch(r"[a-f0-9]{12}", trusted_training_id):
            raise ValueError("Choose a valid ModelGuard training run.")
        model_path = TRAINING_MODEL_DIR / f"{trusted_training_id}.joblib"
        report_path = TRAINING_RUN_DIR / f"{trusted_training_id}.json"
        if not model_path.is_file() or not report_path.is_file():
            raise ValueError("The selected ModelGuard training artifact is unavailable.")
        training_report = json.loads(report_path.read_text(encoding="utf-8"))
        model = joblib.load(model_path)
        metadata = {"model_type": "ModelGuard trained pipeline", "target": training_report.get("target", ""),
                    "source": model_path.name, "features": training_report.get("features"),
                    "task_type": training_report.get("task", "")}
        mode_label = "ModelGuard Training Artifact"
    else:
        model, metadata = load_existing_model(files.get("model_file"))
        mode_label = "Uploaded Model"
    if requested_target:
        metadata["target"] = requested_target
    if task_override in {"classification", "regression"}:
        metadata["task_type"] = task_override
        metadata["task_source"] = "Selected by user"

    evaluation_data = read_csv_upload(
        files.get("evaluation_dataset"),
        "Evaluation data"
    )
    uploaded_reference = read_optional_csv_upload(files.get("reference_dataset"))
    if uploaded_reference is not None:
        reference_data = uploaded_reference

    target = metadata.get("target", "")
    model_profile = detect_model_profile(model, metadata)
    features = infer_features(
        model, metadata, evaluation_data, target, configured_features
    )
    metadata["features"] = features
    metadata["task_type"] = model_profile["task_type"]
    quality = check_quality(evaluation_data, features)
    prediction_summary = None
    predictions_table = None
    prediction_seconds = 0.0

    if quality["clean_features"] is None:
        predictions = np.array([])
        performance = None
        feature_data = evaluation_data[
            [feature for feature in features if feature in evaluation_data.columns]
        ].apply(pd.to_numeric, errors="coerce")
    else:
        feature_data = quality["clean_features"]
        if model_profile["supports_predict"]:
            try:
                prediction_started = perf_counter()
                predictions = model_predict(model, feature_data)
                prediction_seconds = perf_counter() - prediction_started
            except Exception as exc:
                raise ValueError(
                    "Model loaded successfully, but prediction failed. "
                    f"Check feature order, feature count, and preprocessing. Details: {exc}"
                ) from exc
        elif "transform" in model_profile["methods"]:
            transformed = model.transform(feature_data)
            predictions = np.array([])
            shape = getattr(transformed, "shape", None)
            prediction_summary = (
                f"Transformer output shape: {shape}"
                if shape is not None
                else "Transformer ran successfully."
            )
        else:
            predictions = np.array([])
            prediction_summary = (
                "No predict or transform method was detected for this model."
            )

        performance = None
        if model_profile["task_type"] == "clustering" and len(predictions):
            cluster_metrics = {}
            unique_clusters = np.unique(predictions)
            if len(unique_clusters) > 1 and len(unique_clusters) < len(predictions):
                try:
                    metric_data = (
                        model.named_steps["preprocessor"].transform(feature_data)
                        if isinstance(model, Pipeline) else feature_data
                    )
                    sample_limit = min(2000, len(metric_data))
                    cluster_metrics["Silhouette"] = round(float(silhouette_score(
                        metric_data[:sample_limit], predictions[:sample_limit]
                    )), 4)
                except (ValueError, TypeError, AttributeError):
                    pass
            cluster_metrics["Cluster counts"] = {
                str(label): int(count)
                for label, count in pd.Series(predictions).value_counts().sort_index().items()
            }
            performance = {"task_type": "clustering", "metrics": cluster_metrics}
        if (len(predictions) and np.asarray(predictions).ndim == 1
                and target in evaluation_data.columns
                and model_profile["task_type"] in {"classification", "regression"}):
            actual = evaluation_data[target]
            if actual.notna().all() and model_profile["task_type"] != "clustering":
                if model_profile["task_type"] != "clustering":
                    performance = calculate_metrics(actual, predictions, model_profile["task_type"])
                    if performance and performance.get("task_type") == "classification":
                        labels = sorted(set(actual.tolist()) | set(np.asarray(predictions).tolist()), key=str)
                        performance["confusion_matrix"] = {
                            "labels": [str(label) for label in labels],
                            "values": confusion_matrix(actual, predictions, labels=labels).tolist()
                        }
                        performance["class_metrics"] = classification_report(
                            actual, predictions, labels=labels, output_dict=True,
                            zero_division=0
                        )
                        if model_profile["supports_proba"]:
                            try:
                                probabilities = model_predict_proba(model, feature_data)
                                if len(labels) == 2:
                                    performance["metrics"]["ROC-AUC"] = round(float(roc_auc_score(
                                        actual, probabilities[:, 1]
                                    )), 4)
                                elif probabilities.shape[1] == len(labels):
                                    performance["metrics"]["ROC-AUC (OVR)"] = round(float(roc_auc_score(
                                        actual, probabilities, multi_class="ovr", labels=labels
                                    )), 4)
                            except (ValueError, TypeError, IndexError):
                                pass

        if prediction_summary is None:
            prediction_summary = summarize_predictions(
                predictions, model_profile["task_type"]
            )

        if len(predictions):
            predictions_table = pd.DataFrame({
                "row_index": evaluation_data.index,
                "prediction": predictions
            })
            if target in evaluation_data.columns:
                predictions_table["actual"] = evaluation_data[target].to_numpy()
            if model_profile["supports_proba"]:
                try:
                    row_probabilities = model_predict_proba(model, feature_data)
                    output_estimator = model
                    if isinstance(model, Pipeline) and model.steps:
                        output_estimator = model.steps[-1][1]
                    output_classes = list(getattr(output_estimator, "classes_", range(row_probabilities.shape[1])))
                    for class_index in range(row_probabilities.shape[1]):
                        class_name = output_classes[class_index] if class_index < len(output_classes) else class_index
                        predictions_table[f"probability_{class_name}"] = row_probabilities[:, class_index]
                except (ValueError, TypeError, IndexError):
                    pass

    reference = load_reference_data(reference_data, features)
    drift = detect_drift(reference, evaluation_data, features)
    ood = detect_ood(reference, evaluation_data, features)
    confidence = analyze_confidence(
        model, feature_data, predictions, reference, target, model_profile
    )
    try:
        selected_row = int(selected_row)
    except (TypeError, ValueError) as exc:
        raise ValueError("Selected row must be a whole number starting at 1.") from exc
    if selected_row < 0 or selected_row >= len(feature_data):
        raise ValueError(f"Selected row must be between 1 and {len(feature_data)}.")
    explanation = explain_predictions(
        model, features, feature_data, predictions, reference, selected_row,
        model_profile["task_type"]
    )
    model_information = inspect_model(model, metadata, model_profile, features)
    diagnostics = {
        "prediction_latency_ms_per_row": round(prediction_seconds * 1000 / max(len(evaluation_data), 1), 3),
        "prediction_latency_ms_total": round(prediction_seconds * 1000, 3),
        "unexpected_columns": [column for column in evaluation_data.columns if column not in features and column != target]
    }

    verdict, status_class, risk_score, assessment, recommendations = (
        assess_reliability(quality, performance, drift, ood, confidence)
    )

    run_id = uuid.uuid4().hex[:12]
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = {
        "run_id": run_id,
        "created_at": created_at,
        "mode_label": mode_label,
        "model_name": metadata.get("model_type", type(model).__name__),
        "model_profile": model_profile,
        "features": features,
        "target": target,
        "rows": len(evaluation_data),
        "selected_row": selected_row + 1,
        "selected_prediction": (
            json_safe(predictions[selected_row]) if len(predictions) > selected_row else None
        ),
        "selected_probabilities": (
            json_safe({
                str(column)[len("probability_"):]: predictions_table.iloc[selected_row][column]
                for column in predictions_table.columns
                if str(column).startswith("probability_")
            })
            if predictions_table is not None and len(predictions_table) > selected_row else {}
        ),
        "selected_inputs": (
            json_safe(feature_data.iloc[selected_row].to_dict())
            if len(feature_data) > selected_row else {}
        ),
        "data_profile": profile_dataset(evaluation_data),
        "dataset_type": "model inspection",
        "explainability": explanation,
        "model_information": model_information,
        "diagnostics": diagnostics,
        "quality": {
            key: value for key, value in quality.items()
            if key != "clean_features"
        },
        "performance": performance,
        "prediction_summary": prediction_summary,
        "prediction_rows": (
            json_safe(predictions_table.head(100).to_dict("records"))
            if predictions_table is not None else []
        ),
        "drift": drift,
        "ood": ood,
        "confidence": confidence,
        "verdict": verdict,
        "status_class": status_class,
        "risk_score": risk_score,
        "assessment": assessment,
        "explanation": assessment,
        "recommendations": recommendations
    }
    save_workflow_artifacts(result, predictions_table)
    return result


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "ModelGuard", "status": "ready"})


@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")


@app.route("/image-test", methods=["GET", "POST"])
def image_test():
    # Bound this endpoint before accessing form/files, without changing CSV uploads.
    request.max_content_length = 24 * 1024 * 1024
    try:
        from .image_testing import run_image_test
    except ImportError:
        from image_testing import run_image_test

    result = None
    error = None
    form_values = {
        "labels": request.form.get("labels", ""),
        "resize_strategy": request.form.get("resize_strategy", ""),
        "preprocessing": request.form.get("preprocessing", "auto"),
        "color_order": request.form.get("color_order", "rgb"),
        "output_interpretation": request.form.get("output_interpretation", "auto"),
        "true_label": request.form.get("true_label", ""),
    }
    if request.method == "POST":
        try:
            result = run_image_test(
                request.files.get("model_file"), request.files.get("image_file"),
                form_values["labels"], form_values["resize_strategy"], form_values["preprocessing"],
                form_values["output_interpretation"], form_values["color_order"], form_values["true_label"]
            )
            # Reflect any exact-model profile that corrected incompatible submitted settings.
            form_values["preprocessing"] = result["preprocessing_used"]
            form_values["resize_strategy"] = result["resize_strategy_used"]
            form_values["color_order"] = result["color_order"].lower()
            form_values["output_interpretation"] = result["output_interpretation_used"]
        except Exception as exc:
            error = public_error_message(exc)
    return render_template("image_test.html", result=result, error=error, form_values=form_values)


@app.errorhandler(413)
def upload_too_large(_error):
    if request.endpoint == "image_test":
        return render_template(
            "image_test.html", result=None,
            error="The combined upload is too large. Keep the model and image under 24 MB total.",
            form_values={"labels": "", "resize_strategy": "", "preprocessing": "auto", "color_order": "rgb", "output_interpretation": "auto", "true_label": ""}
        ), 413
    return "Upload too large.", 413


@app.route("/check", methods=["GET", "POST"])
def check_model():
    result = None
    error = None
    target_column = request.form.get("target_column", "")
    feature_columns = request.form.get("feature_columns", "")
    selected_row = request.form.get("selected_row", "1")
    task_override = request.form.get("task_override", "")
    trusted_training_id = request.form.get("trusted_training_id", "")
    if request.method == "POST":
        try:
            result = run_workflow(
                request.files, target_column, feature_columns,
                int(selected_row) - 1, task_override, trusted_training_id
            )
        except Exception as exc:
            error = public_error_message(exc)
    return render_template(
        "check.html", result=result, error=error,
        target_column=target_column, feature_columns=feature_columns,
        selected_row=selected_row, task_override=task_override,
        training_runs=recent_training_runs()
    )


@app.route("/train", methods=["GET", "POST"])
def train_model():
    error = None
    result = None
    if request.method == "POST":
        try:
            dataset = read_csv_upload(request.files.get("dataset"), "Training dataset")
            if dataset.empty or len(dataset.columns) < 2:
                raise ValueError("The CSV needs at least one feature column and data rows.")
            options = {key: request.form.get(key, default) for key, default in {
                "n_estimators": "150", "max_depth": "", "n_clusters": "3", "eps": ".5",
                "min_samples": "5", "split": "random", "time_column": "", "layers": "64,32",
                "activation": "relu", "learning_rate": ".001", "batch_size": "32",
                "epochs": "200", "dropout": ".0001"}.items()}
            pipeline, report, predictions = train_dataset(
                dataset, request.form.get("target", ""), request.form.get("task", "auto"),
                request.form.get("algorithm", "auto"), options)
            run_id = uuid.uuid4().hex[:12]
            TRAINING_RUN_DIR.mkdir(parents=True, exist_ok=True)
            TRAINING_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model_path = TRAINING_MODEL_DIR / f"{run_id}.joblib"
            report_path = TRAINING_RUN_DIR / f"{run_id}.json"
            predictions_path = TRAINING_RUN_DIR / f"{run_id}_predictions.csv"
            joblib.dump(pipeline, model_path, compress=3)
            report.update({"run_id": run_id, "created_at": datetime.now().isoformat(timespec="seconds"),
                           "model_file": model_path.name, "prediction_file": predictions_path.name})
            report_path.write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
            predictions.to_csv(predictions_path, index=True, index_label="row")
            result = report
        except Exception as exc:
            error = public_error_message(exc)
    return render_template("train.html", error=error, result=result,
                           history=recent_training_runs())


@app.route("/train/preview", methods=["POST"])
def preview_training_data():
    try:
        data = read_csv_upload(request.files.get("dataset"), "Training dataset")
        if data.empty or len(data.columns) < 2:
            raise ValueError("The CSV needs at least two columns and some data rows.")
        return jsonify(training_profile(data))
    except Exception as exc:
        return jsonify({"error": public_error_message(exc)}), 400


@app.route("/train/download/<run_id>/<kind>")
def download_training_file(run_id, kind):
    if not re.fullmatch(r"[a-f0-9]{12}", run_id):
        abort(404)
    paths = {
        "model": (TRAINING_MODEL_DIR / f"{run_id}.joblib", "model.joblib"),
        "report": (TRAINING_RUN_DIR / f"{run_id}.json", "training_report.json"),
        "predictions": (TRAINING_RUN_DIR / f"{run_id}_predictions.csv", "holdout_predictions.csv"),
    }
    if kind not in paths or not paths[kind][0].is_file():
        abort(404)
    return send_file(paths[kind][0], as_attachment=True, download_name=paths[kind][1])


def recent_training_runs():
    history = []
    if TRAINING_RUN_DIR.exists():
        for path in sorted(TRAINING_RUN_DIR.glob("[a-f0-9]*.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:10]:
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                if re.fullmatch(r"[a-f0-9]{12}", item.get("run_id", "")):
                    history.append(item)
            except (OSError, ValueError):
                continue
    return history


@app.route("/download/report/<run_id>")
def download_report(run_id):
    safe_run_id = Path(run_id).stem
    path = WORKFLOW_REPORT_DIR / f"{safe_run_id}.json"
    if not path.exists():
        abort(404)
    return send_file(path, as_attachment=True, download_name=f"{safe_run_id}.json")


@app.route("/download/predictions/<run_id>")
def download_predictions(run_id):
    safe_run_id = Path(run_id).stem
    path = WORKFLOW_PREDICTION_DIR / f"{safe_run_id}.csv"
    if not path.exists():
        abort(404)
    return send_file(path, as_attachment=True, download_name=f"{safe_run_id}_predictions.csv")


if __name__ == "__main__":
    app.run(debug=False)
