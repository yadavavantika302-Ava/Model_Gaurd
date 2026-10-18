
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = ROOT / "models" / "bike_rental_model.joblib"
TEST_FILE = ROOT / "data" / "processed" / "test_data.csv"
METADATA_FILE = ROOT / "models" / "model_metadata.json"
REPORT_DIR = ROOT / "reports"

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        "Pre-trained bike rental model not found at the configured models path."
    )

if not TEST_FILE.exists():
    raise FileNotFoundError(
        "Test data not found. Run prepare_data.py first."
    )

if not METADATA_FILE.exists():
    raise FileNotFoundError(
        "Model metadata not found at the configured models path."
    )

model = joblib.load(MODEL_FILE)

with open(METADATA_FILE, "r", encoding="utf-8") as file:
    metadata = json.load(file)

features = metadata["features"]
target = metadata["target"]

df = pd.read_csv(TEST_FILE)

required_columns = features + [target]
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if df[required_columns].isna().any().any():
    raise ValueError(
        "Missing values found in the test dataset."
    )

X_test = df[features]
y_test = df[target]

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

results = pd.DataFrame({
    "actual_cnt": y_test.to_numpy(),
    "predicted_cnt": predictions.round(2)
})

REPORT_DIR.mkdir(parents=True, exist_ok=True)

results.to_csv(
    REPORT_DIR / "test_predictions.csv",
    index=False
)

report = f"""
MODEL GUARD - UNSEEN TEST EVALUATION
====================================
Model: {metadata['model_type']}
Test rows: {len(df)}

MAE: {mae:.3f}
RMSE: {rmse:.3f}
R2: {r2:.4f}

These metrics describe performance on this test period.
They do not guarantee performance on future data.
"""

with open(
    REPORT_DIR / "test_evaluation_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)

print(report)
print("Saved predictions to:", REPORT_DIR / "test_predictions.csv")
print("Saved report to:", REPORT_DIR / "test_evaluation_report.txt")

print("\nFirst 10 predictions:")
print(results.head(10).to_string(index=False))
