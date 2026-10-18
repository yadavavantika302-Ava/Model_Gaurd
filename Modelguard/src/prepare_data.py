
from pathlib import Path
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "raw" / "hour.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_FILE}"
    )

# Load the real dataset
df = pd.read_csv(INPUT_FILE)

# Required columns for this experiment
required = [
    "dteday", "hr", "season", "yr", "mnth",
    "holiday", "weekday", "workingday",
    "weathersit", "temp", "atemp", "hum",
    "windspeed", "cnt"
]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Keep only fields needed for this experiment
df = df[required].copy()

# Validate and parse dates
df["dteday"] = pd.to_datetime(
    df["dteday"], errors="coerce"
)

if df["dteday"].isna().any():
    raise ValueError("Invalid dates found. Inspect raw data first.")

# Sort chronologically
df = df.sort_values(
    ["dteday", "hr"]
).reset_index(drop=True)

# Basic target checks
if df["cnt"].isna().any():
    raise ValueError("Target cnt contains missing values.")

if (df["cnt"] < 0).any():
    raise ValueError("Negative rental counts found.")

# Check for duplicate date-hour records
if df.duplicated(["dteday", "hr"]).any():
    raise ValueError("Duplicate date-hour records found.")

# Exclude the final 20% of time-ordered rows for testing.
# The first 80% remain for training/validation.
split_index = int(len(df) * 0.8)

development = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

# Separate features and target.
# Do not include the date as a raw model feature.
feature_columns = [
    "hr", "season", "yr", "mnth",
    "holiday", "weekday", "workingday",
    "weathersit", "temp", "atemp",
    "hum", "windspeed"
]

X_development = development[feature_columns]
y_development = development["cnt"]

X_test = test[feature_columns]
y_test = test["cnt"]

# Save reproducible outputs
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

development.to_csv(
    OUTPUT_DIR / "development_data.csv", index=False
)
test.to_csv(
    OUTPUT_DIR / "test_data.csv", index=False
)

print("Dataset preparation completed.")
print("Total rows:", len(df))
print("Development rows:", len(development))
print("Test rows:", len(test))
print("Features:", feature_columns)
print("Training target: cnt")
print("First test date:", test["dteday"].min())
print("Last test date:", test["dteday"].max())
print("Saved files:", OUTPUT_DIR)
