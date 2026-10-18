
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METADATA_FILE = ROOT / "models" / "model_metadata.json"
TEST_FILE = ROOT / "data" / "processed" / "test_data.csv"
REPORT_DIR = ROOT / "reports"


def check_data_quality(data):
    issues = []
    warnings = []

    # Load the features expected by the trained model
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            "Model metadata not found. Train the model first."
        )

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    required_features = metadata["features"]

    # 1. Check required columns
    missing_columns = [
        col for col in required_features
        if col not in data.columns
    ]

    if missing_columns:
        issues.append(
            f"Missing required features: {missing_columns}"
        )

    # Continue only with checks that can be performed
    available_features = [
        col for col in required_features
        if col in data.columns
    ]

    if not available_features:
        issues.append("No required model features are available.")
        return issues, warnings

    features_data = data[available_features]

    # 2. Check missing values
    missing_values = features_data.isna().sum()
    missing_values = missing_values[missing_values > 0]

    if not missing_values.empty:
        issues.append(
            "Missing values: " + str(missing_values.to_dict())
        )

    # 3. Check numeric data types
    non_numeric = [
        col for col in available_features
        if not pd.api.types.is_numeric_dtype(data[col])
    ]

    if non_numeric:
        issues.append(
            f"Non-numeric features found: {non_numeric}"
        )

    # 4. Check duplicate rows
    duplicates = int(data.duplicated().sum())

    if duplicates > 0:
        warnings.append(
            f"Found {duplicates} duplicate rows."
        )

    # 5. Check infinite values and invalid ranges
    numeric_data = features_data.select_dtypes(
        include="number"
    )

    if not numeric_data.empty:
        infinite_count = int(
            numeric_data.isin([float("inf"), float("-inf")])
            .sum()
            .sum()
        )

        if infinite_count > 0:
            issues.append(
                f"Found {infinite_count} infinite values."
            )

    # Bike Sharing feature-specific range checks
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

    for col, (minimum, maximum) in range_limits.items():
        if col in available_features and pd.api.types.is_numeric_dtype(data[col]):
            invalid = (
                (data[col] < minimum) |
                (data[col] > maximum)
            ).sum()

            if invalid > 0:
                issues.append(
                    f"{col}: {invalid} values outside "
                    f"the expected range [{minimum}, {maximum}]."
                )

    return issues, warnings


def main():
    if not TEST_FILE.exists():
        raise FileNotFoundError(
            "Test data not found. Run prepare_data.py first."
        )

    data = pd.read_csv(TEST_FILE)

    issues, warnings = check_data_quality(data)

    status = (
        "CHECKS PASSED"
        if not issues
        else "ISSUES FOUND"
    )

    report = [
        "MODEL GUARD - DATA QUALITY REPORT",
        "=" * 38,
        f"Rows checked: {len(data)}",
        f"Columns received: {len(data.columns)}",
        f"Status: {status}",
        "",
        "ISSUES:"
    ]

    report.extend(
        [f"- {item}" for item in issues] or ["- None found"]
    )

    report.append("")
    report.append("WARNINGS:")

    report.extend(
        [f"- {item}" for item in warnings] or ["- None found"]
    )

    report_text = "\n".join(report)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = REPORT_DIR / "data_quality_report.txt"
    output_file.write_text(report_text, encoding="utf-8")

    print(report_text)
    print("\nReport saved to:", output_file)


if __name__ == "__main__":
    main()
