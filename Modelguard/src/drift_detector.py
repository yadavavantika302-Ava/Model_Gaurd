
from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

METADATA_FILE = ROOT / "models" / "model_metadata.json"
REFERENCE_FILE = ROOT / "data" / "processed" / "development_data.csv"
CURRENT_FILE = ROOT / "data" / "processed" / "test_data.csv"
REPORT_DIR = ROOT / "reports"


def detect_drift(reference, current, features, threshold=0.10):
    results = []

    for feature in features:
        if feature not in reference.columns:
            results.append({
                "feature": feature,
                "status": "REFERENCE FEATURE MISSING"
            })
            continue

        if feature not in current.columns:
            results.append({
                "feature": feature,
                "status": "CURRENT FEATURE MISSING"
            })
            continue

        ref = pd.to_numeric(
            reference[feature], errors="coerce"
        ).dropna()

        cur = pd.to_numeric(
            current[feature], errors="coerce"
        ).dropna()

        if len(ref) < 2 or len(cur) < 2:
            results.append({
                "feature": feature,
                "status": "INSUFFICIENT DATA"
            })
            continue

        # Compare means after scaling by the reference
        # standard deviation. This is a screening indicator,
        # not a formal statistical significance test.
        ref_std = ref.std()

        if ref_std == 0:
            difference = abs(cur.mean() - ref.mean())
            drift_score = 0.0 if difference == 0 else float("inf")
        else:
            drift_score = abs(cur.mean() - ref.mean()) / ref_std

        drift_found = drift_score > threshold

        results.append({
            "feature": feature,
            "reference_mean": round(float(ref.mean()), 4),
            "current_mean": round(float(cur.mean()), 4),
            "standardized_mean_difference": (
                "infinity"
                if not np.isfinite(drift_score)
                else round(float(drift_score), 4)
            ),
            "status": (
                "REVIEW POSSIBLE DRIFT"
                if drift_found
                else "NO LARGE MEAN SHIFT DETECTED"
            )
        })

    return results


def main():
    for path in [METADATA_FILE, REFERENCE_FILE, CURRENT_FILE]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    features = metadata["features"]

    reference = pd.read_csv(REFERENCE_FILE)
    current = pd.read_csv(CURRENT_FILE)

    results = detect_drift(reference, current, features)

    report = [
        "MODEL GUARD - DATA DRIFT REPORT",
        "=" * 36,
        f"Reference rows: {len(reference)}",
        f"Current rows: {len(current)}",
        "Method: Standardized difference in feature means",
        "Screening threshold: 0.10",
        "",
        "FEATURE RESULTS:"
    ]

    for item in results:
        report.append(f"\nFeature: {item['feature']}")
        for key, value in item.items():
            if key != "feature":
                report.append(f"  {key}: {value}")

    report.extend([
        "",
        "INTERPRETATION:",
        "A flagged feature needs investigation.",
        "A flag does not prove the model is inaccurate.",
        "No flag does not prove the model is reliable.",
        "This report checks mean shifts, not every type of drift."
    ])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = REPORT_DIR / "drift_report.txt"
    output_file.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print("\n".join(report))
    print("\nReport saved to:", output_file)


if __name__ == "__main__":
    main()
