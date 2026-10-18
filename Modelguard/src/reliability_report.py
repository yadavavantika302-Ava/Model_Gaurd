
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_METADATA = ROOT / "models" / "model_metadata.json"

QUALITY_REPORT = ROOT / "reports" / "data_quality_report.txt"
DRIFT_REPORT = ROOT / "reports" / "drift_report.txt"
EVALUATION_REPORT = ROOT / "reports" / "test_evaluation_report.txt"

OUTPUT_DIR = ROOT / "reports"


def read_report(path):
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def read_test_metrics(text):
    metrics = {}

    for line in text.splitlines():
        parts = line.split(":", 1)

        if len(parts) != 2:
            continue

        key = parts[0].strip()
        value = parts[1].strip()

        if key in ["MAE", "RMSE", "R2"]:
            try:
                metrics[key] = float(value)
            except ValueError:
                pass

    return metrics


def main():
    if not MODEL_METADATA.exists():
        raise FileNotFoundError(
            "Model metadata missing. Train the model first."
        )

    with open(
        MODEL_METADATA, "r", encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    quality = read_report(QUALITY_REPORT)
    drift = read_report(DRIFT_REPORT)
    evaluation = read_report(EVALUATION_REPORT)

    # Do not claim an overall assessment if evidence is missing.
    missing_reports = []

    if quality is None:
        missing_reports.append("Data quality report")

    if drift is None:
        missing_reports.append("Data drift report")

    if evaluation is None:
        missing_reports.append("Test evaluation report")

    lines = [
        "MODEL GUARD - RELIABILITY ASSESSMENT",
        "=" * 42,
        f"Model: {metadata.get('model_type', 'Unknown')}",
        f"Target: {metadata.get('target', 'Unknown')}",
        f"Features: {len(metadata.get('features', []))}",
        "",
        "ASSESSMENT STATUS:"
    ]

    if missing_reports:
        lines.append(
            "INCOMPLETE - required reports are missing."
        )
        for item in missing_reports:
            lines.append(f"- Missing: {item}")
    else:
        lines.append(
            "Evidence collected. Review the sections below."
        )

    lines.extend([
        "",
        "1. DATA QUALITY",
        "-" * 20
    ])

    lines.append(
        quality if quality else "Not available. Run data_quality_checker.py."
    )

    lines.extend([
        "",
        "2. DATA DRIFT",
        "-" * 20
    ])

    lines.append(
        drift if drift else "Not available. Run drift_detector.py."
    )

    lines.extend([
        "",
        "3. UNSEEN TEST PERFORMANCE",
        "-" * 28
    ])

    lines.append(
        evaluation if evaluation
        else "Not available. Run evaluate_model.py."
    )

    if evaluation:
        metrics = read_test_metrics(evaluation)

        lines.extend([
            "",
            "METRIC INTERPRETATION",
            "-" * 20
        ])

        if "MAE" in metrics:
            lines.append(
                f"MAE: {metrics['MAE']:.3f} rentals per hour "
                "(average absolute error)."
            )

        if "RMSE" in metrics:
            lines.append(
                f"RMSE: {metrics['RMSE']:.3f} rentals per hour "
                "(penalizes larger errors)."
            )

        if "R2" in metrics:
            lines.append(
                f"R2: {metrics['R2']:.4f} "
                "(explained variation on this test period)."
            )

    lines.extend([
        "",
        "IMPORTANT LIMITATIONS",
        "-" * 22,
        "- This report does not guarantee model reliability.",
        "- Data drift alone does not prove poor predictions.",
        "- Review metrics, data quality, and context together.",
        "- Evaluate with ground-truth labels whenever possible.",
        "- Missing evidence must not be treated as a passing check."
    ])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "reliability_assessment.txt"

    output_file.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("\n".join(lines))
    print("\nFinal report saved to:", output_file)


if __name__ == "__main__":
    main()
