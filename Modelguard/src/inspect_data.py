
from pathlib import Path
import pandas as pd

# Find the main project folder
project_folder = Path(__file__).resolve().parents[1]

# Dataset and report paths
data_path = project_folder / "data" / "raw" / "hour.csv"
report_folder = project_folder / "reports"

# Check whether the dataset exists
if not data_path.exists():
    print("ERROR: hour.csv was not found.")
    print("Place it inside data/raw/")
    raise SystemExit(1)

# Load the real dataset
df = pd.read_csv(data_path)

print("\n===== MODELGUARD DATA INSPECTION =====")

# 1. Dataset size
print("\n1. Dataset shape:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 2. Column names
print("\n2. Column names:")
print(df.columns.tolist())

# 3. First five records
print("\n3. First five records:")
print(df.head())

# 4. Data types
print("\n4. Data types:")
print(df.dtypes)

# 5. Missing values
print("\n5. Missing values per column:")
print(df.isnull().sum())

# 6. Duplicate rows
print("\n6. Duplicate rows:")
print(df.duplicated().sum())

# 7. Numerical summary
print("\n7. Numerical summary:")
print(df.describe().T)

# 8. Validate the target column
print("\n8. Target validation:")
if "cnt" in df.columns:
    print("Target column 'cnt' exists.")
    print("Negative rental counts:", (df["cnt"] < 0).sum())
else:
    print("ERROR: Target column 'cnt' is missing.")

# 9. Check date range
print("\n9. Time range:")
if "dteday" in df.columns:
    dates = pd.to_datetime(df["dteday"], errors="coerce")
    print("Earliest date:", dates.min())
    print("Latest date:", dates.max())
    print("Invalid dates:", dates.isna().sum())
else:
    print("Date column 'dteday' is missing.")

# 10. Save inspection report
report_folder.mkdir(parents=True, exist_ok=True)

report_path = report_folder / "dataset_quality_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("MODELGUARD DATASET QUALITY REPORT\n")
    file.write("=" * 35 + "\n\n")
    file.write(f"Rows: {df.shape[0]}\n")
    file.write(f"Columns: {df.shape[1]}\n\n")
    file.write("Column names:\n")
    file.write(str(df.columns.tolist()) + "\n\n")
    file.write("Data types:\n")
    file.write(df.dtypes.to_string() + "\n\n")
    file.write("Missing values:\n")
    file.write(df.isnull().sum().to_string() + "\n\n")
    file.write(f"Duplicate rows: {df.duplicated().sum()}\n\n")
    file.write("Numerical summary:\n")
    file.write(df.describe().T.to_string() + "\n")

print("\nInspection completed.")
print("Report saved to:", report_path)
