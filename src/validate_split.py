
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "reports" / "dataset_split.csv"

df = pd.read_csv(MANIFEST)

required_columns = {"image_path", "label", "split"}
assert required_columns.issubset(df.columns), "Required columns are missing."

# 1. Confirm expected splits
expected_splits = {"train", "validation", "test"}
assert set(df["split"].unique()) == expected_splits, "Unexpected split names."

# 2. Confirm every image path exists
missing = [
    path for path in df["image_path"]
    if not (PROJECT_ROOT / path).is_file()
]

# 3. Check duplicate paths
duplicate_paths = int(df["image_path"].duplicated().sum())

# 4. Check whether an image path appears in multiple splits
split_counts = df.groupby("image_path")["split"].nunique()
cross_split_paths = int((split_counts > 1).sum())

# 5. Check label distribution in every split
distribution = pd.crosstab(df["label"], df["split"])

print("========== SPLIT VALIDATION ==========")
print(f"Manifest rows: {len(df)}")
print(f"Unique classes: {df['label'].nunique()}")
print(f"Missing files: {len(missing)}")
print(f"Duplicate paths: {duplicate_paths}")
print(f"Paths across multiple splits: {cross_split_paths}")

print("\nSplit totals:")
print(df["split"].value_counts().to_string())

if missing:
    print("\nFirst missing paths:")
    for path in missing[:10]:
        print(path)

assert not missing, "Some image files are missing."
assert duplicate_paths == 0, "Duplicate image paths found."
assert cross_split_paths == 0, "An image path occurs in multiple splits."

print("\nPASS: Manifest checks completed.")
