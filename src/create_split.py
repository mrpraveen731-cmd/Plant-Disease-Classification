
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Project root: one level above src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "plantvillage"
    / "PlantVillage"
    / "PlantVillage"
)

OUTPUT_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
RANDOM_STATE = 42

records = []

for class_folder in sorted(DATASET_ROOT.iterdir()):
    if not class_folder.is_dir():
        continue

    for image_path in class_folder.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS:
            records.append({
                "image_path": str(image_path.relative_to(PROJECT_ROOT)),
                "label": class_folder.name,
            })

df = pd.DataFrame(records)

if df.empty:
    raise RuntimeError(f"No images found at: {DATASET_ROOT}")

# 80% train, 20% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df["label"],
)

# Divide the temporary 20% equally into validation and test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["label"],
)

train_df = train_df.assign(split="train")
val_df = val_df.assign(split="validation")
test_df = test_df.assign(split="test")

split_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True,
)

output_path = OUTPUT_DIR / "dataset_split.csv"
split_df.to_csv(output_path, index=False)

print("\n========== DATASET SPLIT ==========")
print(f"Total images: {len(split_df)}")
print(f"Classes: {split_df['label'].nunique()}")
print("\nImages per split:")
print(split_df["split"].value_counts().to_string())

print("\nImages per split and class:")
print(
    pd.crosstab(split_df["label"], split_df["split"])
    .to_string()
)

print(f"\nManifest saved to: {output_path}")
