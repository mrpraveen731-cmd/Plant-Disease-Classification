from pathlib import Path
from collections import Counter

ROOT = Path(
    r".\dataset\plantvillage\PlantVillage\PlantVillage"
)

extensions = {".jpg", ".jpeg", ".png"}

class_counts = Counter()

for class_folder in ROOT.iterdir():
    if class_folder.is_dir():
        count = sum(
            1
            for file in class_folder.iterdir()
            if file.is_file() and file.suffix.lower() in extensions
        )
        class_counts[class_folder.name] = count

print("\n========== DATASET SUMMARY ==========\n")

total = sum(class_counts.values())

for class_name, count in sorted(class_counts.items()):
    percentage = (count / total) * 100
    print(f"{class_name:55} {count:5} ({percentage:5.2f}%)")

print("\n--------------------------------------")
print(f"Total classes  : {len(class_counts)}")
print(f"Total images   : {total}")
print("--------------------------------------")