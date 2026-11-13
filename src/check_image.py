from pathlib import Path
from PIL import Image

ROOT = Path(
    r".\dataset\plantvillage\PlantVillage\PlantVillage"
)

extensions = {".jpg", ".jpeg", ".png"}

files = [
    p for p in ROOT.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
]

bad_files = []

print(f"Checking {len(files)} images...\n")

for i, path in enumerate(files, start=1):
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as e:
        bad_files.append((path, str(e)))

    if i % 2000 == 0:
        print(f"Checked {i}/{len(files)}")

print("\n========== RESULT ==========")
print(f"Total images checked : {len(files)}")
print(f"Corrupted/unreadable : {len(bad_files)}")

if bad_files:
    print("\nProblem files:")
    for path, error in bad_files:
        print(f"{path} -> {error}")
else:
    print("\nAll images are readable.")