
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "reports" / "dataset_split.csv"

IMAGE_SIZE = 224
BATCH_SIZE = 16
RANDOM_SEED = 42

# Read the existing manifest
df = pd.read_csv(MANIFEST_PATH)

# Keep one consistent class-to-index mapping for all splits
class_names = sorted(df["label"].unique())
class_to_idx = {
    class_name: index
    for index, class_name in enumerate(class_names)
}

# Convert images to tensors suitable for model input
image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


class PlantDiseaseDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = PROJECT_ROOT / row["image_path"]
        label_name = row["label"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = class_to_idx[label_name]

        return image, label


# Separate the existing splits
train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "validation"]
test_df = df[df["split"] == "test"]

train_dataset = PlantDiseaseDataset(train_df, image_transform)
val_dataset = PlantDiseaseDataset(val_df, image_transform)
test_dataset = PlantDiseaseDataset(test_df, image_transform)

# Seed the training shuffle for reproducibility
generator = torch.Generator()
generator.manual_seed(RANDOM_SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    generator=generator,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


if __name__ == "__main__":
    images, labels = next(iter(train_loader))

    print("========== DATA PIPELINE CHECK ==========")
    print(f"Classes: {len(class_names)}")
    print(f"Class names: {class_names}")
    print(f"Train images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Test images: {len(test_dataset)}")
    print(f"Batch image shape: {tuple(images.shape)}")
    print(f"Batch label shape: {tuple(labels.shape)}")
    print(f"Pixel tensor dtype: {images.dtype}")
    print(f"Label dtype: {labels.dtype}")
    print(f"First batch labels: {labels.tolist()}")
