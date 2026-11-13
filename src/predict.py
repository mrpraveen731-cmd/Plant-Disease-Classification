
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .resnet_model import create_resnet18


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_resnet18_model.pth"

DEVICE = torch.device("cpu")

# Load the checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

class_names = checkpoint["class_names"]

# Create the ResNet18 architecture and load trained weights
model = create_resnet18(
    num_classes=len(class_names),
    pretrained=False,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

# Image preprocessing must match the training pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def predict_image(image: Image.Image):
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_index = torch.max(
            probabilities, dim=1
        )

    return {
        "predicted_disease": class_names[predicted_index.item()],
        "confidence": round(confidence.item(), 4),
    }
