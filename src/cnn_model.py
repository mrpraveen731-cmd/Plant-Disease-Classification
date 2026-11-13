
import torch
import torch.nn as nn


class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 channels -> 32 feature maps
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 2: 32 -> 64 feature maps
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 3: 64 -> 128 feature maps
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = PlantDiseaseCNN(num_classes=15)
    sample = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        output = model(sample)

    print("Model:", model.__class__.__name__)
    print("Input shape:", tuple(sample.shape))
    print("Output shape:", tuple(output.shape))
    print("Trainable parameters:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))
