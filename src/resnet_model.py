
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def create_resnet18(num_classes=15, pretrained=True):
    if pretrained:
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    else:
        model = resnet18(weights=None)

    # Freeze the pretrained feature extractor
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Replace the final layer for our 15 plant classes
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)

    return model


if __name__ == "__main__":
    model = create_resnet18(num_classes=15, pretrained=False)

    sample = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        output = model(sample)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Model: ResNet18")
    print("Input shape:", tuple(sample.shape))
    print("Output shape:", tuple(output.shape))
    print("Trainable parameters:", trainable_parameters)
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
