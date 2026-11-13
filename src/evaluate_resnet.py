
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report

from data_pipeline import test_loader, class_names
from resnet_model import create_resnet18


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_resnet18_model.pth"
REPORT_PATH = PROJECT_ROOT / "reports" / "resnet18_classification_report.txt"

DEVICE = torch.device("cpu")


def main():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model = create_resnet18(
        num_classes=len(class_names),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(DEVICE))
            predictions = outputs.argmax(dim=1).cpu()

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.tolist())

    accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )

    print(f"Best training epoch: {checkpoint['best_epoch']}")
    print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nClassification Report:\n")
    print(report)

    REPORT_PATH.write_text(
        f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n"
        + report,
        encoding="utf-8",
    )

    print("Report saved to:", REPORT_PATH)


if __name__ == "__main__":
    main()
