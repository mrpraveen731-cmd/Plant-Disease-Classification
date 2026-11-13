
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from data_pipeline import test_loader, class_names
from cnn_model import PlantDiseaseCNN


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_cnn_model.pth"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")


def main():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model = PlantDiseaseCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.tolist())

    accuracy = accuracy_score(all_labels, all_predictions)

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
    )

    report_path = REPORT_DIR / "cnn_classification_report.txt"
    matrix_path = REPORT_DIR / "cnn_confusion_matrix.csv"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(f"Test accuracy: {accuracy:.4f}\n\n")
        file.write(report)

    import pandas as pd

    pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    ).to_csv(matrix_path)

    print("========== TEST EVALUATION ==========")
    print(f"Test images: {len(test_loader.dataset)}")
    print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nClassification report:")
    print(report)
    print(f"Report saved: {report_path}")
    print(f"Confusion matrix saved: {matrix_path}")


if __name__ == "__main__":
    main()
