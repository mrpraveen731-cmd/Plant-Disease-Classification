
from pathlib import Path
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from data_pipeline import train_loader, val_loader, class_names
from cnn_model import PlantDiseaseCNN


# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")

EPOCHS = 10
LEARNING_RATE = 0.001
PATIENCE = 3

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "best_cnn_model.pth"


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            if training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size

    return total_loss / total, correct / total


def main():

    model = PlantDiseaseCNN(num_classes=len(class_names)).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    # Resume from the best saved checkpoint, if it exists.
    if BEST_MODEL_PATH.exists():
        checkpoint = torch.load(
            BEST_MODEL_PATH,
            map_location=DEVICE,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        best_state = copy.deepcopy(model.state_dict())
        best_val_loss = checkpoint["best_val_loss"]
        best_epoch = checkpoint["best_epoch"]

        print(f"Resumed from saved epoch: {best_epoch}")
        print(f"Best validation loss so far: {best_val_loss:.4f}")


    print(f"Device: {DEVICE}")
    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")
    print(f"Classes: {len(class_names)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(best_epoch + 1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer
        )

        val_loss, val_acc = run_epoch(
            model, val_loader, criterion
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        })

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val accuracy: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            torch.save({
                "model_state_dict": best_state,
                "class_names": class_names,
                "image_size": 224,
                "architecture": "PlantDiseaseCNN",
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
            }, BEST_MODEL_PATH)

            print("  Saved new best model.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    import pandas as pd

    history_path = REPORT_DIR / "cnn_training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    print("\n========== TRAINING COMPLETE ==========")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model: {BEST_MODEL_PATH}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
