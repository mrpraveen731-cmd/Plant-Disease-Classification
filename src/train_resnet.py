
from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam

from data_pipeline import train_loader, val_loader, class_names
from resnet_model import create_resnet18


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")
EPOCHS = 5
LEARNING_RATE = 0.001

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_resnet18_model.pth"
HISTORY_PATH = REPORT_DIR / "resnet18_training_history.csv"


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with (torch.enable_grad() if training else torch.no_grad()):
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
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += batch_size

    return total_loss / total, correct / total


def main():
    model = create_resnet18(
        num_classes=len(class_names),
        pretrained=True,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.fc.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    history = []

    print("Device:", DEVICE)
    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))
    print("Trainable parameters:", sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ))

    for epoch in range(1, EPOCHS + 1):
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

            torch.save({
                "model_state_dict": best_state,
                "class_names": class_names,
                "image_size": 224,
                "architecture": "ResNet18",
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
            }, MODEL_PATH)

            print("  Saved new best ResNet18 model.")

    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

    print("\n========== RESNET18 TRAINING COMPLETE ==========")
    print("Best epoch:", best_epoch)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Best model:", MODEL_PATH)
    print("History:", HISTORY_PATH)


if __name__ == "__main__":
    main()
