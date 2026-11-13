
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = PROJECT_ROOT / "reports" / "cnn_confusion_matrix.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "cnn_confusion_matrix.png"

# Load the confusion matrix saved during evaluation
matrix_df = pd.read_csv(CSV_PATH, index_col=0)

# Draw the heatmap
fig, ax = plt.subplots(figsize=(16, 14))

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix_df.values,
    display_labels=matrix_df.index.tolist(),
)

display.plot(
    ax=ax,
    xticks_rotation=90,
    cmap="Blues",
    colorbar=False,
    values_format="d",
)

ax.set_title("Plant Disease Classification - Confusion Matrix")
ax.set_xlabel("Predicted class")
ax.set_ylabel("Actual class")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
plt.close()

print("Confusion matrix saved to:")
print(OUTPUT_PATH)
