
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATRIX_PATH = PROJECT_ROOT / "reports" / "cnn_confusion_matrix.csv"

matrix = pd.read_csv(MATRIX_PATH, index_col=0)

# Find incorrect predictions, excluding the diagonal
errors = []

for actual_class in matrix.index:
    for predicted_class in matrix.columns:
        if actual_class != predicted_class:
            count = int(matrix.loc[actual_class, predicted_class])

            if count > 0:
                errors.append({
                    "Actual disease": actual_class,
                    "Predicted disease": predicted_class,
                    "Incorrect predictions": count,
                })

errors_df = pd.DataFrame(errors)
errors_df = errors_df.sort_values(
    "Incorrect predictions",
    ascending=False,
)

output_path = PROJECT_ROOT / "reports" / "top_classification_errors.csv"
errors_df.to_csv(output_path, index=False)

print("========== TOP CLASSIFICATION ERRORS ==========")
print(errors_df.head(10).to_string(index=False))
print(f"\nFull error analysis saved to: {output_path}")
