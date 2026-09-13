# ============================================================
# BotoniiQ - Disease Model Evaluation
# EfficientNet-B0 - 12 Medicinal Plant Disease Classes
# ============================================================

from pathlib import Path
import json
import csv

import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

import numpy as np


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

TEST_DIR = PROJECT_DIR / "dataset" / "disease" / "test"

MODEL_PATH = (
    PROJECT_DIR
    / "outputs"
    / "models"
    / "disease_efficientnet_b0_best.pth"
)

CLASS_NAMES_PATH = (
    PROJECT_DIR
    / "outputs"
    / "models"
    / "disease_class_names.json"
)

REPORT_DIR = PROJECT_DIR / "outputs" / "reports"
PLOT_DIR = PROJECT_DIR / "outputs" / "plots"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 4
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 3. PRINT HEADER
# ============================================================

print("=" * 70)
print("BOTONIIQ - DISEASE MODEL EVALUATION")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print()


# ============================================================
# 4. CHECK FILES
# ============================================================

print("=" * 70)
print("CHECKING FILES")
print("=" * 70)

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test dataset not found:\n{TEST_DIR}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found:\n{MODEL_PATH}"
    )

print(f"Test dataset found : {TEST_DIR}")
print(f"Model found        : {MODEL_PATH}")

if CLASS_NAMES_PATH.exists():
    print(f"Class mapping found : {CLASS_NAMES_PATH}")
else:
    print("Class mapping file not found.")
    print("Class names will be taken from ImageFolder.")

print()


# ============================================================
# 5. TEST TRANSFORM
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# 6. LOAD TEST DATASET
# ============================================================

print("=" * 70)
print("LOADING TEST DATASET")
print("=" * 70)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

class_names = test_dataset.classes
num_classes = len(class_names)

print(f"Test images : {len(test_dataset)}")
print(f"Classes     : {num_classes}")

print("\nClasses:")

for index, name in enumerate(class_names):
    print(f"{index:2d} -> {name}")

print()


# ============================================================
# 7. LOAD EFFICIENTNET-B0
# ============================================================

print("=" * 70)
print("LOADING EFFICIENTNET-B0")
print("=" * 70)

model = models.efficientnet_b0(weights=None)

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    num_classes
)


# ============================================================
# 8. LOAD TRAINED CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

# The training script saved the state dictionary
# under the "model_state_dict" key.

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])

elif "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])

else:
    # In case the checkpoint itself is the state dictionary
    model.load_state_dict(checkpoint)


model = model.to(DEVICE)
model.eval()

print("Trained model loaded successfully.")

if isinstance(checkpoint, dict):

    if "epoch" in checkpoint:
        print(f"Saved epoch          : {checkpoint['epoch']}")

    if "val_accuracy" in checkpoint:
        print(
            f"Saved validation acc : "
            f"{checkpoint['val_accuracy']:.2f}%"
        )

print()


# ============================================================
# 9. EVALUATION
# ============================================================

print("=" * 70)
print("RUNNING TEST EVALUATION")
print("=" * 70)

all_labels = []
all_predictions = []
all_probabilities = []

correct = 0
total = 0

with torch.no_grad():

    for batch_index, (images, labels) in enumerate(test_loader):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )


# ============================================================
# 10. TOP-1 ACCURACY
# ============================================================

top1_accuracy = (
    correct / total
) * 100


# ============================================================
# 11. TOP-3 ACCURACY
# ============================================================

all_probabilities = np.array(
    all_probabilities
)

all_labels_np = np.array(
    all_labels
)

top3_correct = 0

for i in range(len(all_labels_np)):

    top3_indices = np.argsort(
        all_probabilities[i]
    )[-3:]

    if all_labels_np[i] in top3_indices:
        top3_correct += 1


top3_accuracy = (
    top3_correct / total
) * 100


# ============================================================
# 12. PRECISION / RECALL / F1
# ============================================================

precision, recall, f1, support = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        labels=list(range(num_classes)),
        zero_division=0
    )
)

macro_precision = precision.mean()
macro_recall = recall.mean()
macro_f1 = f1.mean()


# ============================================================
# 13. PRINT MAIN RESULTS
# ============================================================

print()
print("=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(f"Test Images       : {total}")
print(f"Correct           : {correct}")
print(f"Incorrect         : {total - correct}")

print(
    f"Top-1 Accuracy    : "
    f"{top1_accuracy:.2f}%"
)

print(
    f"Top-3 Accuracy    : "
    f"{top3_accuracy:.2f}%"
)

print(
    f"Macro Precision   : "
    f"{macro_precision:.4f}"
)

print(
    f"Macro Recall      : "
    f"{macro_recall:.4f}"
)

print(
    f"Macro F1-Score    : "
    f"{macro_f1:.4f}"
)

print()


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(num_classes)),
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(report)


# ============================================================
# 15. SAVE CLASSIFICATION REPORT
# ============================================================

classification_report_path = (
    REPORT_DIR
    / "disease_classification_report.txt"
)

with open(
    classification_report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "BOTONIIQ - DISEASE MODEL EVALUATION\n"
    )

    file.write("=" * 70 + "\n\n")

    file.write(
        f"Test Images: {total}\n"
    )

    file.write(
        f"Top-1 Accuracy: {top1_accuracy:.4f}%\n"
    )

    file.write(
        f"Top-3 Accuracy: {top3_accuracy:.4f}%\n"
    )

    file.write(
        f"Macro Precision: {macro_precision:.4f}\n"
    )

    file.write(
        f"Macro Recall: {macro_recall:.4f}\n"
    )

    file.write(
        f"Macro F1-Score: {macro_f1:.4f}\n\n"
    )

    file.write(
        "Classification Report\n"
    )

    file.write("=" * 70 + "\n")

    file.write(report)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(num_classes))
)


# ============================================================
# 17. SAVE CONFUSION MATRIX CSV
# ============================================================

confusion_csv_path = (
    REPORT_DIR
    / "disease_confusion_matrix.csv"
)

with open(
    confusion_csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow(
        ["Actual / Predicted"] + class_names
    )

    for i, row in enumerate(cm):

        writer.writerow(
            [class_names[i]] + row.tolist()
        )


# ============================================================
# 18. SAVE PER-CLASS METRICS CSV
# ============================================================

metrics_csv_path = (
    REPORT_DIR
    / "disease_per_class_metrics.csv"
)

with open(
    metrics_csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Class",
        "Precision",
        "Recall",
        "F1-Score",
        "Support"
    ])

    for i in range(num_classes):

        writer.writerow([
            class_names[i],
            f"{precision[i]:.4f}",
            f"{recall[i]:.4f}",
            f"{f1[i]:.4f}",
            int(support[i])
        ])


# ============================================================
# 19. SAVE SUMMARY JSON
# ============================================================

summary = {
    "model": "EfficientNet-B0",
    "num_classes": num_classes,
    "test_images": int(total),
    "correct_predictions": int(correct),
    "incorrect_predictions": int(total - correct),
    "top1_accuracy_percent": round(top1_accuracy, 4),
    "top3_accuracy_percent": round(top3_accuracy, 4),
    "macro_precision": round(float(macro_precision), 4),
    "macro_recall": round(float(macro_recall), 4),
    "macro_f1": round(float(macro_f1), 4),
    "device": str(DEVICE),
    "classes": class_names
}

summary_path = (
    REPORT_DIR
    / "disease_evaluation_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


# ============================================================
# 20. CONFUSION MATRIX IMAGE
# ============================================================

try:

    import matplotlib.pyplot as plt

    plt.figure(
        figsize=(14, 12)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        "BotoniiQ Disease Classification - Confusion Matrix"
    )

    plt.colorbar()

    tick_marks = np.arange(
        num_classes
    )

    plt.xticks(
        tick_marks,
        class_names,
        rotation=90
    )

    plt.yticks(
        tick_marks,
        class_names
    )

    threshold = cm.max() / 2.0

    for i in range(num_classes):

        for j in range(num_classes):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black"
            )

    plt.ylabel("Actual Class")
    plt.xlabel("Predicted Class")

    plt.tight_layout()

    confusion_plot_path = (
        PLOT_DIR
        / "disease_confusion_matrix.png"
    )

    plt.savefig(
        confusion_plot_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Confusion matrix plot saved to:\n"
        f"{confusion_plot_path}"
    )

except Exception as e:

    print(
        f"Could not create confusion matrix plot: {e}"
    )


# ============================================================
# 21. FINAL OUTPUT PATHS
# ============================================================

print()
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    f"Classification report:\n"
    f"{classification_report_path}"
)

print(
    f"\nConfusion matrix CSV:\n"
    f"{confusion_csv_path}"
)

print(
    f"\nPer-class metrics:\n"
    f"{metrics_csv_path}"
)

print(
    f"\nEvaluation summary:\n"
    f"{summary_path}"
)

print()
print("=" * 70)