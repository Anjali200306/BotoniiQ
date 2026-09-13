from pathlib import Path
import csv

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(
    r"D:\BotoniiQ\BotoniiQ-ML"
)

DATASET_DIR = (
    PROJECT_DIR
    / "dataset"
    / "species"
)

MODEL_PATH = (
    PROJECT_DIR
    / "outputs"
    / "models"
    / "species_finetuned_best.pth"
)

REPORT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "reports"
)

PLOT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "plots"
)


REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("BOTONIIQ - DETAILED SPECIES MODEL EVALUATION")
print("=" * 70)

print()

print("Device:", DEVICE)

print()


# ============================================================
# 3. TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        256
    ),

    transforms.CenterCrop(
        224
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# 4. LOAD TEST DATASET
# ============================================================

test_dataset = datasets.ImageFolder(

    DATASET_DIR / "test",

    transform=transform
)


test_loader = DataLoader(

    test_dataset,

    batch_size=16,

    shuffle=False,

    num_workers=0
)


class_names = test_dataset.classes

num_classes = len(class_names)


print("Test images:", len(test_dataset))

print("Number of classes:", num_classes)

print()


# ============================================================
# 5. LOAD MODEL
# ============================================================

print("Loading trained model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


model = efficientnet_b0(
    weights=None
)


input_features = (
    model.classifier[1].in_features
)


model.classifier[1] = nn.Linear(
    input_features,
    num_classes
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model = model.to(DEVICE)

model.eval()


print("Model loaded successfully.")

print()


# ============================================================
# 6. PREDICTIONS
# ============================================================

all_labels = []

all_predictions = []

top3_correct = 0

total = 0


print("=" * 70)
print("GENERATING PREDICTIONS")
print("=" * 70)

print()


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        outputs = model(
            images
        )


        # ----------------------------------------------------
        # Top-1
        # ----------------------------------------------------

        _, predictions = torch.max(
            outputs,
            1
        )


        # ----------------------------------------------------
        # Top-3
        # ----------------------------------------------------

        _, top3_predictions = torch.topk(
            outputs,
            k=3,
            dim=1
        )


        top3_correct += (
            top3_predictions
            == labels.unsqueeze(1)
        ).any(
            dim=1
        ).sum().item()


        total += labels.size(0)


        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# ============================================================
# 7. ACCURACY
# ============================================================

top1_accuracy = accuracy_score(
    all_labels,
    all_predictions
)

top3_accuracy = (
    top3_correct / total
)


print("Top-1 Accuracy:")

print(
    f"{top1_accuracy * 100:.2f}%"
)

print()

print("Top-3 Accuracy:")

print(
    f"{top3_accuracy * 100:.2f}%"
)

print()


# ============================================================
# 8. CLASSIFICATION REPORT
# ============================================================

print("=" * 70)
print("PER-CLASS METRICS")
print("=" * 70)

print()


report = classification_report(

    all_labels,

    all_predictions,

    target_names=class_names,

    digits=4,

    zero_division=0
)


print(report)


# ============================================================
# 9. SAVE CLASSIFICATION REPORT
# ============================================================

report_path = (
    REPORT_DIR
    / "species_finetuned_classification_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "BOTONIIQ SPECIES CLASSIFICATION REPORT\n"
    )

    file.write(
        "=" * 70
        + "\n\n"
    )

    file.write(
        f"Test Images: {total}\n"
    )

    file.write(
        f"Number of Classes: {num_classes}\n"
    )

    file.write(
        f"Top-1 Accuracy: "
        f"{top1_accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Top-3 Accuracy: "
        f"{top3_accuracy * 100:.2f}%\n\n"
    )

    file.write(
        report
    )


print(
    "Classification report saved:"
)

print(
    report_path
)

print()


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    all_labels,

    all_predictions,

    labels=list(
        range(num_classes)
    )
)


# ============================================================
# 11. SAVE CONFUSION MATRIX CSV
# ============================================================

cm_csv_path = (
    REPORT_DIR
    / "species_finetuned_confusion_matrix.csv"
)


with open(
    cm_csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)


    writer.writerow(
        ["Actual / Predicted"]
        + class_names
    )


    for index, row in enumerate(cm):

        writer.writerow(
            [class_names[index]]
            + row.tolist()
        )


print(
    "Confusion matrix CSV saved:"
)

print(
    cm_csv_path
)

print()


# ============================================================
# 12. PLOT CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(24, 22)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "BotoniiQ Species Confusion Matrix"
)

plt.xlabel(
    "Predicted Species"
)

plt.ylabel(
    "Actual Species"
)

plt.xticks(
    range(num_classes),
    class_names,
    rotation=90,
    fontsize=6
)

plt.yticks(
    range(num_classes),
    class_names,
    fontsize=6
)

plt.colorbar()

plt.tight_layout()


cm_plot_path = (
    PLOT_DIR
    / "species_finetuned_confusion_matrix.png"
)


plt.savefig(
    cm_plot_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print(
    "Confusion matrix plot saved:"
)

print(
    cm_plot_path
)

print()


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("=" * 70)
print("DETAILED EVALUATION COMPLETE")
print("=" * 70)

print()

print(
    f"Top-1 Accuracy : "
    f"{top1_accuracy * 100:.2f}%"
)

print(
    f"Top-3 Accuracy : "
    f"{top3_accuracy * 100:.2f}%"
)

print()

print(
    "Reports:"
)

print(
    report_path
)

print(
    cm_csv_path
)

print()

print(
    "Plot:"
)

print(
    cm_plot_path
)

print()

print("=" * 70)