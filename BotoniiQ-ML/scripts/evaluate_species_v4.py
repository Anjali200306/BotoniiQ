"""
BOTONIIQ - SPECIES V4 EVALUATION
=================================

Evaluates V4 on the ORIGINAL untouched 71-class test set.

Outputs:
    Top-1 Accuracy
    Top-3 Accuracy
    Classification Report
    Confusion Matrix
"""

from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

TEST_DIR = PROJECT_DIR / "dataset" / "species" / "test"

MODEL_PATH = (
    PROJECT_DIR
    / "outputs"
    / "models"
    / "species_v4_best.pth"
)

MAPPING_PATH = (
    PROJECT_DIR
    / "outputs"
    / "models"
    / "species_v4_class_names.json"
)

REPORT_PATH = (
    PROJECT_DIR
    / "outputs"
    / "reports"
    / "species_v4_classification_report.txt"
)

CONFUSION_PATH = (
    PROJECT_DIR
    / "outputs"
    / "reports"
    / "species_v4_confusion_matrix.csv"
)

SUMMARY_PATH = (
    PROJECT_DIR
    / "outputs"
    / "reports"
    / "species_v4_evaluation_summary.json"
)


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 4


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("BOTONIIQ SPECIES V4 - ORIGINAL TEST EVALUATION")
print("=" * 70)

print()
print("Device :", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU    :",
        torch.cuda.get_device_name(0)
    )

print()


# ============================================================
# CHECK FILES
# ============================================================

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test directory not found:\n{TEST_DIR}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"V4 model not found:\n{MODEL_PATH}"
    )

if not MAPPING_PATH.exists():
    raise FileNotFoundError(
        f"V4 class mapping not found:\n{MAPPING_PATH}"
    )


# ============================================================
# TEST TRANSFORM
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD DATASET
# ============================================================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

dataset_classes = test_dataset.classes

print("=" * 70)
print("TEST DATASET")
print("=" * 70)

print()
print("Test images  :", len(test_dataset))
print("Test classes :", len(dataset_classes))
print()


# ============================================================
# LOAD CLASS MAPPING
# ============================================================

with open(
    MAPPING_PATH,
    "r",
    encoding="utf-8"
) as f:

    mapping_data = json.load(f)


if isinstance(mapping_data, dict):

    if "classes" in mapping_data:

        class_names = mapping_data["classes"]

    else:

        class_names = [
            mapping_data[str(i)]
            for i in range(len(mapping_data))
        ]

else:

    class_names = mapping_data


# ============================================================
# VERIFY CLASS ORDER
# ============================================================

if dataset_classes != class_names:

    print("ERROR: CLASS ORDER MISMATCH")
    print()

    print("Dataset classes:")
    print(dataset_classes)

    print()

    print("Mapping classes:")
    print(class_names)

    raise RuntimeError(
        "Dataset class order does not match V4 mapping."
    )


num_classes = len(class_names)

print("Class mapping verified.")
print("Number of classes:", num_classes)
print()


# ============================================================
# LOAD MODEL
# ============================================================

model = models.efficientnet_b0(
    weights=None
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)


if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint["state_dict"]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict,
    strict=True
)

model = model.to(DEVICE)

model.eval()

print("V4 model loaded successfully.")
print()


# ============================================================
# EVALUATION
# ============================================================

all_predictions = []
all_labels = []

top1_correct = 0
top3_correct = 0
total = 0


print("=" * 70)
print("RUNNING EVALUATION")
print("=" * 70)

print()


with torch.no_grad():

    for images, labels in test_loader:

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

        # ----------------------------------------------------
        # TOP-1
        # ----------------------------------------------------

        predictions = outputs.argmax(
            dim=1
        )

        top1_correct += (
            predictions == labels
        ).sum().item()


        # ----------------------------------------------------
        # TOP-3
        # ----------------------------------------------------

        top3_predictions = torch.topk(
            probabilities,
            k=3,
            dim=1
        ).indices

        for i in range(labels.size(0)):

            actual = labels[i].item()

            if actual in top3_predictions[i].tolist():

                top3_correct += 1


        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_labels.extend(
            labels.cpu().tolist()
        )

        total += labels.size(0)


# ============================================================
# METRICS
# ============================================================

top1_accuracy = (
    top1_correct / total
)

top3_accuracy = (
    top3_correct / total
)


print("=" * 70)
print("RESULTS")
print("=" * 70)

print()
print("Test Images     :", total)
print("Top-1 Correct   :", top1_correct)
print("Top-1 Incorrect :", total - top1_correct)

print()

print(
    "Top-1 Accuracy  :",
    f"{top1_accuracy * 100:.2f}%"
)

print(
    "Top-3 Accuracy  :",
    f"{top3_accuracy * 100:.2f}%"
)

print()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(num_classes)),
    target_names=class_names,
    digits=4,
    zero_division=0
)

print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print()
print(report)


# ============================================================
# SAVE REPORT
# ============================================================

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "BOTONIIQ SPECIES V4 - ORIGINAL TEST EVALUATION\n"
    )

    f.write("=" * 70)
    f.write("\n\n")

    f.write(
        f"Test Images     : {total}\n"
    )

    f.write(
        f"Top-1 Correct   : {top1_correct}\n"
    )

    f.write(
        f"Top-1 Incorrect : {total - top1_correct}\n"
    )

    f.write(
        f"Top-1 Accuracy  : {top1_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Top-3 Accuracy  : {top3_accuracy * 100:.2f}%\n"
    )

    f.write("\n")
    f.write("=" * 70)
    f.write("\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("=" * 70)
    f.write("\n\n")

    f.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(num_classes))
)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

cm_df.index.name = "actual"

cm_df.to_csv(
    CONFUSION_PATH,
    encoding="utf-8"
)


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {
    "model": "EfficientNet-B0",
    "version": "V4",
    "test_images": total,
    "top1_correct": top1_correct,
    "top1_incorrect": total - top1_correct,
    "top1_accuracy": top1_accuracy,
    "top3_accuracy": top3_accuracy,
    "num_classes": num_classes,
    "model_path": str(MODEL_PATH),
    "test_path": str(TEST_DIR)
}

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# FINAL
# ============================================================

print("=" * 70)
print("REPORTS SAVED")
print("=" * 70)

print()
print("Classification report:")
print(REPORT_PATH)

print()

print("Confusion matrix:")
print(CONFUSION_PATH)

print()

print("Summary:")
print(SUMMARY_PATH)

print()

print("=" * 70)