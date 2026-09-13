from pathlib import Path
import json
import csv

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"D:\BotoniiQ\BotoniiQ-ML")

TEST_DIR = PROJECT_DIR / "dataset" / "species" / "test"

MODEL_PATH = PROJECT_DIR / "outputs" / "models" / "species_v2_best.pth"

CLASS_NAMES_PATH = (
    PROJECT_DIR / "outputs" / "models" / "species_v2_class_names.json"
)

REPORT_DIR = PROJECT_DIR / "outputs" / "reports"
PLOT_DIR = PROJECT_DIR / "outputs" / "plots"


REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 4
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print("=" * 70)
print("BOTONIIQ SPECIES V2 EVALUATION")
print("=" * 70)

print(f"Device       : {DEVICE}")
print(f"Test folder  : {TEST_DIR}")
print(f"Model        : {MODEL_PATH}")
print(f"Class mapping: {CLASS_NAMES_PATH}")
print()


# ============================================================
# CHECK FILES
# ============================================================

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test dataset not found:\n{TEST_DIR}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"V2 model not found:\n{MODEL_PATH}"
    )

if not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError(
        f"Class mapping not found:\n{CLASS_NAMES_PATH}"
    )


# ============================================================
# LOAD CLASS MAPPING
# ============================================================

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_mapping = json.load(f)


# Support both:
# {"0": "Neem", "1": "Tulsi"}
#
# and:
# {"classes": ["Neem", "Tulsi"]}

if isinstance(class_mapping, dict) and "classes" in class_mapping:
    class_names = class_mapping["classes"]

elif isinstance(class_mapping, dict):
    class_names = [
        class_mapping[str(i)]
        for i in range(len(class_mapping))
    ]

elif isinstance(class_mapping, list):
    class_names = class_mapping

else:
    raise ValueError(
        "Unsupported class mapping format."
    )


num_classes = len(class_names)

print(f"Number of classes: {num_classes}")

print("\nFirst few classes:")
for i, name in enumerate(class_names[:10]):
    print(f"  {i}: {name}")

print()


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
# LOAD TEST DATASET
# ============================================================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

print(f"Test images: {len(test_dataset)}")


# ============================================================
# VERIFY CLASS ORDER
# ============================================================

dataset_classes = test_dataset.classes

print("\nChecking class mapping...")

if dataset_classes != class_names:
    print("\nWARNING: Dataset class order differs from mapping.")

    print("\nDataset classes:")
    for i, name in enumerate(dataset_classes):
        print(f"  {i}: {name}")

    print("\nMapping classes:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")

    raise ValueError(
        "\nClass mapping mismatch. "
        "Evaluation stopped to prevent incorrect results."
    )

print("Class mapping verified successfully.")


# ============================================================
# DATA LOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda")
)


# ============================================================
# CREATE EFFICIENTNET-B0
# ============================================================

print("\nCreating EfficientNet-B0...")

model = models.efficientnet_b0(
    weights=None
)

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    num_classes
)


# ============================================================
# LOAD V2 MODEL
# ============================================================

print("Loading V2 model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("Checkpoint format: full checkpoint")

    if "val_accuracy" in checkpoint:
        print(
            f"Saved validation accuracy: "
            f"{checkpoint['val_accuracy'] * 100:.2f}%"
        )

elif isinstance(checkpoint, dict):

    model.load_state_dict(checkpoint)

    print("Checkpoint format: state_dict")

else:

    raise ValueError(
        "Unsupported model checkpoint format."
    )


model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# EVALUATION
# ============================================================

print("\nRunning evaluation...")
print("-" * 70)

all_targets = []
all_predictions = []

correct = 0
total = 0

top3_correct = 0


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

        # Top-1
        _, predictions = torch.max(
            probabilities,
            dim=1
        )

        # Top-3
        top3 = torch.topk(
            probabilities,
            k=min(3, num_classes),
            dim=1
        ).indices

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        top3_correct += (
            top3 == labels.unsqueeze(1)
        ).any(dim=1).sum().item()

        all_targets.extend(
            labels.cpu().numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )


# ============================================================
# METRICS
# ============================================================

top1_accuracy = correct / total
top3_accuracy = top3_correct / total


print("\n" + "=" * 70)
print("SPECIES V2 RESULTS")
print("=" * 70)

print(f"Test Images : {total}")
print(f"Correct     : {correct}")
print(f"Incorrect   : {total - correct}")
print()
print(f"Top-1 Accuracy : {top1_accuracy * 100:.2f}%")
print(f"Top-3 Accuracy : {top3_accuracy * 100:.2f}%")
print()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_targets,
    all_predictions,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_path = (
    REPORT_DIR /
    "species_v2_classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("BOTONIIQ SPECIES V2 EVALUATION\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Test Images: {total}\n")
    f.write(
        f"Top-1 Accuracy: "
        f"{top1_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Top-3 Accuracy: "
        f"{top3_accuracy * 100:.2f}%\n\n"
    )

    f.write("=" * 70 + "\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("=" * 70 + "\n\n")

    f.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_targets,
    all_predictions
)


cm_path = (
    REPORT_DIR /
    "species_v2_confusion_matrix.csv"
)

with open(
    cm_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow(
        ["Actual \\ Predicted"] + class_names
    )

    for i, row in enumerate(cm):

        writer.writerow(
            [class_names[i]] + row.tolist()
        )


# ============================================================
# SAVE SUMMARY JSON
# ============================================================

summary = {
    "model": "EfficientNet-B0 Species V2",
    "model_path": str(MODEL_PATH),
    "test_images": total,
    "correct": correct,
    "incorrect": total - correct,
    "top1_accuracy": top1_accuracy,
    "top3_accuracy": top3_accuracy,
    "num_classes": num_classes,
    "device": str(DEVICE)
}


summary_path = (
    REPORT_DIR /
    "species_v2_evaluation_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# FINAL
# ============================================================

print("=" * 70)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nSaved reports:")

print(
    f"1. {report_path}"
)

print(
    f"2. {cm_path}"
)

print(
    f"3. {summary_path}"
)

print("\nDone.")