from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

TEST_DIR = PROJECT_DIR / "dataset" / "species" / "test"

MODEL_PATH = (
    PROJECT_DIR
    / "outputs"
    / "models"
    / "species_v3_best.pth"
)

CLASS_MAPPING_PATH = (
    PROJECT_DIR
    / "outputs"
    / "models"
    / "species_v3_class_names.json"
)

REPORT_PATH = (
    PROJECT_DIR
    / "outputs"
    / "reports"
    / "species_v3_classification_report.txt"
)

CM_PATH = (
    PROJECT_DIR
    / "outputs"
    / "reports"
    / "species_v3_confusion_matrix.csv"
)

SUMMARY_PATH = (
    PROJECT_DIR
    / "outputs"
    / "reports"
    / "species_v3_evaluation_summary.json"
)


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 70)
print("BOTONIIQ SPECIES V3 - ORIGINAL TEST EVALUATION")
print("=" * 70)

print(f"Device : {DEVICE}")
print(f"Test   : {TEST_DIR}")
print(f"Model  : {MODEL_PATH}")
print()


# ============================================================
# LOAD CLASS MAPPING
# ============================================================

with open(
    CLASS_MAPPING_PATH,
    "r",
    encoding="utf-8"
) as f:
    mapping_data = json.load(f)


if isinstance(mapping_data, dict) and "classes" in mapping_data:

    class_names = mapping_data["classes"]

elif isinstance(mapping_data, dict):

    class_names = [
        mapping_data[str(i)]
        for i in range(len(mapping_data))
    ]

elif isinstance(mapping_data, list):

    class_names = mapping_data

else:
    raise ValueError("Unsupported class mapping format.")


class_names = list(class_names)

print(f"Number of classes: {len(class_names)}")
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
# DATASET
# ============================================================

dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)


print(f"Test images: {len(dataset)}")
print()


# ============================================================
# VERIFY CLASS ORDER
# ============================================================

dataset_classes = dataset.classes

if dataset_classes != class_names:

    print("WARNING: Dataset class order differs from mapping.")

    print()
    print("Dataset classes:")
    for i, name in enumerate(dataset_classes):
        print(i, name)

    print()
    print("Mapping classes:")
    for i, name in enumerate(class_names):
        print(i, name)

    raise ValueError(
        "Class mapping does not match ImageFolder class order."
    )

print("Class mapping verified.")
print()


# ============================================================
# DATALOADER
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# MODEL
# ============================================================

model = models.efficientnet_b0(
    weights=None
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(class_names)
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
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


# ============================================================
# EVALUATION
# ============================================================

all_predictions = []
all_targets = []

top1_correct = 0
top3_correct = 0
total = 0

print("=" * 70)
print("RUNNING EVALUATION")
print("=" * 70)

with torch.no_grad():

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = probabilities.argmax(dim=1)

        top1_correct += (
            predictions == labels
        ).sum().item()

        top3 = torch.topk(
            probabilities,
            k=3,
            dim=1
        ).indices

        for i in range(labels.size(0)):

            if labels[i].item() in top3[i].tolist():

                top3_correct += 1

        total += labels.size(0)

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_targets.extend(
            labels.cpu().tolist()
        )


# ============================================================
# METRICS
# ============================================================

top1_accuracy = top1_correct / total
top3_accuracy = top3_correct / total

report = classification_report(
    all_targets,
    all_predictions,
    target_names=class_names,
    digits=4,
    zero_division=0
)

cm = confusion_matrix(
    all_targets,
    all_predictions
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("V3 ORIGINAL TEST RESULTS")
print("=" * 70)

print(f"Test images     : {total}")
print(f"Top-1 correct   : {top1_correct}")
print(f"Top-1 incorrect : {total - top1_correct}")
print()
print(f"Top-1 Accuracy  : {top1_accuracy * 100:.2f}%")
print(f"Top-3 Accuracy  : {top3_accuracy * 100:.2f}%")

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

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
        "BOTONIIQ SPECIES V3 - ORIGINAL TEST EVALUATION\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Test images     : {total}\n"
        f"Top-1 correct   : {top1_correct}\n"
        f"Top-1 incorrect : {total - top1_correct}\n"
        f"Top-1 Accuracy  : {top1_accuracy * 100:.2f}%\n"
        f"Top-3 Accuracy  : {top3_accuracy * 100:.2f}%\n\n"
    )

    f.write(report)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

with open(
    CM_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "," + ",".join(class_names) + "\n"
    )

    for name, row in zip(class_names, cm):

        f.write(
            name
            + ","
            + ",".join(map(str, row))
            + "\n"
        )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "model": "species_v3_best.pth",
    "test_images": total,
    "top1_correct": top1_correct,
    "top1_incorrect": total - top1_correct,
    "top1_accuracy": top1_accuracy,
    "top3_accuracy": top3_accuracy,
    "num_classes": len(class_names),
    "device": str(DEVICE)
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


print()
print("=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)

print(f"Report : {REPORT_PATH}")
print(f"Matrix : {CM_PATH}")
print(f"Summary: {SUMMARY_PATH}")

print("=" * 70)