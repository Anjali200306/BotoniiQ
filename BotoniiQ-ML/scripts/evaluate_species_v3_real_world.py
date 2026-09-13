from pathlib import Path
import json
import csv

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

REAL_WORLD_DIR = PROJECT_DIR / "dataset" / "real_world_test"

MODEL_PATH = PROJECT_DIR / "outputs" / "models" / "species_v3_best.pth"
CLASS_NAMES_PATH = PROJECT_DIR / "outputs" / "models" / "species_v3_class_names.json"

REPORT_DIR = PROJECT_DIR / "outputs" / "reports"

CSV_PATH = REPORT_DIR / "species_v3_real_world_results.csv"
JSON_PATH = REPORT_DIR / "species_v3_real_world_summary.json"


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD CLASS MAPPING
# ============================================================

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_data = json.load(f)


if isinstance(class_data, dict):

    if "classes" in class_data:
        class_names = class_data["classes"]

    else:
        class_names = [
            class_data[str(i)]
            for i in range(len(class_data))
        ]

else:
    class_names = class_data

NUM_CLASSES = len(class_names)


# ============================================================
# MODEL
# ============================================================

print("=" * 70)
print("BOTONIIQ SPECIES V3 - REAL WORLD EVALUATION")
print("=" * 70)

print(f"Device       : {DEVICE}")
print(f"Real dataset : {REAL_WORLD_DIR}")
print(f"Model        : {MODEL_PATH}")
print()

print(f"Classes loaded: {NUM_CLASSES}")
print()

print("Creating EfficientNet-B0...")

model = models.efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    NUM_CLASSES
)


print("Loading Species V3 model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
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


model.load_state_dict(state_dict, strict=True)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")
print()


# ============================================================
# COLLECT REAL-WORLD IMAGES
# ============================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}

samples = []

for class_dir in sorted(REAL_WORLD_DIR.iterdir()):

    if not class_dir.is_dir():
        continue

    actual_class = class_dir.name

    for image_path in sorted(class_dir.iterdir()):

        if image_path.suffix.lower() in image_extensions:

            samples.append(
                (
                    actual_class,
                    image_path
                )
            )


print(f"Real-world images found: {len(samples)}")
print()


# ============================================================
# PREDICTION
# ============================================================

results = []

top1_correct = 0
top3_correct = 0


print("=" * 70)
print("IMAGE-BY-IMAGE RESULTS")
print("=" * 70)


with torch.no_grad():

    for actual_class, image_path in samples:

        image = Image.open(image_path).convert("RGB")

        tensor = transform(image).unsqueeze(0).to(DEVICE)

        outputs = model(tensor)

        probabilities = torch.softmax(outputs, dim=1)

        top_probs, top_indices = torch.topk(
            probabilities,
            k=3,
            dim=1
        )

        top_probs = top_probs[0].cpu().tolist()
        top_indices = top_indices[0].cpu().tolist()

        top3_names = [
            class_names[index]
            for index in top_indices
        ]

        prediction = top3_names[0]
        confidence = top_probs[0] * 100

        is_top1_correct = prediction == actual_class

        is_top3_correct = actual_class in top3_names

        if is_top1_correct:
            top1_correct += 1

        if is_top3_correct:
            top3_correct += 1

        status = "CORRECT" if is_top1_correct else "WRONG"

        print()
        print(f"[{status}] {actual_class}\\{image_path.name}")
        print(f"  Prediction : {prediction}")
        print(f"  Confidence : {confidence:.2f}%")
        print(f"  Top-3      : {', '.join(top3_names)}")

        results.append({
            "actual": actual_class,
            "image": str(image_path.relative_to(REAL_WORLD_DIR)),
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "top1_correct": is_top1_correct,
            "top3_correct": is_top3_correct,
            "top1": top3_names[0],
            "top2": top3_names[1],
            "top3": top3_names[2]
        })


# ============================================================
# OVERALL RESULTS
# ============================================================

total = len(results)

top1_accuracy = (
    top1_correct / total * 100
    if total > 0
    else 0
)

top3_accuracy = (
    top3_correct / total * 100
    if total > 0
    else 0
)


print()
print()
print("=" * 70)
print("REAL-WORLD SPECIES V3 RESULTS")
print("=" * 70)

print(f"Total images : {total}")
print(f"Top-1 correct: {top1_correct}")
print(f"Top-1 wrong  : {total - top1_correct}")

print()

print(f"Top-1 Accuracy : {top1_accuracy:.2f}%")
print(f"Top-3 Accuracy : {top3_accuracy:.2f}%")


# ============================================================
# PER-CLASS RESULTS
# ============================================================

print()
print()
print("=" * 70)
print("PER-CLASS REAL-WORLD RESULTS")
print("=" * 70)


class_results = {}

for actual_class in sorted(
    set(result["actual"] for result in results)
):

    class_items = [
        result
        for result in results
        if result["actual"] == actual_class
    ]

    class_total = len(class_items)

    class_top1 = sum(
        item["top1_correct"]
        for item in class_items
    )

    class_top3 = sum(
        item["top3_correct"]
        for item in class_items
    )

    class_top1_accuracy = (
        class_top1 / class_total * 100
        if class_total
        else 0
    )

    class_top3_accuracy = (
        class_top3 / class_total * 100
        if class_total
        else 0
    )

    print(
        f"{actual_class:<12}"
        f"{class_top1}/{class_total} "
        f"Top-1={class_top1_accuracy:6.2f}% "
        f"Top-3={class_top3_accuracy:6.2f}%"
    )

    class_results[actual_class] = {
        "total": class_total,
        "top1_correct": class_top1,
        "top3_correct": class_top3,
        "top1_accuracy": round(class_top1_accuracy, 4),
        "top3_accuracy": round(class_top3_accuracy, 4)
    }


# ============================================================
# SAVE CSV
# ============================================================

REPORT_DIR.mkdir(parents=True, exist_ok=True)

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "actual",
            "image",
            "prediction",
            "confidence",
            "top1_correct",
            "top3_correct",
            "top1",
            "top2",
            "top3"
        ]
    )

    writer.writeheader()

    writer.writerows(results)


# ============================================================
# SAVE JSON
# ============================================================

summary = {
    "model": "species_v3_best.pth",
    "dataset": "real_world_test",
    "device": str(DEVICE),
    "total_images": total,
    "top1_correct": top1_correct,
    "top1_wrong": total - top1_correct,
    "top1_accuracy": round(top1_accuracy, 4),
    "top3_correct": top3_correct,
    "top3_accuracy": round(top3_accuracy, 4),
    "per_class": class_results,
    "images": results
}


with open(
    JSON_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print()
print("=" * 70)
print("REAL-WORLD V3 EVALUATION COMPLETED")
print("=" * 70)

print(f"CSV report : {CSV_PATH}")
print(f"JSON report: {JSON_PATH}")

print()
print("Done.")