from pathlib import Path
import json
import csv

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"D:\BotoniiQ\BotoniiQ-ML")

REAL_WORLD_DIR = PROJECT_DIR / "dataset" / "real_world_test"

MODEL_PATH = (
    PROJECT_DIR
    / "outputs"
    / "models"
    / "species_v2_best.pth"
)

CLASS_NAMES_PATH = (
    PROJECT_DIR
    / "outputs"
    / "models"
    / "species_v2_class_names.json"
)

REPORT_DIR = PROJECT_DIR / "outputs" / "reports"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("BOTONIIQ SPECIES V2 - REAL WORLD EVALUATION")
print("=" * 70)

print(f"Device       : {DEVICE}")
print(f"Real dataset : {REAL_WORLD_DIR}")
print(f"Model        : {MODEL_PATH}")
print()


# ============================================================
# CHECK PATHS
# ============================================================

if not REAL_WORLD_DIR.exists():
    raise FileNotFoundError(
        f"Real-world dataset not found:\n{REAL_WORLD_DIR}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError(
        f"Class mapping not found:\n{CLASS_NAMES_PATH}"
    )


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as f:

    class_mapping = json.load(f)


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

print(f"Classes loaded: {num_classes}")


# ============================================================
# CLASS NAME NORMALIZATION
# ============================================================

def normalize_name(name):

    return (
        name
        .lower()
        .replace("_", "")
        .replace(" ", "")
    )


# ============================================================
# REAL-WORLD FOLDER → MODEL CLASS
# ============================================================

folder_to_class = {}

for folder in REAL_WORLD_DIR.iterdir():

    if not folder.is_dir():
        continue

    folder_name = folder.name

    matches = [
        i
        for i, class_name in enumerate(class_names)
        if normalize_name(class_name)
        == normalize_name(folder_name)
    ]

    if not matches:

        raise ValueError(
            f"Could not match real-world folder "
            f"'{folder_name}' to model classes."
        )

    folder_to_class[folder_name] = matches[0]


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
# CREATE MODEL
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
# LOAD MODEL
# ============================================================

print("Loading Species V2 model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(checkpoint)


model = model.to(DEVICE)

model.eval()

print("Model loaded successfully.")


# ============================================================
# COLLECT IMAGES
# ============================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}

samples = []

for folder in sorted(REAL_WORLD_DIR.iterdir()):

    if not folder.is_dir():
        continue

    actual_class_name = folder.name

    actual_index = folder_to_class[
        actual_class_name
    ]

    for image_path in sorted(folder.iterdir()):

        if image_path.suffix.lower() not in image_extensions:
            continue

        samples.append(
            (
                image_path,
                actual_class_name,
                actual_index
            )
        )


print(f"\nReal-world images found: {len(samples)}")


# ============================================================
# EVALUATION VARIABLES
# ============================================================

results = []

overall_correct = 0
overall_top3_correct = 0


# Per-class statistics

class_stats = {}

for folder_name in folder_to_class:

    class_stats[folder_name] = {
        "total": 0,
        "correct": 0,
        "top3_correct": 0
    }


# ============================================================
# RUN PREDICTIONS
# ============================================================

print("\n")
print("=" * 70)
print("IMAGE-BY-IMAGE RESULTS")
print("=" * 70)


for image_path, actual_folder, actual_index in samples:

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

    except Exception as e:

        print(
            f"\nERROR reading {image_path.name}: {e}"
        )

        continue


    input_tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)


    with torch.no_grad():

        output = model(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

        top_values, top_indices = torch.topk(
            probabilities,
            k=3
        )


    predicted_index = top_indices[0].item()

    predicted_name = class_names[
        predicted_index
    ]

    predicted_confidence = (
        top_values[0].item() * 100
    )


    top3_names = [
        class_names[index.item()]
        for index in top_indices
    ]


    correct = (
        predicted_index == actual_index
    )

    top3_correct = (
        actual_index
        in top_indices.tolist()
    )


    if correct:

        overall_correct += 1

    if top3_correct:

        overall_top3_correct += 1


    class_stats[
        actual_folder
    ]["total"] += 1


    if correct:

        class_stats[
            actual_folder
        ]["correct"] += 1


    if top3_correct:

        class_stats[
            actual_folder
        ]["top3_correct"] += 1


    results.append({
        "image": image_path.name,
        "actual": actual_folder,
        "prediction": predicted_name,
        "confidence": round(
            predicted_confidence,
            2
        ),
        "top3": top3_names,
        "correct": correct,
        "top3_correct": top3_correct
    })


    status = "CORRECT" if correct else "WRONG"

    print(
        f"\n[{status}] {actual_folder}\\{image_path.name}"
    )

    print(
        f"  Prediction : {predicted_name}"
    )

    print(
        f"  Confidence : "
        f"{predicted_confidence:.2f}%"
    )

    print(
        f"  Top-3      : "
        f"{', '.join(top3_names)}"
    )


# ============================================================
# OVERALL RESULTS
# ============================================================

total_images = len(results)

if total_images == 0:

    raise RuntimeError(
        "No valid images were evaluated."
    )


top1_accuracy = (
    overall_correct
    / total_images
)

top3_accuracy = (
    overall_top3_correct
    / total_images
)


print("\n")
print("=" * 70)
print("REAL-WORLD SPECIES V2 RESULTS")
print("=" * 70)

print(
    f"Total images : {total_images}"
)

print(
    f"Top-1 correct: {overall_correct}"
)

print(
    f"Top-1 wrong  : "
    f"{total_images - overall_correct}"
)

print()

print(
    f"Top-1 Accuracy : "
    f"{top1_accuracy * 100:.2f}%"
)

print(
    f"Top-3 Accuracy : "
    f"{top3_accuracy * 100:.2f}%"
)


# ============================================================
# PER-CLASS RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("PER-CLASS REAL-WORLD RESULTS")
print("=" * 70)


for class_name, stats in class_stats.items():

    total = stats["total"]

    if total == 0:
        continue

    top1 = (
        stats["correct"]
        / total
        * 100
    )

    top3 = (
        stats["top3_correct"]
        / total
        * 100
    )

    print(
        f"{class_name:<12} "
        f"{stats['correct']}/{total} "
        f"Top-1={top1:6.2f}% "
        f"Top-3={top3:6.2f}%"
    )


# ============================================================
# SAVE CSV
# ============================================================

csv_path = (
    REPORT_DIR
    / "species_v2_real_world_results.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "image",
        "actual",
        "prediction",
        "confidence",
        "top3",
        "correct",
        "top3_correct"
    ])

    for result in results:

        writer.writerow([
            result["image"],
            result["actual"],
            result["prediction"],
            result["confidence"],
            " | ".join(result["top3"]),
            result["correct"],
            result["top3_correct"]
        ])


# ============================================================
# SAVE JSON
# ============================================================

json_path = (
    REPORT_DIR
    / "species_v2_real_world_summary.json"
)

summary = {
    "model": "Species V2 EfficientNet-B0",
    "total_images": total_images,
    "top1_correct": overall_correct,
    "top1_wrong": (
        total_images - overall_correct
    ),
    "top1_accuracy": top1_accuracy,
    "top3_correct": overall_top3_correct,
    "top3_accuracy": top3_accuracy,
    "per_class": class_stats,
    "results": results
}


with open(
    json_path,
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

print("\n")
print("=" * 70)
print("REAL-WORLD EVALUATION COMPLETED")
print("=" * 70)

print(
    f"CSV report : {csv_path}"
)

print(
    f"JSON report: {json_path}"
)

print("\nDone.")