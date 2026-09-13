"""
BOTONIIQ SPECIES V4 - REAL WORLD EVALUATION

Purpose:
    Evaluate the V4 species classification model on the
    completely untouched real-world test dataset.

IMPORTANT:
    Do NOT modify images inside dataset/real_world_test.

Expected structure:

dataset/
└── real_world_test/
    ├── Aloevera/
    │   ├── aloevera_1.jpg
    │   └── ...
    ├── Mint/
    │   ├── mint_1.jpg
    │   └── ...
    ├── Neem/
    │   ├── neem_1.jpg
    │   └── ...
    └── Tulsi/
        ├── tulsi_1.jpg
        └── ...

Model:
    outputs/models/species_v4_best.pth

Class mapping:
    outputs/models/species_v4_class_names.json
"""

from pathlib import Path
import json
import csv

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

REAL_WORLD_TEST_DIR = PROJECT_DIR / "dataset" / "real_world_test"

MODEL_PATH = (
    PROJECT_DIR
    / "outputs"
    / "models"
    / "species_v4_best.pth"
)

CLASS_MAPPING_PATH = (
    PROJECT_DIR
    / "outputs"
    / "models"
    / "species_v4_class_names.json"
)

REPORT_DIR = PROJECT_DIR / "outputs" / "reports"

CSV_PATH = REPORT_DIR / "species_v4_real_world_results.csv"

JSON_PATH = REPORT_DIR / "species_v4_real_world_summary.json"


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 4

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# TRANSFORM
# ============================================================

# IMPORTANT:
# Use the same evaluation-style preprocessing used for V4
# original test evaluation.

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# LOAD CLASS MAPPING
# ============================================================

def load_class_mapping(path):
    """
    Supports both possible mapping formats:

    Format 1:
    {
        "classes": [
            "Aloevera",
            "Amla",
            ...
        ]
    }

    Format 2:
    {
        "0": "Aloevera",
        "1": "Amla",
        ...
    }

    Format 3:
    [
        "Aloevera",
        "Amla",
        ...
    ]
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Class mapping file not found:\n{path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Format 1
    if isinstance(data, dict) and "classes" in data:
        class_names = data["classes"]

    # Format 2
    elif isinstance(data, dict):
        keys = list(data.keys())

        try:
            numeric_keys = sorted(
                keys,
                key=lambda x: int(x)
            )

            class_names = [
                data[key]
                for key in numeric_keys
            ]

        except (ValueError, TypeError):
            raise ValueError(
                "Unsupported class mapping dictionary format."
            )

    # Format 3
    elif isinstance(data, list):
        class_names = data

    else:
        raise ValueError(
            "Unsupported class mapping format."
        )

    if not class_names:
        raise ValueError(
            "Class mapping is empty."
        )

    return class_names


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_path, num_classes):
    """
    Load EfficientNet-B0 V4 model.
    """

    print()
    print("=" * 70)
    print("LOADING V4 MODEL")
    print("=" * 70)

    if not model_path.exists():
        raise FileNotFoundError(
            f"V4 model not found:\n{model_path}"
        )

    # Create EfficientNet-B0 architecture
    model = models.efficientnet_b0(
        weights=None
    )

    # Replace classifier for 71 species
    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )

    # Load checkpoint
    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False
    )

    # Support different checkpoint formats
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # Remove possible "module." prefix
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            new_key = key[len("module."):]
        else:
            new_key = key

        cleaned_state_dict[new_key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    print()
    print("V4 model loaded successfully.")
    print(f"Model path : {model_path}")
    print(f"Classes    : {num_classes}")
    print(f"Device     : {DEVICE}")

    return model


# ============================================================
# COLLECT REAL-WORLD IMAGES
# ============================================================

def collect_real_world_images():
    """
    Collect images from:

        dataset/real_world_test/<actual_class>/

    Returns:
        list of dictionaries containing:
            path
            actual_class
    """

    if not REAL_WORLD_TEST_DIR.exists():
        raise FileNotFoundError(
            f"Real-world test directory not found:\n"
            f"{REAL_WORLD_TEST_DIR}"
        )

    samples = []

    class_directories = sorted(
        [
            p
            for p in REAL_WORLD_TEST_DIR.iterdir()
            if p.is_dir()
        ]
    )

    if not class_directories:
        raise ValueError(
            "No class folders found inside real_world_test."
        )

    for class_dir in class_directories:

        actual_class = class_dir.name

        for image_path in sorted(
            class_dir.iterdir()
        ):

            if (
                image_path.is_file()
                and image_path.suffix.lower()
                in VALID_EXTENSIONS
            ):

                samples.append({
                    "path": image_path,
                    "actual_class": actual_class
                })

    if not samples:
        raise ValueError(
            "No images found inside real_world_test."
        )

    return samples


# ============================================================
# PREDICT ONE IMAGE
# ============================================================

def predict_image(model, image_path, class_names):
    """
    Predict one real-world image.

    Returns:
        predicted_class
        confidence
        top3 list
    """

    try:

        image = Image.open(image_path).convert("RGB")

    except Exception as e:

        raise RuntimeError(
            f"Could not read image:\n"
            f"{image_path}\n"
            f"Error: {e}"
        )

    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        top_k = min(
            3,
            len(class_names)
        )

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=top_k,
            dim=1
        )

    top_probabilities = (
        top_probabilities[0]
        .cpu()
        .numpy()
    )

    top_indices = (
        top_indices[0]
        .cpu()
        .numpy()
    )

    top3 = []

    for probability, index in zip(
        top_probabilities,
        top_indices
    ):

        top3.append({
            "class": class_names[int(index)],
            "confidence": float(probability * 100.0)
        })

    predicted_class = top3[0]["class"]
    confidence = top3[0]["confidence"]

    return (
        predicted_class,
        confidence,
        top3
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("BOTONIIQ SPECIES V4 - REAL WORLD EVALUATION")
    print("=" * 70)

    print()
    print(f"Project : {PROJECT_DIR}")
    print(f"Device  : {DEVICE}")

    if torch.cuda.is_available():

        print(
            f"GPU     : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # LOAD CLASS MAPPING
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CLASS MAPPING")
    print("=" * 70)

    class_names = load_class_mapping(
        CLASS_MAPPING_PATH
    )

    print(
        f"Number of classes : {len(class_names)}"
    )

    # For this project we expect 71 species.
    if len(class_names) != 71:

        raise ValueError(
            f"Expected 71 classes, "
            f"but mapping contains {len(class_names)}."
        )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = load_model(
        MODEL_PATH,
        len(class_names)
    )

    # --------------------------------------------------------
    # COLLECT TEST IMAGES
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("REAL-WORLD TEST DATASET")
    print("=" * 70)

    samples = collect_real_world_images()

    print(
        f"Real-world images found : {len(samples)}"
    )

    # --------------------------------------------------------
    # PRINT ACTUAL CLASS COUNTS
    # --------------------------------------------------------

    actual_class_counts = {}

    for sample in samples:

        actual_class = sample["actual_class"]

        actual_class_counts[
            actual_class
        ] = (
            actual_class_counts.get(
                actual_class,
                0
            ) + 1
        )

    print()
    print("Actual class distribution:")

    for class_name in sorted(
        actual_class_counts.keys()
    ):

        print(
            f"  {class_name:<15} : "
            f"{actual_class_counts[class_name]}"
        )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RUNNING V4 REAL-WORLD EVALUATION")
    print("=" * 70)

    results = []

    top1_correct = 0
    top3_correct = 0

    # Per-class statistics
    per_class = {}

    for sample in samples:

        image_path = sample["path"]
        actual_class = sample["actual_class"]

        (
            predicted_class,
            confidence,
            top3
        ) = predict_image(
            model,
            image_path,
            class_names
        )

        top3_names = [
            item["class"]
            for item in top3
        ]

        is_top1_correct = (
            predicted_class == actual_class
        )

        is_top3_correct = (
            actual_class in top3_names
        )

        if is_top1_correct:
            top1_correct += 1

        if is_top3_correct:
            top3_correct += 1

        # Initialize class statistics
        if actual_class not in per_class:

            per_class[actual_class] = {
                "total": 0,
                "top1_correct": 0,
                "top3_correct": 0
            }

        per_class[actual_class]["total"] += 1

        if is_top1_correct:

            per_class[
                actual_class
            ]["top1_correct"] += 1

        if is_top3_correct:

            per_class[
                actual_class
            ]["top3_correct"] += 1

        # ----------------------------------------------------
        # PRINT INDIVIDUAL RESULT
        # ----------------------------------------------------

        if is_top1_correct:

            status = "[CORRECT]"

        else:

            status = "[WRONG]"

        print()
        print(
            f"{status} "
            f"{actual_class}\\"
            f"{image_path.name}"
        )

        print(
            f"  Prediction : "
            f"{predicted_class}"
        )

        print(
            f"  Confidence : "
            f"{confidence:.2f}%"
        )

        print("  Top-3     :")

        for rank, item in enumerate(
            top3,
            start=1
        ):

            print(
                f"    {rank}. "
                f"{item['class']} "
                f"({item['confidence']:.2f}%)"
            )

        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        results.append({
            "image": image_path.name,
            "image_path": str(
                image_path.relative_to(
                    PROJECT_DIR
                )
            ),
            "actual_class": actual_class,
            "predicted_class": predicted_class,
            "confidence_percent": round(
                confidence,
                4
            ),
            "top1_correct": is_top1_correct,
            "top3_correct": is_top3_correct,
            "top1_prediction": top3[0]["class"],
            "top1_confidence_percent": round(
                top3[0]["confidence"],
                4
            ),
            "top2_prediction": (
                top3[1]["class"]
                if len(top3) > 1
                else ""
            ),
            "top2_confidence_percent": (
                round(
                    top3[1]["confidence"],
                    4
                )
                if len(top3) > 1
                else ""
            ),
            "top3_prediction": (
                top3[2]["class"]
                if len(top3) > 2
                else ""
            ),
            "top3_confidence_percent": (
                round(
                    top3[2]["confidence"],
                    4
                )
                if len(top3) > 2
                else ""
            )
        })

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    total_images = len(samples)

    top1_accuracy = (
        top1_correct / total_images * 100
    )

    top3_accuracy = (
        top3_correct / total_images * 100
    )

    print()
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print()
    print(
        f"Total images     : {total_images}"
    )

    print(
        f"Top-1 correct    : {top1_correct}"
    )

    print(
        f"Top-1 wrong      : "
        f"{total_images - top1_correct}"
    )

    print(
        f"Top-1 Accuracy   : "
        f"{top1_accuracy:.2f}%"
    )

    print(
        f"Top-3 correct    : {top3_correct}"
    )

    print(
        f"Top-3 Accuracy   : "
        f"{top3_accuracy:.2f}%"
    )

    # ========================================================
    # PER-CLASS RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("PER-CLASS RESULTS")
    print("=" * 70)

    per_class_summary = {}

    for class_name in sorted(
        per_class.keys()
    ):

        total = per_class[
            class_name
        ]["total"]

        correct1 = per_class[
            class_name
        ]["top1_correct"]

        correct3 = per_class[
            class_name
        ]["top3_correct"]

        accuracy1 = (
            correct1 / total * 100
        )

        accuracy3 = (
            correct3 / total * 100
        )

        print()
        print(
            f"{class_name}"
        )

        print(
            f"  Images : {total}"
        )

        print(
            f"  Top-1  : "
            f"{correct1}/{total} "
            f"({accuracy1:.2f}%)"
        )

        print(
            f"  Top-3  : "
            f"{correct3}/{total} "
            f"({accuracy3:.2f}%)"
        )

        per_class_summary[class_name] = {
            "total": total,
            "top1_correct": correct1,
            "top3_correct": correct3,
            "top1_accuracy_percent": round(
                accuracy1,
                4
            ),
            "top3_accuracy_percent": round(
                accuracy3,
                4
            )
        }

    # ========================================================
    # CREATE REPORT DIRECTORY
    # ========================================================

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_fields = [
        "image",
        "image_path",
        "actual_class",
        "predicted_class",
        "confidence_percent",
        "top1_correct",
        "top3_correct",
        "top1_prediction",
        "top1_confidence_percent",
        "top2_prediction",
        "top2_confidence_percent",
        "top3_prediction",
        "top3_confidence_percent"
    ]

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=csv_fields
        )

        writer.writeheader()

        writer.writerows(results)

    # ========================================================
    # SAVE JSON SUMMARY
    # ========================================================

    summary = {
        "model": "species_v4_best.pth",
        "model_path": str(
            MODEL_PATH
        ),
        "class_mapping": str(
            CLASS_MAPPING_PATH
        ),
        "dataset": str(
            REAL_WORLD_TEST_DIR
        ),
        "device": str(DEVICE),
        "num_classes": len(class_names),
        "total_images": total_images,
        "top1_correct": top1_correct,
        "top1_incorrect": (
            total_images - top1_correct
        ),
        "top1_accuracy_percent": round(
            top1_accuracy,
            4
        ),
        "top3_correct": top3_correct,
        "top3_accuracy_percent": round(
            top3_accuracy,
            4
        ),
        "actual_class_counts": actual_class_counts,
        "per_class": per_class_summary
    }

    with open(
        JSON_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print("REPORTS SAVED")
    print("=" * 70)

    print()
    print(
        f"CSV results:\n{CSV_PATH}"
    )

    print()
    print(
        f"JSON summary:\n{JSON_PATH}"
    )

    print()
    print("=" * 70)
    print("V4 REAL-WORLD EVALUATION COMPLETED")
    print("=" * 70)

    print()
    print(
        f"FINAL TOP-1 ACCURACY : "
        f"{top1_accuracy:.2f}%"
    )

    print(
        f"FINAL TOP-3 ACCURACY : "
        f"{top3_accuracy:.2f}%"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()