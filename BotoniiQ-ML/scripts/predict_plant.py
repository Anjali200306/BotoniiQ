from pathlib import Path
import json
import sys

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

# ============================================================
# MODEL PATHS
# ============================================================

SPECIES_MODEL_PATH = (
    PROJECT_DIR
    / "outputs"
    / "models"
    / "species_v4_best.pth"
)

SPECIES_MAPPING_PATH = (
    PROJECT_DIR
    / "outputs"
    / "models"
    / "species_v4_class_names.json"
)

DISEASE_MODEL_PATH = (
    PROJECT_DIR
    / "outputs"
    / "models"
    / "disease_efficientnet_b0_best.pth"
)

DISEASE_MAPPING_PATH = (
    PROJECT_DIR
    / "outputs"
    / "models"
    / "disease_class_names.json"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD CLASS MAPPING
# ============================================================

def load_class_mapping(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Class mapping not found:\n{path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Format:
    # {"classes": [...]}
    if isinstance(data, dict) and "classes" in data:
        return data["classes"]

    # Format:
    # {"0": "Aloevera", "1": "Amla", ...}
    if isinstance(data, dict):

        keys = sorted(
            data.keys(),
            key=lambda x: int(x)
        )

        return [data[k] for k in keys]

    # Format:
    # [...]
    if isinstance(data, list):
        return data

    raise ValueError(
        f"Unsupported class mapping format:\n{path}"
    )


# ============================================================
# LOAD EFFICIENTNET MODEL
# ============================================================

def load_efficientnet(model_path, num_classes):

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    model = models.efficientnet_b0(
        weights=None
    )

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )

    checkpoint = torch.load(
        model_path,
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

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# PREDICT
# ============================================================

def predict(model, image_path, class_names, top_k=3):

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        k = min(
            top_k,
            len(class_names)
        )

        values, indices = torch.topk(
            probabilities,
            k=k,
            dim=1
        )

    results = []

    for probability, index in zip(
        values[0],
        indices[0]
    ):

        results.append({
            "class": class_names[
                int(index)
            ],
            "confidence": float(
                probability.item() * 100
            )
        })

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("BOTONIIQ - COMPLETE PLANT PREDICTION")
    print("=" * 70)

    # --------------------------------------------------------
    # CHECK IMAGE ARGUMENT
    # --------------------------------------------------------

    if len(sys.argv) < 2:

        print()
        print("Usage:")
        print(
            'python scripts\\predict_plant.py "path\\to\\image.jpg"'
        )
        print()

        return

    image_path = Path(
        sys.argv[1]
    )

    if not image_path.exists():

        print()
        print(
            f"ERROR: Image not found:\n"
            f"{image_path}"
        )
        print()

        return

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    print()
    print(
        f"Device : {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU    : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # LOAD SPECIES MAPPING
    # --------------------------------------------------------

    print()
    print(
        "Loading species class mapping..."
    )

    species_classes = load_class_mapping(
        SPECIES_MAPPING_PATH
    )

    print(
        f"Species classes : "
        f"{len(species_classes)}"
    )

    # --------------------------------------------------------
    # LOAD DISEASE MAPPING
    # --------------------------------------------------------

    print(
        "Loading disease class mapping..."
    )

    disease_classes = load_class_mapping(
        DISEASE_MAPPING_PATH
    )

    print(
        f"Disease classes : "
        f"{len(disease_classes)}"
    )

    # --------------------------------------------------------
    # LOAD SPECIES MODEL
    # --------------------------------------------------------

    print()
    print(
        "Loading Species V4 model..."
    )

    species_model = load_efficientnet(
        SPECIES_MODEL_PATH,
        len(species_classes)
    )

    print(
        "Species V4 model loaded."
    )

    # --------------------------------------------------------
    # LOAD DISEASE MODEL
    # --------------------------------------------------------

    print(
        "Loading Disease model..."
    )

    disease_model = load_efficientnet(
        DISEASE_MODEL_PATH,
        len(disease_classes)
    )

    print(
        "Disease model loaded."
    )

    # ========================================================
    # SPECIES PREDICTION
    # ========================================================

    print()
    print("=" * 70)
    print("SPECIES IDENTIFICATION")
    print("=" * 70)

    species_results = predict(
        species_model,
        image_path,
        species_classes,
        top_k=3
    )

    print()

    for rank, result in enumerate(
        species_results,
        start=1
    ):

        print(
            f"{rank}. "
            f"{result['class']:<30}"
            f"{result['confidence']:.2f}%"
        )

    # ========================================================
    # DISEASE PREDICTION
    # ========================================================

    print()
    print("=" * 70)
    print("DISEASE IDENTIFICATION")
    print("=" * 70)

    disease_results = predict(
        disease_model,
        image_path,
        disease_classes,
        top_k=3
    )

    print()

    for rank, result in enumerate(
        disease_results,
        start=1
    ):

        print(
            f"{rank}. "
            f"{result['class']:<30}"
            f"{result['confidence']:.2f}%"
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    species = species_results[0]

    disease = disease_results[0]

    print()
    print("=" * 70)
    print("FINAL BOTONIIQ RESULT")
    print("=" * 70)

    print()
    print(
        f"Species    : "
        f"{species['class']}"
    )

    print(
        f"Confidence : "
        f"{species['confidence']:.2f}%"
    )

    print()

    print(
        f"Disease    : "
        f"{disease['class']}"
    )

    print(
        f"Confidence : "
        f"{disease['confidence']:.2f}%"
    )

    print()

    print("=" * 70)
    print("PREDICTION COMPLETED")
    print("=" * 70)

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()