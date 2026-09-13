import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_DIR / "outputs" / "models" / "species_finetuned_best.pth"
CLASS_NAMES_PATH = PROJECT_DIR / "outputs" / "models" / "species_class_names.json"


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224
TOP_K = 3


# ============================================================
# IMAGE TRANSFORMATION
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
# LOAD CLASS NAMES
# ============================================================

def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found:\n{CLASS_NAMES_PATH}"
        )

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):

        # Format:
        # {
        #     "0": "Aloevera",
        #     "1": "Amla",
        #     ...
        # }

        if all(str(i) in data for i in range(len(data))):
            class_names = [
                data[str(i)]
                for i in range(len(data))
            ]

        # Alternative format
        elif "classes" in data:
            class_names = data["classes"]

        else:
            raise ValueError(
                "Unsupported species class mapping format."
            )

    elif isinstance(data, list):
        class_names = data

    else:
        raise ValueError(
            "Unsupported species class mapping format."
        )

    return class_names


# ============================================================
# LOAD SPECIES MODEL
# ============================================================

def load_model(num_classes, device):

    print("Loading EfficientNet-B0...")

    model = models.efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # Remove possible DataParallel prefix
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict)

    model.to(device)
    model.eval()

    print("Model loaded successfully.")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    return model


# ============================================================
# PREDICT SPECIES
# ============================================================

def predict_species(model, image_path, class_names, device):

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=min(TOP_K, len(class_names)),
            dim=1
        )

    predictions = []

    for probability, index in zip(
        top_probabilities[0],
        top_indices[0]
    ):

        class_index = index.item()

        confidence = probability.item() * 100

        predictions.append({
            "class_index": class_index,
            "species": class_names[class_index],
            "confidence": confidence
        })

    return predictions


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 55)
    print("BOTONIIQ SPECIES PREDICTION")
    print("=" * 55)

    # --------------------------------------------------------
    # Check command-line argument
    # --------------------------------------------------------

    if len(sys.argv) < 2:

        print()
        print("Usage:")
        print(
            'python scripts\\predict_species.py "path\\to\\image.jpg"'
        )
        print()

        return

    image_path = Path(sys.argv[1])

    print()
    print(f"Image: {image_path}")
    print()

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Species model not found:\n{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # --------------------------------------------------------
    # Load classes
    # --------------------------------------------------------

    class_names = load_class_names()

    print(f"Classes loaded: {len(class_names)}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(
        len(class_names),
        device
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = predict_species(
        model,
        image_path,
        class_names,
        device
    )

    # --------------------------------------------------------
    # Display Top 3
    # --------------------------------------------------------

    print()
    print("=" * 55)
    print("TOP 3 SPECIES PREDICTIONS")
    print("=" * 55)

    for position, prediction in enumerate(
        predictions,
        start=1
    ):

        print()
        print(
            f"{position}. {prediction['species']}"
        )

        print(
            f"   Confidence: "
            f"{prediction['confidence']:.2f}%"
        )

    # --------------------------------------------------------
    # Best prediction
    # --------------------------------------------------------

    best_prediction = predictions[0]

    print()
    print("=" * 55)

    print(
        f"Prediction: {best_prediction['species']}"
    )

    print(
        f"Confidence: "
        f"{best_prediction['confidence']:.2f}%"
    )

    print("=" * 55)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()