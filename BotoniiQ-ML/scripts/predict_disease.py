# ============================================================
# BotoniiQ - Disease Inference
# EfficientNet-B0 - 12 Medicinal Plant Disease Classes
# ============================================================

from pathlib import Path
import sys
import json

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

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


# ============================================================
# 2. SETTINGS
# ============================================================

IMAGE_SIZE = 224
TOP_K = 3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 3. IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 4. LOAD CLASS NAMES
# ============================================================

def load_class_names():
    """
    Load disease class names from disease_class_names.json.

    Expected format:
    {
        "classes": [
            "Kalanchoe_Healthy",
            ...
        ],
        "class_to_idx": {
            ...
        }
    }
    """

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found:\n{CLASS_NAMES_PATH}"
        )

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Your actual training output format
    if isinstance(data, dict) and "classes" in data:
        class_names = data["classes"]

    # Support a simple list format
    elif isinstance(data, list):
        class_names = data

    # Support {"0": "class_name", "1": "class_name", ...}
    elif isinstance(data, dict):
        try:
            class_names = [
                data[str(i)]
                for i in range(len(data))
            ]
        except (KeyError, TypeError):
            raise ValueError(
                "Unsupported class mapping format."
            )

    else:
        raise ValueError(
            "Unsupported class mapping format."
        )

    if not class_names:
        raise ValueError("No class names found.")

    return class_names

# ============================================================
# 5. LOAD MODEL
# ============================================================

def load_model(class_names):

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model file not found:\n"
            f"{MODEL_PATH}"
        )

    print("Loading EfficientNet-B0...")

    model = models.efficientnet_b0(
        weights=None
    )

    in_features = (
        model.classifier[1].in_features
    )

    model.classifier[1] = nn.Linear(
        in_features,
        len(class_names)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model = model.to(DEVICE)

    model.eval()

    print("Model loaded successfully.")
    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    return model


# ============================================================
# 6. PREDICT IMAGE
# ============================================================

def predict_image(
    model,
    image_path,
    class_names
):

    image_path = Path(image_path)

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n"
            f"{image_path}"
        )

    # Open image

    image = Image.open(
        image_path
    ).convert("RGB")

    # Transform

    image_tensor = transform(
        image
    )

    # Add batch dimension

    image_tensor = image_tensor.unsqueeze(
        0
    )

    image_tensor = image_tensor.to(
        DEVICE
    )

    # Model prediction

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        top_probabilities, top_indices = (
            torch.topk(
                probabilities,
                k=min(
                    TOP_K,
                    len(class_names)
                ),
                dim=1
            )
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

    return [
        (
            class_names[index],
            float(probability)
        )
        for index, probability
        in zip(
            top_indices,
            top_probabilities
        )
    ]


# ============================================================
# 7. MAIN
# ============================================================

def main():

    print("=" * 55)
    print("BOTONIIQ DISEASE PREDICTION")
    print("=" * 55)

    # --------------------------------------------------------
    # Check image argument
    # --------------------------------------------------------

    if len(sys.argv) < 2:

        print()
        print("Usage:")
        print(
            "python scripts\\predict_disease.py "
            "\"path\\to\\image.jpg\""
        )

        print()
        print("Example:")
        print(
            "python scripts\\predict_disease.py "
            "\"test_images\\neem_leaf.jpg\""
        )

        sys.exit(1)

    image_path = Path(
        sys.argv[1]
    )

    print()
    print(f"Image: {image_path}")
    print()

    # --------------------------------------------------------
    # Load classes
    # --------------------------------------------------------

    class_names = load_class_names()

    print(
        f"Classes loaded: "
        f"{len(class_names)}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(
        class_names
    )

    print()

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = predict_image(
        model,
        image_path,
        class_names
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("=" * 55)
    print("TOP 3 PREDICTIONS")
    print("=" * 55)

    for rank, (
        class_name,
        confidence
    ) in enumerate(
        predictions,
        start=1
    ):

        print()
        print(
            f"{rank}. {class_name}"
        )

        print(
            f"   Confidence: "
            f"{confidence * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Best prediction
    # --------------------------------------------------------

    best_class, best_confidence = (
        predictions[0]
    )

    print()
    print("=" * 55)
    print(
        f"Prediction: {best_class}"
    )
    print(
        f"Confidence: "
        f"{best_confidence * 100:.2f}%"
    )
    print("=" * 55)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()