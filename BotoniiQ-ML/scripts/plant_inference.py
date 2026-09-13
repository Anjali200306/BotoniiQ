import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ============================================================
# BOTONIIQ - PRODUCTION PLANT INFERENCE
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "outputs" / "models"


# ============================================================
# MODEL PATHS
# ============================================================

SPECIES_MODEL_PATH = MODEL_DIR / "species_v4_best.pth"
SPECIES_MAPPING_PATH = MODEL_DIR / "species_v4_class_names.json"

DISEASE_MODEL_PATH = MODEL_DIR / "disease_efficientnet_b0_best.pth"
DISEASE_MAPPING_PATH = MODEL_DIR / "disease_class_names.json"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DISEASE MODEL SUPPORTED SPECIES
# ============================================================

DISEASE_SUPPORTED_SPECIES = {
    "Neem",
    "Tulsi",
    "Kalanchoe",
}


# ============================================================
# IMAGE TRANSFORM
# ============================================================

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# HELPER - LOAD CLASS MAPPING
# ============================================================

def load_class_mapping(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Format:
    # {"classes": ["Aloevera", "Amla", ...]}
    if isinstance(data, dict) and "classes" in data:
        return data["classes"]

    # Format:
    # {"0": "Aloevera", "1": "Amla", ...}
    if isinstance(data, dict):
        try:
            return [
                data[str(i)]
                for i in range(len(data))
            ]
        except KeyError:
            pass

    # Format:
    # ["Aloevera", "Amla", ...]
    if isinstance(data, list):
        return data

    raise ValueError(
        f"Unsupported class mapping format: {path}"
    )


# ============================================================
# HELPER - LOAD EFFICIENTNET-B0
# ============================================================

def create_model(num_classes):
    model = models.efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )

    return model


# ============================================================
# HELPER - LOAD CHECKPOINT
# ============================================================

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
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

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# SPECIES MODEL
# ============================================================

print("[BotoniiQ] Loading species model...")

species_classes = load_class_mapping(
    SPECIES_MAPPING_PATH
)

species_model = create_model(
    len(species_classes)
)

species_model = load_checkpoint(
    species_model,
    SPECIES_MODEL_PATH
)

print(
    f"[BotoniiQ] Species model loaded: "
    f"{len(species_classes)} classes"
)


# ============================================================
# DISEASE MODEL
# ============================================================

print("[BotoniiQ] Loading disease model...")

disease_classes = load_class_mapping(
    DISEASE_MAPPING_PATH
)

disease_model = create_model(
    len(disease_classes)
)

disease_model = load_checkpoint(
    disease_model,
    DISEASE_MODEL_PATH
)

print(
    f"[BotoniiQ] Disease model loaded: "
    f"{len(disease_classes)} classes"
)


# ============================================================
# TOP-K PREDICTIONS
# ============================================================

def get_top_predictions(
    model,
    class_names,
    image_tensor,
    top_k=3
):
    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        values, indices = torch.topk(
            probabilities,
            k=min(top_k, len(class_names)),
            dim=1
        )

    results = []

    for probability, index in zip(
        values[0],
        indices[0]
    ):
        class_name = class_names[
            index.item()
        ]

        confidence = (
            probability.item() * 100
        )

        results.append({
            "name": class_name,
            "confidence": round(
                confidence,
                2
            )
        })

    return results


# ============================================================
# SPECIES PREDICTION
# ============================================================

def predict_species(image):
    """
    Predict plant species using Species V4 model.
    """

    image_tensor = IMAGE_TRANSFORM(
        image
    ).unsqueeze(0).to(DEVICE)

    predictions = get_top_predictions(
        species_model,
        species_classes,
        image_tensor,
        top_k=3
    )

    return predictions


# ============================================================
# DISEASE PREDICTION
# ============================================================

def predict_disease(
    image,
    species_name
):
    """
    Predict disease only when the detected
    species is supported by the disease model.
    """

    if species_name not in DISEASE_SUPPORTED_SPECIES:

        return {
            "available": False,
            "reason": (
                f"Disease model does not currently "
                f"support {species_name}."
            ),
            "predictions": [],
            "final": None,
        }

    image_tensor = IMAGE_TRANSFORM(
        image
    ).unsqueeze(0).to(DEVICE)

    predictions = get_top_predictions(
        disease_model,
        disease_classes,
        image_tensor,
        top_k=3
    )

    return {
        "available": True,
        "reason": None,
        "predictions": predictions,
        "final": predictions[0],
    }


# ============================================================
# COMPLETE PREDICTION
# ============================================================

def predict_plant(image_path):
    """
    Complete BotoniiQ inference pipeline.

    1. Identify species.
    2. Check disease-model compatibility.
    3. Run disease prediction only when supported.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    # --------------------------------------------------------
    # SPECIES
    # --------------------------------------------------------

    species_predictions = predict_species(
        image
    )

    final_species = species_predictions[0]

    species_name = final_species["name"]

    # --------------------------------------------------------
    # DISEASE
    # --------------------------------------------------------

    disease_result = predict_disease(
        image,
        species_name
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    result = {
        "image": str(image_path),

        "device": str(DEVICE),

        "species": {
            "final": final_species,
            "top_3": species_predictions,
        },

        "disease": disease_result,
    }

    return result


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:

        print()
        print(
            "Usage:"
        )
        print(
            "python scripts\\plant_inference.py "
            "\"path\\to\\image.jpg\""
        )
        print()

        sys.exit(1)

    image_path = sys.argv[1]

    print()
    print("=" * 70)
    print("BOTONIIQ PRODUCTION INFERENCE")
    print("=" * 70)

    print()
    print(f"Device: {DEVICE}")
    print(f"Image : {image_path}")

    result = predict_plant(
        image_path
    )

    print()
    print("-" * 70)
    print("SPECIES")
    print("-" * 70)

    for i, prediction in enumerate(
        result["species"]["top_3"],
        start=1
    ):

        print(
            f"{i}. "
            f"{prediction['name']:<30} "
            f"{prediction['confidence']:.2f}%"
        )

    print()
    print(
        f"Final Species : "
        f"{result['species']['final']['name']}"
    )

    print(
        f"Confidence    : "
        f"{result['species']['final']['confidence']:.2f}%"
    )

    print()
    print("-" * 70)
    print("DISEASE")
    print("-" * 70)

    disease = result["disease"]

    if disease["available"]:

        for i, prediction in enumerate(
            disease["predictions"],
            start=1
        ):

            print(
                f"{i}. "
                f"{prediction['name']:<30} "
                f"{prediction['confidence']:.2f}%"
            )

        print()
        print(
            f"Final Disease : "
            f"{disease['final']['name']}"
        )

        print(
            f"Confidence     : "
            f"{disease['final']['confidence']:.2f}%"
        )

    else:

        print(
            "Disease analysis: NOT AVAILABLE"
        )

        print(
            f"Reason: {disease['reason']}"
        )

    print()
    print("=" * 70)
    print("INFERENCE COMPLETED")
    print("=" * 70)
    print()