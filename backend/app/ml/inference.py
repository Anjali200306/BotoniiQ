import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ============================================================
# BOTONIIQ ML INFERENCE ENGINE
# ============================================================


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# SUPPORTED DISEASE SPECIES
# ============================================================

DISEASE_SUPPORTED_SPECIES = {
    "Neem",
    "Tulsi",
    "Kalanchoe",
}


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# CLASS MAPPING
# ============================================================

def load_class_mapping(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    if isinstance(data, dict) and "classes" in data:
        return data["classes"]

    if isinstance(data, dict):

        try:
            return [
                data[str(i)]
                for i in range(len(data))
            ]

        except KeyError:
            pass

    if isinstance(data, list):
        return data

    raise ValueError(
        f"Unsupported class mapping format: {path}"
    )


# ============================================================
# MODEL CREATION
# ============================================================

def create_model(num_classes):

    model = models.efficientnet_b0(
        weights=None
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )

    return model


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path
):

    checkpoint = torch.load(
        checkpoint_path,
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

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# INFERENCE ENGINE
# ============================================================

class PlantInference:

    def __init__(
        self,
        species_model_path,
        species_mapping_path,
        disease_model_path,
        disease_mapping_path
    ):

        print(
            "[BotoniiQ ML] Initializing inference engine..."
        )

        self.device = DEVICE

        # ----------------------------------------------------
        # SPECIES
        # ----------------------------------------------------

        self.species_classes = load_class_mapping(
            species_mapping_path
        )

        print(
            "[BotoniiQ ML] Species classes:",
            len(self.species_classes)
        )

        self.species_model = create_model(
            len(self.species_classes)
        )

        self.species_model = load_checkpoint(
            self.species_model,
            species_model_path
        )

        print(
            "[BotoniiQ ML] Species model loaded."
        )

        # ----------------------------------------------------
        # DISEASE
        # ----------------------------------------------------

        self.disease_classes = load_class_mapping(
            disease_mapping_path
        )

        print(
            "[BotoniiQ ML] Disease classes:",
            len(self.disease_classes)
        )

        self.disease_model = create_model(
            len(self.disease_classes)
        )

        self.disease_model = load_checkpoint(
            self.disease_model,
            disease_model_path
        )

        print(
            "[BotoniiQ ML] Disease model loaded."
        )

    # ========================================================
    # IMAGE
    # ========================================================

    def _prepare_image(self, image):

        tensor = IMAGE_TRANSFORM(
            image
        )

        return tensor.unsqueeze(0).to(
            self.device
        )

    # ========================================================
    # TOP K
    # ========================================================

    def _predict_top_k(
        self,
        model,
        class_names,
        image_tensor,
        top_k=3
    ):

        with torch.no_grad():

            output = model(
                image_tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )

            values, indices = torch.topk(
                probabilities,
                k=min(
                    top_k,
                    len(class_names)
                ),
                dim=1
            )

        predictions = []

        for probability, index in zip(
            values[0],
            indices[0]
        ):

            predictions.append({
                "name": class_names[
                    index.item()
                ],
                "confidence": round(
                    probability.item() * 100,
                    2
                )
            })

        return predictions

    # ========================================================
    # SPECIES
    # ========================================================

    def predict_species(
        self,
        image
    ):

        image_tensor = self._prepare_image(
            image
        )

        return self._predict_top_k(
            self.species_model,
            self.species_classes,
            image_tensor,
            top_k=3
        )

    # ========================================================
    # DISEASE
    # ========================================================

    def predict_disease(
        self,
        image,
        species_name
    ):

        if species_name not in DISEASE_SUPPORTED_SPECIES:

            return {
                "available": False,
                "reason": (
                    f"Disease model does not "
                    f"currently support "
                    f"{species_name}."
                ),
                "predictions": [],
                "final": None
            }

        image_tensor = self._prepare_image(
            image
        )

        predictions = self._predict_top_k(
            self.disease_model,
            self.disease_classes,
            image_tensor,
            top_k=3
        )

        final_prediction = predictions[0]

        # ----------------------------------------------------
        # HEALTH STATUS
        # ----------------------------------------------------

        disease_name = final_prediction["name"]

        is_healthy = (
            disease_name.endswith("_Healthy")
        )

        return {
            "available": True,
            "reason": None,
            "predictions": predictions,
            "final": final_prediction,
            "is_healthy": is_healthy
        }

    # ========================================================
    # COMPLETE PLANT PREDICTION
    # ========================================================

    def predict(
        self,
        image_path
    ):

        image_path = Path(
            image_path
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------------------
        # SPECIES
        # ----------------------------------------------------

        species_predictions = (
            self.predict_species(image)
        )

        final_species = (
            species_predictions[0]
        )

        species_name = (
            final_species["name"]
        )

        # ----------------------------------------------------
        # DISEASE
        # ----------------------------------------------------

        disease_result = (
            self.predict_disease(
                image,
                species_name
            )
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        return {
            "device": str(self.device),

            "species": {
                "final": final_species,
                "top_3": species_predictions
            },

            "disease": disease_result
        }