from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from torchvision.models import efficientnet_b0


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(
    r"D:\BotoniiQ\BotoniiQ-ML"
)

DATASET_DIR = (
    PROJECT_DIR
    / "dataset"
    / "species"
)

MODEL_PATH = (
    PROJECT_DIR
    / "outputs"
    / "models"
    / "species_best.pth"
)


# ============================================================
# 2. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("BOTONIIQ - SPECIES MODEL EVALUATION")
print("=" * 70)

print()

print("Device:", DEVICE)

print()


# ============================================================
# 3. TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        256
    ),

    transforms.CenterCrop(
        224
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# 4. TEST DATASET
# ============================================================

test_dataset = datasets.ImageFolder(

    DATASET_DIR / "test",

    transform=transform
)


test_loader = DataLoader(

    test_dataset,

    batch_size=16,

    shuffle=False,

    num_workers=0
)


NUM_CLASSES = len(
    test_dataset.classes
)


print(
    "Test images:",
    len(test_dataset)
)

print(
    "Number of classes:",
    NUM_CLASSES
)

print()


# ============================================================
# 5. LOAD MODEL
# ============================================================

print("Loading trained model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


model = efficientnet_b0(
    weights=None
)


input_features = (
    model.classifier[1].in_features
)


model.classifier[1] = nn.Linear(
    input_features,
    NUM_CLASSES
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model = model.to(DEVICE)


model.eval()


print("Model loaded successfully.")

print()


# ============================================================
# 6. TEST EVALUATION
# ============================================================

criterion = nn.CrossEntropyLoss()


total_loss = 0.0

correct = 0

total = 0


print("=" * 70)
print("EVALUATING TEST SET")
print("=" * 70)

print()


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            labels
        )


        total_loss += (
            loss.item()
            * images.size(0)
        )


        _, predictions = torch.max(
            outputs,
            1
        )


        total += labels.size(0)


        correct += (
            predictions == labels
        ).sum().item()


# ============================================================
# 7. RESULTS
# ============================================================

test_loss = (
    total_loss / total
)


test_accuracy = (
    correct / total
)


print(
    "Test Loss     :",
    f"{test_loss:.4f}"
)

print(
    "Test Accuracy :",
    f"{test_accuracy * 100:.2f}%"
)

print()


print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)