# ============================================================
# BotoniiQ - Medicinal Plant Disease Classification
# Model: EfficientNet-B0
# Framework: PyTorch
#
# Dataset:
#   dataset/disease/train
#   dataset/disease/val
#   dataset/disease/test
#
# Classes: 12
#
# Output:
#   outputs/models/disease_efficientnet_b0_best.pth
# ============================================================

from pathlib import Path
import random
import json
import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_DIR / "dataset" / "disease"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"
PLOT_DIR = OUTPUT_DIR / "plots"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CONFIGURATION
# ============================================================

SEED = 42

IMAGE_SIZE = 224

# RTX 3050 4 GB friendly.
# When CUDA is enabled, we can adjust this if necessary.
BATCH_SIZE = 4

NUM_WORKERS = 0

NUM_EPOCHS = 20

LEARNING_RATE = 0.0001

WEIGHT_DECAY = 0.0001

EARLY_STOPPING_PATIENCE = 5

MODEL_NAME = "disease_efficientnet_b0_best.pth"

MODEL_PATH = MODEL_DIR / MODEL_NAME

CLASS_NAMES_PATH = MODEL_DIR / "disease_class_names.json"

HISTORY_PATH = REPORT_DIR / "disease_training_history.json"


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# 4. DEVICE
# ============================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")

    print("=" * 70)
    print("CUDA GPU detected")
    print("=" * 70)
    print("GPU:", torch.cuda.get_device_name(0))

else:
    DEVICE = torch.device("cpu")

    print("=" * 70)
    print("CUDA GPU not available")
    print("Using CPU")
    print("=" * 70)

print("Device:", DEVICE)
print()


# ============================================================
# 5. CHECK DATASET
# ============================================================

print("=" * 70)
print("CHECKING DATASET")
print("=" * 70)

for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:

    if not folder.exists():
        raise FileNotFoundError(
            f"Dataset folder not found:\n{folder}"
        )

    print("Found:", folder)

print()


# ============================================================
# 6. IMAGE TRANSFORMS
# ============================================================

# Training augmentation
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomVerticalFlip(p=0.2),

    transforms.RandomRotation(degrees=20),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Validation/test transforms
eval_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 7. LOAD DATASETS
# ============================================================

print("=" * 70)
print("LOADING DATASETS")
print("=" * 70)

train_dataset = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
    root=VAL_DIR,
    transform=eval_transforms
)

test_dataset = datasets.ImageFolder(
    root=TEST_DIR,
    transform=eval_transforms
)


# ============================================================
# 8. VERIFY CLASSES
# ============================================================

class_names = train_dataset.classes

num_classes = len(class_names)

print("Number of classes:", num_classes)

print("\nClasses:")

for index, class_name in enumerate(class_names):
    print(f"{index:2d} -> {class_name}")

print()


# Make sure all datasets use exactly the same class mapping.
if train_dataset.class_to_idx != val_dataset.class_to_idx:
    raise RuntimeError(
        "ERROR: Train and validation class mappings are different."
    )

if train_dataset.class_to_idx != test_dataset.class_to_idx:
    raise RuntimeError(
        "ERROR: Train and test class mappings are different."
    )


# ============================================================
# 9. DATASET COUNTS
# ============================================================

print("=" * 70)
print("DATASET SIZE")
print("=" * 70)

print("Training images  :", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images      :", len(test_dataset))
print("Total images     :", len(train_dataset) +
      len(val_dataset) +
      len(test_dataset))

print()


# ============================================================
# 10. SAVE CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as file:
    json.dump(
        {
            "classes": class_names,
            "class_to_idx": train_dataset.class_to_idx
        },
        file,
        indent=4
    )

print("Class mapping saved to:")
print(CLASS_NAMES_PATH)
print()


# ============================================================
# 11. DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# 12. LOAD PRETRAINED EFFICIENTNET-B0
# ============================================================

print("=" * 70)
print("LOADING EFFICIENTNET-B0")
print("=" * 70)

try:

    weights = models.EfficientNet_B0_Weights.DEFAULT

    model = models.efficientnet_b0(
        weights=weights
    )

    print("Pretrained ImageNet weights loaded.")

except Exception as error:

    print("WARNING: Could not load pretrained weights.")

    print("Reason:", error)

    print("Creating EfficientNet-B0 without pretrained weights.")

    model = models.efficientnet_b0(
        weights=None
    )


# ============================================================
# 13. REPLACE CLASSIFIER
# ============================================================

input_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    input_features,
    num_classes
)

model = model.to(DEVICE)

print()
print("EfficientNet-B0 configured.")
print("Input image size:", IMAGE_SIZE)
print("Number of classes:", num_classes)
print()


# ============================================================
# 14. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 15. OPTIMIZER
# ============================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# 16. LEARNING RATE SCHEDULER
# ============================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# 17. TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            (predictions == labels)
            .sum()
            .item()
        )

        total += labels.size(0)

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# 18. VALIDATION FUNCTION
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                (predictions == labels)
                .sum()
                .item()
            )

            total += labels.size(0)

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# 19. TRAINING LOOP
# ============================================================

print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

print("Model       : EfficientNet-B0")
print("Classes     :", num_classes)
print("Batch size  :", BATCH_SIZE)
print("Epochs      :", NUM_EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("Device      :", DEVICE)
print()

best_val_accuracy = 0.0

best_model_state = None

epochs_without_improvement = 0

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
    "learning_rate": []
}


training_start_time = time.time()


for epoch in range(NUM_EPOCHS):

    epoch_start_time = time.time()

    print("-" * 70)

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )

    # -------------------------
    # Training
    # -------------------------

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        DEVICE
    )

    # -------------------------
    # Validation
    # -------------------------

    val_loss, val_accuracy = evaluate(
        model,
        val_loader,
        criterion,
        DEVICE
    )

    # -------------------------
    # Scheduler
    # -------------------------

    scheduler.step(val_accuracy)

    current_lr = optimizer.param_groups[0]["lr"]

    # -------------------------
    # Save history
    # -------------------------

    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_accuracy"].append(
        val_accuracy
    )

    history["learning_rate"].append(
        current_lr
    )

    epoch_time = time.time() - epoch_start_time

    # -------------------------
    # Print results
    # -------------------------

    print(
        f"Train Loss     : {train_loss:.4f}"
    )

    print(
        f"Train Accuracy : {train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss       : {val_loss:.4f}"
    )

    print(
        f"Val Accuracy   : {val_accuracy * 100:.2f}%"
    )

    print(
        f"Learning Rate  : {current_lr:.8f}"
    )

    print(
        f"Epoch Time     : {epoch_time:.1f} seconds"
    )

    # -------------------------
    # Best model
    # -------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        checkpoint = {
            "model_name": "EfficientNet-B0",

            "num_classes": num_classes,

            "class_names": class_names,

            "class_to_idx": train_dataset.class_to_idx,

            "image_size": IMAGE_SIZE,

            "state_dict": model.state_dict(),

            "val_accuracy": best_val_accuracy,

            "epoch": epoch + 1
        }

        torch.save(
            checkpoint,
            MODEL_PATH
        )

        print()
        print("✓ NEW BEST MODEL SAVED")

        print(
            f"Best Validation Accuracy: "
            f"{best_val_accuracy * 100:.2f}%"
        )

        print(
            f"Saved to: {MODEL_PATH}"
        )

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement for "
            f"{epochs_without_improvement} epoch(s)."
        )

    # -------------------------
    # Early stopping
    # -------------------------

    if (
        epochs_without_improvement
        >= EARLY_STOPPING_PATIENCE
    ):

        print()
        print(
            "Early stopping triggered."
        )

        break


# ============================================================
# 20. RESTORE BEST MODEL
# ============================================================

if best_model_state is not None:

    model.load_state_dict(
        best_model_state
    )


# ============================================================
# 21. SAVE TRAINING HISTORY
# ============================================================

history["best_val_accuracy"] = (
    best_val_accuracy
)

history["completed_epochs"] = len(
    history["train_loss"]
)

with open(
    HISTORY_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        history,
        file,
        indent=4
    )


# ============================================================
# 22. FINAL TEST EVALUATION
# ============================================================

print()
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

test_loss, test_accuracy = evaluate(
    model,
    test_loader,
    criterion,
    DEVICE
)

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy * 100:.2f}%"
)

print()


# ============================================================
# 23. TRAINING SUMMARY
# ============================================================

total_training_time = (
    time.time() - training_start_time
)

print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Final Test Accuracy     : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Training Time           : "
    f"{total_training_time / 60:.2f} minutes"
)

print()

print("Best model:")
print(MODEL_PATH)

print()

print("Class names:")
print(CLASS_NAMES_PATH)

print()

print("Training history:")
print(HISTORY_PATH)

print()

print("=" * 70)
print("NEXT STEP")
print("=" * 70)

print(
    "Run evaluate_disease.py to generate "
    "classification report, confusion matrix "
    "and Top-3 accuracy."
)

print("=" * 70)