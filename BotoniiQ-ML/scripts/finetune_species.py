from pathlib import Path
import time

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from torchvision.models import efficientnet_b0

from sklearn.utils.class_weight import compute_class_weight

import numpy as np


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

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "models"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
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
print("BOTONIIQ - SPECIES MODEL FINE-TUNING")
print("=" * 70)

print()

print("Device:", DEVICE)

print()


# ============================================================
# 3. SETTINGS
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 16

NUM_WORKERS = 0

NUM_EPOCHS = 5

LEARNING_RATE = 0.0001

RANDOM_SEED = 42


torch.manual_seed(
    RANDOM_SEED
)


# ============================================================
# 4. TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.75, 1.0)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=15
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
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


val_transform = transforms.Compose([

    transforms.Resize(
        256
    ),

    transforms.CenterCrop(
        IMAGE_SIZE
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
# 5. LOAD DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(

    DATASET_DIR / "train",

    transform=train_transform
)


val_dataset = datasets.ImageFolder(

    DATASET_DIR / "val",

    transform=val_transform
)


NUM_CLASSES = len(
    train_dataset.classes
)


print("Dataset information")
print("-" * 70)

print(
    "Training images   :",
    len(train_dataset)
)

print(
    "Validation images :",
    len(val_dataset)
)

print(
    "Classes           :",
    NUM_CLASSES
)

print()


# ============================================================
# 6. CALCULATE CLASS WEIGHTS
# ============================================================

print("Calculating class weights...")

train_labels = np.array(
    train_dataset.targets
)


class_weights = compute_class_weight(

    class_weight="balanced",

    classes=np.arange(
        NUM_CLASSES
    ),

    y=train_labels
)


class_weights = torch.tensor(

    class_weights,

    dtype=torch.float32
)


class_weights = class_weights.to(
    DEVICE
)


print(
    "Class weights calculated."
)

print()


# ============================================================
# 7. DATA LOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS
)


# ============================================================
# 8. LOAD EXISTING MODEL
# ============================================================

print("Loading existing best model...")

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

    checkpoint[
        "model_state_dict"
    ]
)


print(
    "Existing model loaded."
)

print()


# ============================================================
# 9. FREEZE EVERYTHING FIRST
# ============================================================

for parameter in model.parameters():

    parameter.requires_grad = False


# ============================================================
# 10. UNFREEZE LAST FEATURES
# ============================================================

print(
    "Unfreezing final EfficientNet layers..."
)


for parameter in model.features[6:].parameters():

    parameter.requires_grad = True


# ============================================================
# 11. CLASSIFIER MUST BE TRAINABLE
# ============================================================

for parameter in model.classifier.parameters():

    parameter.requires_grad = True


model = model.to(
    DEVICE
)


# ============================================================
# 12. COUNT TRAINABLE PARAMETERS
# ============================================================

trainable_parameters = sum(

    parameter.numel()

    for parameter in model.parameters()

    if parameter.requires_grad
)


print(
    "Trainable parameters:",
    f"{trainable_parameters:,}"
)

print()


# ============================================================
# 13. WEIGHTED LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(

    weight=class_weights
)


# ============================================================
# 14. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(

    filter(
        lambda parameter:
        parameter.requires_grad,

        model.parameters()
    ),

    lr=LEARNING_RATE
)


# ============================================================
# 15. TRAIN FUNCTION
# ============================================================

def train_one_epoch(

    model,

    loader,

    criterion,

    optimizer

):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        optimizer.zero_grad()


        outputs = model(
            images
        )


        loss = criterion(

            outputs,

            labels
        )


        loss.backward()


        optimizer.step()


        running_loss += (

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


    loss_value = (

        running_loss / total
    )


    accuracy = (

        correct / total
    )


    return (

        loss_value,

        accuracy
    )


# ============================================================
# 16. VALIDATION FUNCTION
# ============================================================

def validate(

    model,

    loader,

    criterion

):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0


    with torch.no_grad():

        for images, labels in loader:

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


            running_loss += (

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


    loss_value = (

        running_loss / total
    )


    accuracy = (

        correct / total
    )


    return (

        loss_value,

        accuracy
    )


# ============================================================
# 17. FINE-TUNING LOOP
# ============================================================

best_val_accuracy = 0.0

best_model_path = (

    OUTPUT_DIR
    / "species_finetuned_best.pth"
)


print("=" * 70)
print("STARTING FINE-TUNING")
print("=" * 70)

print()


for epoch in range(
    NUM_EPOCHS
):

    start_time = time.time()


    train_loss, train_accuracy = (

        train_one_epoch(

            model,

            train_loader,

            criterion,

            optimizer
        )
    )


    val_loss, val_accuracy = (

        validate(

            model,

            val_loader,

            criterion
        )
    )


    elapsed = (

        time.time()
        - start_time
    )


    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )

    print(
        f"Train Loss      : "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy  : "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss        : "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Accuracy    : "
        f"{val_accuracy * 100:.2f}%"
    )

    print(
        f"Time            : "
        f"{elapsed:.1f} seconds"
    )

    print()


    if val_accuracy > best_val_accuracy:

        best_val_accuracy = (
            val_accuracy
        )


        torch.save(

            {

                "model_state_dict":
                    model.state_dict(),

                "num_classes":
                    NUM_CLASSES,

                "class_names":
                    train_dataset.classes,

                "image_size":
                    IMAGE_SIZE,

                "architecture":
                    "efficientnet_b0",

                "fine_tuned":
                    True

            },

            best_model_path
        )


        print(
            "✓ Fine-tuned best model saved."
        )

        print()


# ============================================================
# 18. COMPLETE
# ============================================================

print("=" * 70)
print("FINE-TUNING COMPLETE")
print("=" * 70)

print()

print(
    "Best validation accuracy:",
    f"{best_val_accuracy * 100:.2f}%"
)

print()

print(
    "Fine-tuned model:"
)

print(
    best_model_path
)

print()

print("=" * 70)