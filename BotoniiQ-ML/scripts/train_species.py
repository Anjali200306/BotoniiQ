from pathlib import Path
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(r"D:\BotoniiQ\BotoniiQ-ML")

DATASET_DIR = PROJECT_DIR / "dataset" / "species"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "models"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("BOTONIIQ - SPECIES CLASSIFIER TRAINING")
print("=" * 70)

print()
print("Device:", DEVICE)
print()


# ============================================================
# 3. TRAINING SETTINGS
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 16

NUM_WORKERS = 0

NUM_EPOCHS = 5

LEARNING_RATE = 0.001

RANDOM_SEED = 42


torch.manual_seed(RANDOM_SEED)


# ============================================================
# 4. IMAGE TRANSFORMS
# ============================================================

weights = EfficientNet_B0_Weights.DEFAULT

imagenet_mean = [
    0.485,
    0.456,
    0.406
]

imagenet_std = [
    0.229,
    0.224,
    0.225
]


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
        mean=imagenet_mean,
        std=imagenet_std
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
        mean=imagenet_mean,
        std=imagenet_std
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


test_dataset = datasets.ImageFolder(
    DATASET_DIR / "test",
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
    "Test images       :",
    len(test_dataset)
)

print(
    "Number of classes :",
    NUM_CLASSES
)

print()


# ============================================================
# 6. SAVE CLASS MAPPING
# ============================================================

class_names = train_dataset.classes

class_mapping = {
    str(index): class_name
    for index, class_name in enumerate(class_names)
}


class_mapping_path = (
    OUTPUT_DIR / "species_class_names.json"
)


with open(
    class_mapping_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        class_mapping,
        file,
        indent=4,
        ensure_ascii=False
    )


print(
    "Class mapping saved:",
    class_mapping_path
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


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# 8. LOAD PRETRAINED EFFICIENTNET-B0
# ============================================================

print("Loading EfficientNet-B0...")

model = efficientnet_b0(
    weights=weights
)


# ============================================================
# 9. FREEZE BACKBONE
# ============================================================

for parameter in model.features.parameters():

    parameter.requires_grad = False


# ============================================================
# 10. REPLACE CLASSIFIER
# ============================================================

input_features = (
    model.classifier[1].in_features
)


model.classifier[1] = nn.Linear(
    input_features,
    NUM_CLASSES
)


model = model.to(DEVICE)


print(
    "EfficientNet-B0 loaded."
)

print(
    "Classifier output classes:",
    NUM_CLASSES
)

print()


# ============================================================
# 11. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 12. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 13. TRAINING FUNCTION
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

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)


        optimizer.zero_grad()


        outputs = model(images)


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


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )


    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# 14. VALIDATION FUNCTION
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

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            running_loss += (
                loss.item()
                * images.size(0)
            )


            _, predicted = torch.max(
                outputs,
                1
            )


            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()


    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )


    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# 15. TRAINING LOOP
# ============================================================

best_val_accuracy = 0.0

best_model_path = (
    OUTPUT_DIR / "species_best.pth"
)


print("=" * 70)
print("STARTING TRAINING")
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
        time.time() - start_time
    )


    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )

    print(
        f"Train Loss      : {train_loss:.4f}"
    )

    print(
        f"Train Accuracy  : {train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss        : {val_loss:.4f}"
    )

    print(
        f"Val Accuracy    : {val_accuracy * 100:.2f}%"
    )

    print(
        f"Time            : {elapsed:.1f} seconds"
    )

    print()


    if val_accuracy > best_val_accuracy:

        best_val_accuracy = (
            val_accuracy
        )


        torch.save(
            {
                "model_state_dict": model.state_dict(),

                "num_classes": NUM_CLASSES,

                "class_names": class_names,

                "image_size": IMAGE_SIZE,

                "architecture": "efficientnet_b0"
            },
            best_model_path
        )


        print(
            "✓ Best model saved."
        )

        print()


# ============================================================
# 16. FINISHED
# ============================================================

print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print()

print(
    "Best validation accuracy:",
    f"{best_val_accuracy * 100:.2f}%"
)

print()

print(
    "Best model:",
    best_model_path
)

print(
    "Class mapping:",
    class_mapping_path
)

print()

print("=" * 70)