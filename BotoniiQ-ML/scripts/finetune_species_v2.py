from pathlib import Path
import json
import time
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0

from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(r"D:\BotoniiQ\BotoniiQ-ML")

DATASET_DIR = PROJECT_DIR / "dataset" / "species"

INPUT_MODEL_PATH = (
    PROJECT_DIR
    / "outputs"
    / "models"
    / "species_finetuned_best.pth"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "models"
)

REPORT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "reports"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. OUTPUT FILES
# ============================================================

BEST_MODEL_PATH = (
    OUTPUT_DIR
    / "species_v2_best.pth"
)

CLASS_NAMES_PATH = (
    OUTPUT_DIR
    / "species_v2_class_names.json"
)

HISTORY_PATH = (
    REPORT_DIR
    / "species_v2_training_history.json"
)


# ============================================================
# 3. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 4. SETTINGS
# ============================================================

IMAGE_SIZE = 224

# RTX 3050 4 GB:
# use a conservative batch size
BATCH_SIZE = 4

NUM_WORKERS = 0

NUM_EPOCHS = 15

# Small LR for already-trained backbone
BACKBONE_LR = 0.00001

# Larger LR for classifier
CLASSIFIER_LR = 0.00005

WEIGHT_DECAY = 0.0001

PATIENCE = 4

RANDOM_SEED = 42


# ============================================================
# 5. REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# 6. PRINT HEADER
# ============================================================

print("=" * 75)
print("BOTONIIQ - SPECIES MODEL V2 FINE-TUNING")
print("=" * 75)

print()
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()


# ============================================================
# 7. IMAGE NORMALIZATION
# ============================================================

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


# ============================================================
# 8. V2 TRAINING TRANSFORM
# ============================================================

train_transform = transforms.Compose([

    # Different crop sizes and positions
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.60, 1.0),
        ratio=(0.75, 1.33)
    ),

    # Orientation variation
    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomVerticalFlip(
        p=0.2
    ),

    # Camera angle / positioning variation
    transforms.RandomRotation(
        degrees=25
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.08, 0.08),
        scale=(0.90, 1.10),
        shear=8
    ),

    # Lighting and camera variation
    transforms.ColorJitter(
        brightness=0.30,
        contrast=0.30,
        saturation=0.25,
        hue=0.06
    ),

    # Occasional camera blur
    transforms.RandomApply(
        [
            transforms.GaussianBlur(
                kernel_size=3,
                sigma=(0.1, 1.5)
            )
        ],
        p=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


# ============================================================
# 9. VALIDATION TRANSFORM
# ============================================================

# IMPORTANT:
# Keep validation deterministic.

val_transform = transforms.Compose([

    transforms.Resize(
        256
    ),

    transforms.CenterCrop(
        IMAGE_SIZE
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


# ============================================================
# 10. LOAD DATASETS
# ============================================================

print("=" * 75)
print("LOADING DATASET")
print("=" * 75)

print()

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
# 11. VERIFY MODEL/DATASET COMPATIBILITY
# ============================================================

if not INPUT_MODEL_PATH.exists():

    raise FileNotFoundError(
        f"\nStarting model not found:\n"
        f"{INPUT_MODEL_PATH}\n"
    )


# ============================================================
# 12. SAVE CLASS MAPPING
# ============================================================

class_names = train_dataset.classes

class_mapping = {
    str(index): class_name
    for index, class_name
    in enumerate(class_names)
}

with open(
    CLASS_NAMES_PATH,
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
    "Class mapping saved:"
)

print(
    CLASS_NAMES_PATH
)

print()


# ============================================================
# 13. DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda")
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda")
)


# ============================================================
# 14. CLASS WEIGHTS
# ============================================================

print("=" * 75)
print("CALCULATING CLASS WEIGHTS")
print("=" * 75)

print()

train_labels = np.array(
    train_dataset.targets
)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_labels
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE
)

print("Class weights calculated.")

print()


# ============================================================
# 15. LOAD EXISTING FINE-TUNED MODEL
# ============================================================

print("=" * 75)
print("LOADING EXISTING SPECIES MODEL")
print("=" * 75)

print()

print(
    "Starting model:"
)

print(
    INPUT_MODEL_PATH
)

print()

checkpoint = torch.load(
    INPUT_MODEL_PATH,
    map_location=DEVICE
)


# ============================================================
# 16. CREATE EFFICIENTNET-B0
# ============================================================

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


# ============================================================
# 17. LOAD CHECKPOINT
# ============================================================

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


# Remove DataParallel prefix if present

cleaned_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):

        key = key[7:]

    cleaned_state_dict[key] = value


model.load_state_dict(
    cleaned_state_dict
)

print()
print(
    "Existing fine-tuned model loaded successfully."
)

print()


# ============================================================
# 18. FREEZE EVERYTHING
# ============================================================

for parameter in model.parameters():

    parameter.requires_grad = False


# ============================================================
# 19. UNFREEZE MORE EFFICIENTNET FEATURES
# ============================================================

print("=" * 75)
print("CONFIGURING V2 FINE-TUNING")
print("=" * 75)

print()

print(
    "Unfreezing EfficientNet feature blocks 4, 5, 6, 7 and 8..."
)

for parameter in model.features[4:].parameters():

    parameter.requires_grad = True


# ============================================================
# 20. KEEP CLASSIFIER TRAINABLE
# ============================================================

for parameter in model.classifier.parameters():

    parameter.requires_grad = True


# ============================================================
# 21. MOVE MODEL TO DEVICE
# ============================================================

model = model.to(
    DEVICE
)


# ============================================================
# 22. TRAINABLE PARAMETER COUNT
# ============================================================

trainable_parameters = sum(

    parameter.numel()

    for parameter in model.parameters()

    if parameter.requires_grad
)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(
    "Total parameters     :",
    f"{total_parameters:,}"
)

print(
    "Trainable parameters :",
    f"{trainable_parameters:,}"
)

print()


# ============================================================
# 23. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1
)


# ============================================================
# 24. DIFFERENTIAL LEARNING RATES
# ============================================================

backbone_parameters = []
classifier_parameters = []

for name, parameter in model.named_parameters():

    if not parameter.requires_grad:
        continue

    if name.startswith("classifier"):

        classifier_parameters.append(
            parameter
        )

    else:

        backbone_parameters.append(
            parameter
        )


optimizer = torch.optim.AdamW(

    [
        {
            "params": backbone_parameters,
            "lr": BACKBONE_LR
        },

        {
            "params": classifier_parameters,
            "lr": CLASSIFIER_LR
        }
    ],

    weight_decay=WEIGHT_DECAY
)


# ============================================================
# 25. LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=2,

    min_lr=1e-7
)


# ============================================================
# 26. TRAIN FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    for images, labels in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        # Prevent very large gradient updates
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
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
# 27. VALIDATION FUNCTION
# ============================================================

def validate():

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
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

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            total += labels.size(0)

            correct += (
                predictions == labels
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
# 28. TRAINING
# ============================================================

print("=" * 75)
print("STARTING SPECIES V2 TRAINING")
print("=" * 75)

print()

print("Epochs              :", NUM_EPOCHS)
print("Batch size          :", BATCH_SIZE)
print("Backbone LR         :", BACKBONE_LR)
print("Classifier LR       :", CLASSIFIER_LR)
print("Weight decay        :", WEIGHT_DECAY)
print("Early stopping      :", PATIENCE)
print()


best_val_accuracy = 0.0

epochs_without_improvement = 0

history = []


# ============================================================
# 29. TRAINING LOOP
# ============================================================

for epoch in range(NUM_EPOCHS):

    start_time = time.time()

    train_loss, train_accuracy = (
        train_one_epoch()
    )

    val_loss, val_accuracy = (
        validate()
    )

    scheduler.step(
        val_accuracy
    )

    elapsed = (
        time.time()
        - start_time
    )

    current_backbone_lr = (
        optimizer.param_groups[0]["lr"]
    )

    current_classifier_lr = (
        optimizer.param_groups[1]["lr"]
    )

    print("=" * 75)

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )

    print(
        f"Train Loss       : "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy   : "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss         : "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Accuracy     : "
        f"{val_accuracy * 100:.2f}%"
    )

    print(
        f"Backbone LR       : "
        f"{current_backbone_lr:.8f}"
    )

    print(
        f"Classifier LR     : "
        f"{current_classifier_lr:.8f}"
    )

    print(
        f"Time              : "
        f"{elapsed:.1f} seconds"
    )

    print()

    history.append({

        "epoch": epoch + 1,

        "train_loss": train_loss,

        "train_accuracy": train_accuracy,

        "val_loss": val_loss,

        "val_accuracy": val_accuracy,

        "backbone_lr": current_backbone_lr,

        "classifier_lr": current_classifier_lr,

        "time_seconds": elapsed

    })


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = (
            val_accuracy
        )

        epochs_without_improvement = 0

        torch.save(

            {
                "model_state_dict":
                    model.state_dict(),

                "num_classes":
                    NUM_CLASSES,

                "class_names":
                    class_names,

                "image_size":
                    IMAGE_SIZE,

                "architecture":
                    "efficientnet_b0",

                "version":
                    "species_v2",

                "best_val_accuracy":
                    best_val_accuracy

            },

            BEST_MODEL_PATH
        )

        print(
            "✓ NEW BEST V2 MODEL SAVED"
        )

        print(
            BEST_MODEL_PATH
        )

        print()

    else:

        epochs_without_improvement += 1

        print(
            "No validation improvement."
        )

        print(
            "Early stopping counter:",
            f"{epochs_without_improvement}/{PATIENCE}"
        )

        print()


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if epochs_without_improvement >= PATIENCE:

        print("=" * 75)
        print("EARLY STOPPING")
        print("=" * 75)

        print()

        print(
            "No validation improvement for",
            PATIENCE,
            "epochs."
        )

        print()

        break


# ============================================================
# 30. SAVE TRAINING HISTORY
# ============================================================

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
# 31. COMPLETE
# ============================================================

print("=" * 75)
print("SPECIES V2 TRAINING COMPLETE")
print("=" * 75)

print()

print(
    "Best validation accuracy:",
    f"{best_val_accuracy * 100:.2f}%"
)

print()

print(
    "V2 model:"
)

print(
    BEST_MODEL_PATH
)

print()

print(
    "Class mapping:"
)

print(
    CLASS_NAMES_PATH
)

print()

print(
    "Training history:"
)

print(
    HISTORY_PATH
)

print()

print("=" * 75)