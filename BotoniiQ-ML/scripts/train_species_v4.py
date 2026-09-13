"""
BOTONIIQ - SPECIES V4 TRAINING
================================

Goal:
Targeted real-world domain adaptation with extra emphasis on Neem.

Starting model:
    outputs/models/species_v3_best.pth

Training data:
    1. Original species training dataset
    2. Real-world training dataset

Important:
    - The 18-image real-world test set is NOT used.
    - V1, V2 and V3 models are NOT modified.
    - V4 is saved as a new checkpoint.
"""

from pathlib import Path
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

ORIGINAL_TRAIN_DIR = PROJECT_DIR / "dataset" / "species" / "train"
REAL_WORLD_TRAIN_DIR = PROJECT_DIR / "dataset" / "real_world_train"

V3_MODEL_PATH = PROJECT_DIR / "outputs" / "models" / "species_v3_best.pth"

V4_MODEL_PATH = PROJECT_DIR / "outputs" / "models" / "species_v4_best.pth"
V4_MAPPING_PATH = PROJECT_DIR / "outputs" / "models" / "species_v4_class_names.json"
V4_SUMMARY_PATH = PROJECT_DIR / "outputs" / "reports" / "species_v4_training_summary.json"


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

IMAGE_SIZE = 224
BATCH_SIZE = 4

EPOCHS = 12
PATIENCE = 4

# Very small learning rates because V4 starts from V3.
BACKBONE_LR = 3e-6
CLASSIFIER_LR = 1e-5

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

# Overall real-world emphasis.
REAL_WORLD_WEIGHT = 5.0

# Additional emphasis specifically for Neem.
NEEM_WEIGHT = 2.0


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("BOTONIIQ SPECIES V4 TRAINING")
print("=" * 70)

print()
print("Device       :", DEVICE)

if torch.cuda.is_available():
    print("GPU          :", torch.cuda.get_device_name(0))

print("Project      :", PROJECT_DIR)
print("V3 model     :", V3_MODEL_PATH)
print()


# ============================================================
# CHECK PATHS
# ============================================================

if not ORIGINAL_TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Original training directory not found:\n{ORIGINAL_TRAIN_DIR}"
    )

if not REAL_WORLD_TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Real-world training directory not found:\n{REAL_WORLD_TRAIN_DIR}"
    )

if not V3_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"V3 model not found:\n{V3_MODEL_PATH}"
    )


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize(256),

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.75, 1.0),
        ratio=(0.75, 1.33)
    ),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomVerticalFlip(p=0.15),

    transforms.RandomRotation(degrees=15),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.03
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD ORIGINAL DATASET
# ============================================================

print("=" * 70)
print("LOADING ORIGINAL DATASET")
print("=" * 70)

original_dataset = datasets.ImageFolder(
    ORIGINAL_TRAIN_DIR,
    transform=train_transform
)

class_names = original_dataset.classes

num_classes = len(class_names)

print()
print("Original training images :", len(original_dataset))
print("Number of classes        :", num_classes)
print()

print("Class mapping:")

for index, name in enumerate(class_names):
    print(f"{index:3d} -> {name}")

print()


# ============================================================
# CUSTOM REAL-WORLD DATASET
# ============================================================

class RealWorldDataset(Dataset):

    VALID_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp"
    }

    def __init__(self, root_dir, class_names, transform=None):

        self.root_dir = Path(root_dir)
        self.class_names = class_names
        self.class_to_idx = {
            name: index
            for index, name in enumerate(class_names)
        }

        self.transform = transform

        self.samples = []

        for class_name in class_names:

            class_dir = self.root_dir / class_name

            if not class_dir.exists():
                continue

            class_index = self.class_to_idx[class_name]

            for file_path in sorted(class_dir.iterdir()):

                if file_path.suffix.lower() not in self.VALID_EXTENSIONS:
                    continue

                self.samples.append(
                    (
                        file_path,
                        class_index,
                        class_name
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        file_path, class_index, class_name = self.samples[index]

        image = Image.open(file_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, class_index, class_name, str(file_path)


# ============================================================
# LOAD REAL-WORLD DATA
# ============================================================

real_world_dataset = RealWorldDataset(
    REAL_WORLD_TRAIN_DIR,
    class_names,
    transform=train_transform
)

print("=" * 70)
print("REAL-WORLD DATASET")
print("=" * 70)

print()
print("Real-world images :", len(real_world_dataset))
print()


# ============================================================
# REAL-WORLD CLASS COUNTS
# ============================================================

real_world_counts = {
    class_name: 0
    for class_name in class_names
}

for _, _, class_name in real_world_dataset.samples:
    real_world_counts[class_name] += 1

print("Real-world class counts:")

for class_name in class_names:
    if real_world_counts[class_name] > 0:
        print(
            f"{class_name:30s}: "
            f"{real_world_counts[class_name]}"
        )

print()


# ============================================================
# COMBINED DATASET
# ============================================================

class CombinedDataset(Dataset):

    def __init__(
        self,
        original_dataset,
        real_world_dataset
    ):

        self.original_dataset = original_dataset
        self.real_world_dataset = real_world_dataset

        self.original_length = len(original_dataset)

    def __len__(self):

        return (
            len(self.original_dataset)
            +
            len(self.real_world_dataset)
        )

    def __getitem__(self, index):

        if index < self.original_length:

            image, label = self.original_dataset[index]

            return (
                image,
                label,
                "original"
            )

        real_index = index - self.original_length

        image, label, class_name, file_path = (
            self.real_world_dataset[real_index]
        )

        return (
            image,
            label,
            "real_world"
        )


combined_dataset = CombinedDataset(
    original_dataset,
    real_world_dataset
)


# ============================================================
# SAMPLE WEIGHTS
# ============================================================

print("=" * 70)
print("BUILDING TARGETED SAMPLER")
print("=" * 70)

sample_weights = []

# ------------------------------------------------------------
# Original dataset
# ------------------------------------------------------------

for _, label in original_dataset.samples:

    # Normal weight for original dataset.
    weight = 1.0

    sample_weights.append(weight)


# ------------------------------------------------------------
# Real-world dataset
# ------------------------------------------------------------

for _, label, class_name in real_world_dataset.samples:

    # Base real-world emphasis.
    weight = REAL_WORLD_WEIGHT

    # Extra emphasis specifically for Neem.
    if class_name.lower() == "neem":
        weight *= NEEM_WEIGHT

    sample_weights.append(weight)


sample_weights = torch.DoubleTensor(sample_weights)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(combined_dataset),
    replacement=True
)


print()
print("Real-world base weight :", REAL_WORLD_WEIGHT)
print("Neem additional weight:", NEEM_WEIGHT)
print("Neem effective weight  :", REAL_WORLD_WEIGHT * NEEM_WEIGHT)
print()


# ============================================================
# DATALOADER
# ============================================================

train_loader = DataLoader(
    combined_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# MODEL
# ============================================================

print("=" * 70)
print("LOADING V3 MODEL")
print("=" * 70)

model = models.efficientnet_b0(
    weights=None
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)


# ============================================================
# LOAD V3 CHECKPOINT
# ============================================================

checkpoint = torch.load(
    V3_MODEL_PATH,
    map_location="cpu"
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


model.load_state_dict(
    state_dict,
    strict=True
)

print()
print("V3 model loaded successfully.")
print()


# ============================================================
# FREEZE BACKBONE
# ============================================================

for parameter in model.features.parameters():
    parameter.requires_grad = False


# ============================================================
# UNFREEZE LAST EFFICIENTNET BLOCKS
# ============================================================

print("=" * 70)
print("V4 FINE-TUNING CONFIGURATION")
print("=" * 70)

print()
print("Freezing early EfficientNet blocks...")

# EfficientNet-B0:
#
# features:
# 0
# 1
# 2
# 3
# 4
# 5
# 6
# 7
# 8
#
# V4 fine-tunes blocks 6, 7 and 8.
#

for index in range(6, len(model.features)):

    for parameter in model.features[index].parameters():

        parameter.requires_grad = True


# Classifier always trainable.

for parameter in model.classifier.parameters():

    parameter.requires_grad = True


# ============================================================
# COUNT PARAMETERS
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print()
print("Total parameters     :", total_parameters)
print("Trainable parameters :", trainable_parameters)
print()


# ============================================================
# MOVE MODEL TO DEVICE
# ============================================================

model = model.to(DEVICE)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# DIFFERENTIAL LEARNING RATES
# ============================================================

backbone_parameters = []
classifier_parameters = []

for name, parameter in model.named_parameters():

    if not parameter.requires_grad:
        continue

    if name.startswith("classifier"):
        classifier_parameters.append(parameter)

    else:
        backbone_parameters.append(parameter)


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
# LR SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ============================================================
# TRAINING INFORMATION
# ============================================================

print("=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)

print()
print("Epochs              :", EPOCHS)
print("Batch size          :", BATCH_SIZE)
print("Backbone LR         :", BACKBONE_LR)
print("Classifier LR       :", CLASSIFIER_LR)
print("Weight decay        :", WEIGHT_DECAY)
print("Early stopping      :", PATIENCE)
print("Real-world weight   :", REAL_WORLD_WEIGHT)
print("Neem multiplier     :", NEEM_WEIGHT)
print()


# ============================================================
# TRAINING LOOP
# ============================================================

best_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0

history = []


for epoch in range(1, EPOCHS + 1):

    epoch_start = time.time()

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    original_seen = 0
    real_world_seen = 0

    # --------------------------------------------------------
    # BATCH LOOP
    # --------------------------------------------------------

    for batch_images, batch_labels, batch_sources in train_loader:

        batch_images = batch_images.to(
            DEVICE,
            non_blocking=True
        )

        batch_labels = batch_labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(batch_images)

        loss = criterion(
            outputs,
            batch_labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        batch_size_actual = batch_labels.size(0)

        running_loss += (
            loss.item()
            *
            batch_size_actual
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == batch_labels
        ).sum().item()

        total += batch_size_actual

        for source in batch_sources:

            if source == "original":
                original_seen += 1

            else:
                real_world_seen += 1


    # --------------------------------------------------------
    # EPOCH METRICS
    # --------------------------------------------------------

    epoch_loss = running_loss / total

    epoch_accuracy = (
        correct / total
    )

    epoch_time = time.time() - epoch_start


    # --------------------------------------------------------
    # LR SCHEDULER
    # --------------------------------------------------------

    scheduler.step(epoch_loss)


    current_backbone_lr = optimizer.param_groups[0]["lr"]
    current_classifier_lr = optimizer.param_groups[1]["lr"]


    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history.append(
        {
            "epoch": epoch,
            "loss": epoch_loss,
            "accuracy": epoch_accuracy,
            "original_seen": original_seen,
            "real_world_seen": real_world_seen,
            "backbone_lr": current_backbone_lr,
            "classifier_lr": current_classifier_lr,
            "time_seconds": epoch_time
        }
    )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print()
    print(
        f"Epoch {epoch:02d}/{EPOCHS}"
    )

    print(
        f"Loss     : {epoch_loss:.4f}"
    )

    print(
        f"Accuracy : {epoch_accuracy * 100:.2f}%"
    )

    print(
        f"Original : {original_seen}"
    )

    print(
        f"Real-world: {real_world_seen}"
    )

    print(
        f"Backbone LR   : {current_backbone_lr:.2e}"
    )

    print(
        f"Classifier LR : {current_classifier_lr:.2e}"
    )

    print(
        f"Time     : {epoch_time:.1f} sec"
    )


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if epoch_loss < best_loss:

        best_loss = epoch_loss

        best_epoch = epoch

        epochs_without_improvement = 0

        checkpoint_to_save = {
            "model_state_dict": model.state_dict(),

            "num_classes": num_classes,

            "class_names": class_names,

            "epoch": epoch,

            "training_loss": epoch_loss,

            "training_accuracy": epoch_accuracy,

            "backbone_lr": current_backbone_lr,

            "classifier_lr": current_classifier_lr,

            "real_world_weight": REAL_WORLD_WEIGHT,

            "neem_weight": NEEM_WEIGHT,

            "source_model": str(
                V3_MODEL_PATH
            )
        }

        torch.save(
            checkpoint_to_save,
            V4_MODEL_PATH
        )

        print()
        print(">>> BEST V4 MODEL SAVED")
        print(
            "    ",
            V4_MODEL_PATH
        )

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement "
            f"({epochs_without_improvement}/{PATIENCE})"
        )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print()
        print(
            "Early stopping triggered."
        )

        break


# ============================================================
# SAVE CLASS MAPPING
# ============================================================

V4_MAPPING_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    V4_MAPPING_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "classes": class_names
        },
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# SAVE TRAINING SUMMARY
# ============================================================

V4_SUMMARY_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

summary = {

    "model": "EfficientNet-B0",

    "version": "V4",

    "task": "71-class plant species classification",

    "device": str(DEVICE),

    "starting_model": str(
        V3_MODEL_PATH
    ),

    "output_model": str(
        V4_MODEL_PATH
    ),

    "original_training_images": len(
        original_dataset
    ),

    "real_world_training_images": len(
        real_world_dataset
    ),

    "combined_dataset_size": len(
        combined_dataset
    ),

    "num_classes": num_classes,

    "class_names": class_names,

    "real_world_class_counts": real_world_counts,

    "real_world_weight": REAL_WORLD_WEIGHT,

    "neem_weight": NEEM_WEIGHT,

    "effective_neem_weight": (
        REAL_WORLD_WEIGHT *
        NEEM_WEIGHT
    ),

    "batch_size": BATCH_SIZE,

    "epochs_requested": EPOCHS,

    "epochs_completed": len(history),

    "best_epoch": best_epoch,

    "best_training_loss": best_loss,

    "backbone_lr": BACKBONE_LR,

    "classifier_lr": CLASSIFIER_LR,

    "weight_decay": WEIGHT_DECAY,

    "trainable_parameters": trainable_parameters,

    "total_parameters": total_parameters,

    "history": history
}


with open(
    V4_SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("V4 TRAINING COMPLETED")
print("=" * 70)

print()
print("Best epoch :", best_epoch)

print(
    "Best loss  :",
    f"{best_loss:.4f}"
)

print()

print(
    "V4 model:"
)

print(
    V4_MODEL_PATH
)

print()

print(
    "Class mapping:"
)

print(
    V4_MAPPING_PATH
)

print()

print(
    "Training summary:"
)

print(
    V4_SUMMARY_PATH
)

print()

print("=" * 70)
print("IMPORTANT")
print("=" * 70)

print()
print("DO NOT evaluate V4 on the training dataset.")
print()
print("Next step:")
print("1. Evaluate V4 on the original 71-class test set.")
print("2. Evaluate V4 on the untouched real-world test set.")
print("3. Compare V3 vs V4.")
print()

print("=" * 70)