from pathlib import Path
import random
import time
import json
import shutil

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

ORIGINAL_TRAIN_DIR = PROJECT_DIR / "dataset" / "species" / "train"
REAL_WORLD_TRAIN_DIR = PROJECT_DIR / "dataset" / "real_world_train"

MODEL_PATH = PROJECT_DIR / "outputs" / "models" / "species_v2_best.pth"
CLASS_MAPPING_PATH = PROJECT_DIR / "outputs" / "models" / "species_v2_class_names.json"

OUTPUT_MODEL = PROJECT_DIR / "outputs" / "models" / "species_v3_best.pth"
OUTPUT_MAPPING = PROJECT_DIR / "outputs" / "models" / "species_v3_class_names.json"

OUTPUT_REPORT = PROJECT_DIR / "outputs" / "reports" / "species_v3_training_summary.json"


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

IMAGE_SIZE = 224
BATCH_SIZE = 4

EPOCHS = 12
PATIENCE = 4

BACKBONE_LR = 5e-6
CLASSIFIER_LR = 2e-5

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

# Real-world images receive more sampling weight.
REAL_WORLD_WEIGHT = 8.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 70)
print("BOTONIIQ SPECIES V3 - REAL-WORLD DOMAIN ADAPTATION")
print("=" * 70)

print()
print(f"Device       : {DEVICE}")
print(f"Original     : {ORIGINAL_TRAIN_DIR}")
print(f"Real-world   : {REAL_WORLD_TRAIN_DIR}")
print(f"Starting     : {MODEL_PATH}")
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

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"V2 model not found:\n{MODEL_PATH}"
    )

if not CLASS_MAPPING_PATH.exists():
    raise FileNotFoundError(
        f"V2 class mapping not found:\n{CLASS_MAPPING_PATH}"
    )


# ============================================================
# LOAD CLASS MAPPING
# ============================================================

with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
    mapping_data = json.load(f)


if isinstance(mapping_data, dict) and "classes" in mapping_data:
    class_names = mapping_data["classes"]

elif isinstance(mapping_data, dict):
    try:
        class_names = [
            mapping_data[str(i)]
            for i in range(len(mapping_data))
        ]
    except KeyError:
        class_names = list(mapping_data.values())

elif isinstance(mapping_data, list):
    class_names = mapping_data

else:
    raise ValueError("Unsupported class mapping format.")


class_names = list(class_names)

class_to_idx = {
    name: idx
    for idx, name in enumerate(class_names)
}

NUM_CLASSES = len(class_names)

print(f"Number of classes: {NUM_CLASSES}")
print()


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


# ============================================================
# COLLECT ORIGINAL TRAINING IMAGES
# ============================================================

samples = []

print("=" * 70)
print("COLLECTING ORIGINAL TRAINING DATA")
print("=" * 70)

for class_name in class_names:

    class_dir = ORIGINAL_TRAIN_DIR / class_name

    if not class_dir.exists():
        print(f"WARNING: Missing original class: {class_name}")
        continue

    count = 0

    for path in class_dir.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        samples.append({
            "path": path,
            "class_name": class_name,
            "class_idx": class_to_idx[class_name],
            "source": "original"
        })

        count += 1

    if count > 0:
        print(f"{class_name:<25}: {count}")


original_count = sum(
    1 for s in samples
    if s["source"] == "original"
)

print()
print(f"Original images collected: {original_count}")


# ============================================================
# COLLECT REAL-WORLD TRAINING IMAGES
# ============================================================

print()
print("=" * 70)
print("COLLECTING REAL-WORLD TRAINING DATA")
print("=" * 70)

real_world_count = 0

for class_name in ["Neem", "Tulsi", "Mint", "Aloevera"]:

    class_dir = REAL_WORLD_TRAIN_DIR / class_name

    if not class_dir.exists():
        print(f"WARNING: Missing real-world class: {class_name}")
        continue

    if class_name not in class_to_idx:
        print(
            f"WARNING: {class_name} is not present in the "
            f"71-class mapping."
        )
        continue

    count = 0

    for path in class_dir.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        samples.append({
            "path": path,
            "class_name": class_name,
            "class_idx": class_to_idx[class_name],
            "source": "real_world"
        })

        count += 1
        real_world_count += 1

    print(f"{class_name:<25}: {count}")


print()
print(f"Real-world images collected: {real_world_count}")


# ============================================================
# FINAL DATASET SUMMARY
# ============================================================

print()
print("=" * 70)
print("COMBINED DATASET")
print("=" * 70)

print(f"Original images   : {original_count}")
print(f"Real-world images : {real_world_count}")
print(f"Total images      : {len(samples)}")

print()


# ============================================================
# CHECK REAL-WORLD CLASSES
# ============================================================

print("=" * 70)
print("REAL-WORLD CLASS CHECK")
print("=" * 70)

for class_name in ["Neem", "Tulsi", "Mint", "Aloevera"]:

    count = sum(
        1
        for s in samples
        if s["class_name"] == class_name
        and s["source"] == "real_world"
    )

    print(f"{class_name:<15}: {count}")


# ============================================================
# DATASET
# ============================================================

class CombinedSpeciesDataset(Dataset):

    def __init__(self, samples, transform=None):

        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        item = self.samples[index]

        image = Image.open(item["path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = item["class_idx"]

        return image, label


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.75, 1.0),
        ratio=(0.75, 1.33)
    ),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomVerticalFlip(p=0.15),

    transforms.RandomRotation(15),

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
# CREATE DATASET
# ============================================================

dataset = CombinedSpeciesDataset(
    samples=samples,
    transform=train_transform
)


# ============================================================
# WEIGHTED SAMPLING
# ============================================================

weights = []

for sample in samples:

    if sample["source"] == "real_world":
        weights.append(REAL_WORLD_WEIGHT)
    else:
        weights.append(1.0)


weights = torch.DoubleTensor(weights)

sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=len(samples),
    replacement=True
)


# ============================================================
# DATALOADER
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# LOAD EFFICIENTNET-B0
# ============================================================

print()
print("=" * 70)
print("LOADING V2 MODEL")
print("=" * 70)

model = models.efficientnet_b0(
    weights=None
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    NUM_CLASSES
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
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

print("V2 model loaded successfully.")

model = model.to(DEVICE)


# ============================================================
# FREEZE / UNFREEZE
# ============================================================

for param in model.features.parameters():
    param.requires_grad = False


# Unfreeze deeper EfficientNet blocks.
for param in model.features[5:].parameters():
    param.requires_grad = True


for param in model.classifier.parameters():
    param.requires_grad = True


# ============================================================
# PARAMETER GROUPS
# ============================================================

backbone_parameters = []
classifier_parameters = []

for name, param in model.named_parameters():

    if not param.requires_grad:
        continue

    if name.startswith("classifier"):
        classifier_parameters.append(param)

    else:
        backbone_parameters.append(param)


trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)


print()
print("=" * 70)
print("MODEL CONFIGURATION")
print("=" * 70)

print(f"Total parameters     : {total_parameters:,}")
print(f"Trainable parameters : {trainable_parameters:,}")
print(f"Backbone LR          : {BACKBONE_LR}")
print(f"Classifier LR        : {CLASSIFIER_LR}")
print(f"Real-world weight    : {REAL_WORLD_WEIGHT}")
print()


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

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
# SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ============================================================
# TRAINING
# ============================================================

best_loss = float("inf")
epochs_without_improvement = 0

history = []

print("=" * 70)
print("STARTING V3 TRAINING")
print("=" * 70)

for epoch in range(1, EPOCHS + 1):

    start_time = time.time()

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    scheduler.step(epoch_loss)

    elapsed = time.time() - start_time

    current_lr = optimizer.param_groups[0]["lr"]

    print()
    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    print(
        f"Train Loss : {epoch_loss:.4f}"
    )

    print(
        f"Train Acc  : {epoch_accuracy * 100:.2f}%"
    )

    print(
        f"Backbone LR: {current_lr:.8f}"
    )

    print(
        f"Time       : {elapsed:.1f} sec"
    )


    history.append({
        "epoch": epoch,
        "loss": epoch_loss,
        "accuracy": epoch_accuracy,
        "backbone_lr": optimizer.param_groups[0]["lr"],
        "classifier_lr": optimizer.param_groups[1]["lr"],
        "time_seconds": elapsed
    })


    # --------------------------------------------------------
    # SAVE BEST TRAINING CHECKPOINT
    # --------------------------------------------------------

    if epoch_loss < best_loss:

        best_loss = epoch_loss
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "num_classes": NUM_CLASSES,
                "class_names": class_names,
                "epoch": epoch,
                "train_loss": epoch_loss,
                "train_accuracy": epoch_accuracy
            },
            OUTPUT_MODEL
        )

        print()
        print("★ New best V3 model saved.")

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement: "
            f"{epochs_without_improvement}/{PATIENCE}"
        )

        if epochs_without_improvement >= PATIENCE:

            print()
            print("Early stopping.")
            break


# ============================================================
# SAVE CLASS MAPPING
# ============================================================

with open(
    OUTPUT_MAPPING,
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

summary = {
    "model": "EfficientNet-B0",
    "starting_model": str(MODEL_PATH),
    "output_model": str(OUTPUT_MODEL),
    "num_classes": NUM_CLASSES,
    "original_images": original_count,
    "real_world_images": real_world_count,
    "total_samples": len(samples),
    "real_world_weight": REAL_WORLD_WEIGHT,
    "batch_size": BATCH_SIZE,
    "epochs_requested": EPOCHS,
    "backbone_lr": BACKBONE_LR,
    "classifier_lr": CLASSIFIER_LR,
    "weight_decay": WEIGHT_DECAY,
    "device": str(DEVICE),
    "best_training_loss": best_loss,
    "history": history
}

OUTPUT_REPORT.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


print()
print("=" * 70)
print("V3 TRAINING COMPLETED")
print("=" * 70)

print(f"Best model : {OUTPUT_MODEL}")
print(f"Mapping    : {OUTPUT_MAPPING}")
print(f"Report     : {OUTPUT_REPORT}")
print("=" * 70)