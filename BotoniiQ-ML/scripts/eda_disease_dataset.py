from pathlib import Path
from collections import Counter
import random

import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# BOTONIIQ - DISEASE DATASET EDA
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_DIR / "dataset" / "disease"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ============================================================
# 1. BASIC INFORMATION
# ============================================================

print("=" * 70)
print("BOTONIIQ - DISEASE DATASET EDA")
print("=" * 70)

print()
print("Dataset:")
print(DATASET_DIR)

print()
print("Output:")
print(OUTPUT_DIR)


# ============================================================
# 2. FIND CLASSES
# ============================================================

classes = sorted(
    [
        folder.name
        for folder in (DATASET_DIR / "train").iterdir()
        if folder.is_dir()
    ],
    key=str.lower
)

print()
print("Classes:", len(classes))

for index, class_name in enumerate(classes):
    print(f"{index:2d}  {class_name}")


# ============================================================
# 3. COUNT IMAGES
# ============================================================

split_counts = {}

for split in ["train", "val", "test"]:

    split_dir = DATASET_DIR / split

    split_counts[split] = {}

    for class_name in classes:

        class_dir = split_dir / class_name

        count = sum(
            1
            for image in class_dir.iterdir()
            if image.is_file()
            and image.suffix.lower() in IMAGE_EXTENSIONS
        )

        split_counts[split][class_name] = count


# ============================================================
# 4. PRINT DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

for class_name in classes:

    train_count = split_counts["train"][class_name]
    val_count = split_counts["val"][class_name]
    test_count = split_counts["test"][class_name]

    total = train_count + val_count + test_count

    print(
        f"{class_name:<35}"
        f"Train: {train_count:<4}"
        f"Val: {val_count:<4}"
        f"Test: {test_count:<4}"
        f"Total: {total}"
    )


# ============================================================
# 5. PLOT CLASS DISTRIBUTION
# ============================================================

total_counts = []

for class_name in classes:

    total = (
        split_counts["train"][class_name]
        + split_counts["val"][class_name]
        + split_counts["test"][class_name]
    )

    total_counts.append(total)


plt.figure(figsize=(14, 7))

plt.bar(classes, total_counts)

plt.title("Disease Dataset - Images per Class")
plt.xlabel("Disease Class")
plt.ylabel("Number of Images")

plt.xticks(
    rotation=70,
    ha="right"
)

plt.tight_layout()

class_distribution_path = (
    OUTPUT_DIR / "disease_class_distribution.png"
)

plt.savefig(
    class_distribution_path,
    dpi=200
)

plt.close()

print()
print("Created:")
print(class_distribution_path)


# ============================================================
# 6. TRAIN / VAL / TEST DISTRIBUTION
# ============================================================

train_total = sum(split_counts["train"].values())
val_total = sum(split_counts["val"].values())
test_total = sum(split_counts["test"].values())

split_names = [
    "Train",
    "Validation",
    "Test"
]

split_values = [
    train_total,
    val_total,
    test_total
]

plt.figure(figsize=(8, 6))

plt.bar(
    split_names,
    split_values
)

plt.title("Disease Dataset - Train / Validation / Test")
plt.xlabel("Dataset Split")
plt.ylabel("Number of Images")

plt.tight_layout()

split_distribution_path = (
    OUTPUT_DIR / "disease_split_distribution.png"
)

plt.savefig(
    split_distribution_path,
    dpi=200
)

plt.close()

print("Created:")
print(split_distribution_path)


# ============================================================
# 7. SAMPLE IMAGES
# ============================================================

print()
print("=" * 70)
print("CREATING SAMPLE IMAGE GRID")
print("=" * 70)

sample_images = []

train_dir = DATASET_DIR / "train"

for class_name in classes:

    class_dir = train_dir / class_name

    images = [
        image
        for image in class_dir.iterdir()
        if image.is_file()
        and image.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if images:
        sample_images.append(
            (
                class_name,
                random.choice(images)
            )
        )


# ------------------------------------------------------------
# Create one image per class
# ------------------------------------------------------------

fig, axes = plt.subplots(
    3,
    4,
    figsize=(16, 12)
)

axes = axes.flatten()

for index, (class_name, image_path) in enumerate(sample_images):

    try:

        image = Image.open(image_path).convert("RGB")

        axes[index].imshow(image)

        axes[index].set_title(
            class_name,
            fontsize=9
        )

        axes[index].axis("off")

    except Exception as error:

        print(
            f"Could not load {image_path}: {error}"
        )

        axes[index].axis("off")


plt.tight_layout()

sample_images_path = (
    OUTPUT_DIR / "disease_sample_images.png"
)

plt.savefig(
    sample_images_path,
    dpi=200
)

plt.close()

print()
print("Created:")
print(sample_images_path)


# ============================================================
# 8. IMAGE DIMENSION ANALYSIS
# ============================================================

print()
print("=" * 70)
print("IMAGE DIMENSION ANALYSIS")
print("=" * 70)

dimension_counter = Counter()

for split in ["train", "val", "test"]:

    split_dir = DATASET_DIR / split

    for image_path in split_dir.rglob("*"):

        if (
            image_path.is_file()
            and image_path.suffix.lower() in IMAGE_EXTENSIONS
        ):

            try:

                with Image.open(image_path) as image:

                    dimension_counter[
                        image.size
                    ] += 1

            except Exception:
                pass


for dimension, count in dimension_counter.most_common():

    print(
        f"{dimension[0]} x {dimension[1]} : {count}"
    )


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("EDA COMPLETE")
print("=" * 70)

print()
print(f"Classes       : {len(classes)}")
print(f"Train images  : {train_total}")
print(f"Val images    : {val_total}")
print(f"Test images   : {test_total}")
print(f"Total images  : {train_total + val_total + test_total}")

print()
print("Generated plots:")

print(
    OUTPUT_DIR / "disease_class_distribution.png"
)

print(
    OUTPUT_DIR / "disease_split_distribution.png"
)

print(
    OUTPUT_DIR / "disease_sample_images.png"
)

print()
print("=" * 70)