from pathlib import Path
from collections import Counter
import shutil

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

# ORIGINAL RAW DATASET
RAW_DIR = (
    PROJECT_DIR
    / "dataset"
    / "raw"
    / "Medicinal Plant Leaf Disease Dataset"
)

# NEW PROCESSED DATASET
OUTPUT_DIR = (
    PROJECT_DIR
    / "dataset"
    / "disease"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# VALIDATE CONFIGURATION
# ============================================================

if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 0.001:
    raise ValueError("Train + validation + test ratios must equal 1.0")


# ============================================================
# FIND IMAGES
# ============================================================

def find_images():

    images = []

    for path in RAW_DIR.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            images.append(path)

    return sorted(images)


# ============================================================
# GET CLASS NAME
# ============================================================

def get_class_name(image_path):

    return image_path.parent.name


# ============================================================
# COPY IMAGES
# ============================================================

def copy_images(images, split_name):

    copied = 0

    for image_path in images:

        class_name = get_class_name(image_path)

        destination_dir = (
            OUTPUT_DIR
            / split_name
            / class_name
        )

        destination_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        destination_path = (
            destination_dir
            / image_path.name
        )

        shutil.copy2(
            image_path,
            destination_path
        )

        copied += 1

    return copied


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BOTONIIQ - DISEASE DATASET SPLITTER")
    print("=" * 70)

    print()
    print("Raw dataset:")
    print(RAW_DIR)

    print()
    print("Output dataset:")
    print(OUTPUT_DIR)

    # --------------------------------------------------------
    # Check raw dataset
    # --------------------------------------------------------

    if not RAW_DIR.exists():

        print()
        print("[ERROR] Raw disease dataset does not exist.")
        print(RAW_DIR)
        return

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    print()
    print("Finding images...")

    all_images = find_images()

    print(
        f"Total images found: {len(all_images)}"
    )

    if len(all_images) == 0:

        print()
        print("[ERROR] No images found.")
        return

    # --------------------------------------------------------
    # Create labels
    # --------------------------------------------------------

    labels = [
        get_class_name(path)
        for path in all_images
    ]

    class_counts = Counter(labels)

    print()
    print(
        f"Classes found: {len(class_counts)}"
    )

    print()
    print("Class distribution:")

    for class_name in sorted(class_counts):

        print(
            f"{class_name:<40}"
            f"{class_counts[class_name]}"
        )

    # --------------------------------------------------------
    # First split:
    #
    # 70% TRAIN
    # 30% TEMPORARY
    # --------------------------------------------------------

    train_images, temp_images = train_test_split(
        all_images,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    # --------------------------------------------------------
    # Labels for temporary set
    # --------------------------------------------------------

    temp_labels = [
        get_class_name(path)
        for path in temp_images
    ]

    # --------------------------------------------------------
    # Second split:
    #
    # TEMPORARY 30%
    #
    # 50% → VAL = 15% total
    # 50% → TEST = 15% total
    # --------------------------------------------------------

    val_images, test_images = train_test_split(
        temp_images,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=temp_labels,
    )

    # --------------------------------------------------------
    # Summary before copying
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)

    print()
    print(
        f"Train : {len(train_images)} "
        f"({len(train_images) / len(all_images) * 100:.2f}%)"
    )

    print(
        f"Val   : {len(val_images)} "
        f"({len(val_images) / len(all_images) * 100:.2f}%)"
    )

    print(
        f"Test  : {len(test_images)} "
        f"({len(test_images) / len(all_images) * 100:.2f}%)"
    )

    print()
    print(
        f"Total : "
        f"{len(train_images) + len(val_images) + len(test_images)}"
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if (
        len(train_images)
        + len(val_images)
        + len(test_images)
        != len(all_images)
    ):

        raise RuntimeError(
            "Split count does not match original dataset."
        )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Copy images
    # --------------------------------------------------------

    print()
    print("Copying training images...")

    train_count = copy_images(
        train_images,
        "train"
    )

    print(
        f"Copied {train_count} training images."
    )

    print()
    print("Copying validation images...")

    val_count = copy_images(
        val_images,
        "val"
    )

    print(
        f"Copied {val_count} validation images."
    )

    print()
    print("Copying test images...")

    test_count = copy_images(
        test_images,
        "test"
    )

    print(
        f"Copied {test_count} test images."
    )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL VERIFICATION")
    print("=" * 70)

    for split_name in ["train", "val", "test"]:

        split_dir = (
            OUTPUT_DIR
            / split_name
        )

        count = sum(
            1
            for path in split_dir.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        )

        print(
            f"{split_name:<10}: {count}"
        )

    print()
    print("Output created at:")

    print(OUTPUT_DIR)

    print()
    print("=" * 70)
    print("DATASET SPLIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()