from pathlib import Path
from collections import defaultdict
from PIL import Image
import hashlib
import shutil
import csv
import random

# ============================================================
# BOTONIIQ
# SPECIES DATASET PREPARATION
#
# RAW DATA IS NEVER MODIFIED.
# ============================================================

RAW_DATASET = Path(
    r"D:\BotoniiQ\BotoniiQ-ML\dataset\raw\Medicinal Leaf dataset"
)

OUTPUT_DATASET = Path(
    r"D:\BotoniiQ\BotoniiQ-ML\dataset\species"
)

REPORT_DIR = Path(
    r"D:\BotoniiQ\BotoniiQ-ML\outputs\reports"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}

# ------------------------------------------------------------
# First version threshold
# ------------------------------------------------------------

MIN_IMAGES_PER_CLASS = 50

# ------------------------------------------------------------
# Dataset split
# ------------------------------------------------------------

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42


# ============================================================
# HASH
# ============================================================

def calculate_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# FIND IMAGES
# ============================================================

def get_images(folder):

    return sorted(
        [
            file
            for file in folder.rglob("*")
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda x: str(x).lower()
    )


# ============================================================
# CHECK IMAGE
# ============================================================

def is_valid_image(file_path):

    try:

        with Image.open(file_path) as image:

            image.verify()

        return True

    except Exception:

        return False


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("BOTONIIQ - SPECIES DATASET PREPARATION")
    print("=" * 75)

    print()
    print("RAW DATASET:")
    print(RAW_DATASET)

    print()
    print("OUTPUT DATASET:")
    print(OUTPUT_DATASET)

    # --------------------------------------------------------
    # Validate raw dataset
    # --------------------------------------------------------

    if not RAW_DATASET.exists():

        print()
        print("ERROR: Raw dataset does not exist.")
        return

    # --------------------------------------------------------
    # Create output folders
    # --------------------------------------------------------

    for split in ["train", "val", "test"]:

        (OUTPUT_DATASET / split).mkdir(
            parents=True,
            exist_ok=True
        )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Discover classes
    # --------------------------------------------------------

    class_folders = sorted(
        [
            folder
            for folder in RAW_DATASET.iterdir()
            if folder.is_dir()
        ],
        key=lambda x: x.name.lower()
    )

    print()
    print(f"Classes discovered: {len(class_folders)}")

    # --------------------------------------------------------
    # Collect valid classes
    # --------------------------------------------------------

    selected_classes = []

    excluded_classes = []

    class_images = {}

    for class_folder in class_folders:

        class_name = class_folder.name

        images = get_images(class_folder)

        print(
            f"{class_name:<35} "
            f"{len(images):>4} images"
        )

        if len(images) < MIN_IMAGES_PER_CLASS:

            excluded_classes.append(
                {
                    "class": class_name,
                    "count": len(images)
                }
            )

            continue

        selected_classes.append(class_name)

        class_images[class_name] = images

    # --------------------------------------------------------
    # Remove duplicate files globally
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("CHECKING DUPLICATES")
    print("=" * 75)

    hashes = defaultdict(list)

    for class_name in selected_classes:

        for image_path in class_images[class_name]:

            file_hash = calculate_hash(image_path)

            hashes[file_hash].append(
                (class_name, image_path)
            )

    duplicate_groups = {
        file_hash: items
        for file_hash, items in hashes.items()
        if len(items) > 1
    }

    print()
    print(
        f"Duplicate groups found: "
        f"{len(duplicate_groups)}"
    )

    # --------------------------------------------------------
    # Detect cross-class duplicates
    # --------------------------------------------------------

    cross_class_duplicates = []

    duplicate_files_to_skip = set()

    for file_hash, items in duplicate_groups.items():

        classes = set(
            item[0]
            for item in items
        )

        if len(classes) > 1:

            cross_class_duplicates.append(
                {
                    "hash": file_hash,
                    "items": items
                }
            )

            # Do not automatically choose a label.
            # Keep these out of the processed dataset.
            for class_name, image_path in items:

                duplicate_files_to_skip.add(
                    str(image_path)
                )

        else:

            # Same-class duplicates.
            #
            # Keep the first image.
            # Skip the remaining identical files.

            for index, (class_name, image_path) in enumerate(items):

                if index > 0:

                    duplicate_files_to_skip.add(
                        str(image_path)
                    )

    # --------------------------------------------------------
    # Report cross-class duplicates
    # --------------------------------------------------------

    cross_class_report = (
        REPORT_DIR /
        "cross_class_duplicates.csv"
    )

    with open(
        cross_class_report,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "hash",
                "class_name",
                "file"
            ]
        )

        for group in cross_class_duplicates:

            for class_name, image_path in group["items"]:

                writer.writerow(
                    [
                        group["hash"],
                        class_name,
                        str(image_path)
                    ]
                )

    # --------------------------------------------------------
    # Build usable image list
    # --------------------------------------------------------

    usable_images = {}

    for class_name in selected_classes:

        usable_images[class_name] = []

        for image_path in class_images[class_name]:

            if str(image_path) in duplicate_files_to_skip:

                continue

            if not is_valid_image(image_path):

                continue

            usable_images[class_name].append(
                image_path
            )

    # --------------------------------------------------------
    # Prepare split report
    # --------------------------------------------------------

    split_rows = []

    # --------------------------------------------------------
    # Copy images
    # --------------------------------------------------------

    total_train = 0
    total_val = 0
    total_test = 0

    for class_name in selected_classes:

        images = usable_images[class_name]

        # Deterministic ordering
        images = sorted(
            images,
            key=lambda x: str(x).lower()
        )

        # Shuffle deterministically so the same dataset
        # always produces the same train/val/test split.
        random.Random(
            RANDOM_SEED
        ).shuffle(images)

        total = len(images)

        train_count = int(
            total * TRAIN_RATIO
        )

        val_count = int(
            total * VAL_RATIO
        )

        test_count = (
            total
            - train_count
            - val_count
        )

        train_images = images[
            :train_count
        ]

        val_images = images[
            train_count:
            train_count + val_count
        ]

        test_images = images[
            train_count + val_count:
        ]

        split_data = [
            ("train", train_images),
            ("val", val_images),
            ("test", test_images)
        ]

        for split_name, split_images in split_data:

            destination_dir = (
                OUTPUT_DATASET
                / split_name
                / class_name
            )

            destination_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            for image_path in split_images:

                destination_path = (
                    destination_dir
                    / image_path.name
                )

                shutil.copy2(
                    image_path,
                    destination_path
                )

            if split_name == "train":

                total_train += len(split_images)

            elif split_name == "val":

                total_val += len(split_images)

            elif split_name == "test":

                total_test += len(split_images)

        split_rows.append(
            [
                class_name,
                total,
                train_count,
                val_count,
                test_count
            ]
        )

    # --------------------------------------------------------
    # Save split report
    # --------------------------------------------------------

    split_report = (
        REPORT_DIR /
        "species_split_report.csv"
    )

    with open(
        split_report,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "class_name",
                "usable_images",
                "train",
                "validation",
                "test"
            ]
        )

        writer.writerows(split_rows)

    # --------------------------------------------------------
    # Save excluded classes
    # --------------------------------------------------------

    excluded_report = (
        REPORT_DIR /
        "excluded_species_classes.csv"
    )

    with open(
        excluded_report,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "class_name",
                "image_count",
                "reason"
            ]
        )

        for item in excluded_classes:

            writer.writerow(
                [
                    item["class"],
                    item["count"],
                    "Below minimum threshold"
                ]
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("FINAL DATASET SUMMARY")
    print("=" * 75)

    print(
        f"Original classes       : "
        f"{len(class_folders)}"
    )

    print(
        f"Selected classes       : "
        f"{len(selected_classes)}"
    )

    print(
        f"Excluded classes       : "
        f"{len(excluded_classes)}"
    )

    print(
        f"Cross-class duplicates : "
        f"{len(cross_class_duplicates)}"
    )

    print()
    print(
        f"Training images        : "
        f"{total_train}"
    )

    print(
        f"Validation images      : "
        f"{total_val}"
    )

    print(
        f"Test images            : "
        f"{total_test}"
    )

    print()
    print("=" * 75)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 75)

    print()
    print("Processed dataset:")
    print(OUTPUT_DATASET)

    print()
    print("Reports:")
    print(split_report)
    print(excluded_report)
    print(cross_class_report)


if __name__ == "__main__":
    main()