from pathlib import Path
from collections import defaultdict
from PIL import Image
import hashlib
import csv

# ============================================================
# BOTONIIQ
# DATASET QUALITY AUDIT
# ============================================================

DATASET_DIR = Path(
    r"D:\BotoniiQ\BotoniiQ-ML\dataset\raw\Medicinal Leaf dataset"
)

OUTPUT_DIR = Path(
    r"D:\BotoniiQ\BotoniiQ-ML\outputs\reports"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}

# Images smaller than this will be reported.
# We are NOT deleting them.
MIN_WIDTH = 100
MIN_HEIGHT = 100


# ============================================================
# HASH FUNCTION
# ============================================================

def calculate_hash(file_path):
    """
    Calculate SHA256 hash of an image file.

    Same hash usually means identical file contents.
    """

    sha256 = hashlib.sha256()

    try:
        with open(file_path, "rb") as file:

            while True:

                chunk = file.read(1024 * 1024)

                if not chunk:
                    break

                sha256.update(chunk)

        return sha256.hexdigest()

    except Exception:
        return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("BOTONIIQ - DATASET QUALITY AUDIT")
    print("=" * 75)

    print()
    print("Dataset:")
    print(DATASET_DIR)

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DATASET_DIR.exists():

        print()
        print("ERROR: Dataset folder not found.")
        print(DATASET_DIR)

        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Containers
    # --------------------------------------------------------

    class_counts = {}

    corrupted_images = []

    small_images = []

    duplicate_groups = defaultdict(list)

    total_images = 0

    valid_images = 0

    # --------------------------------------------------------
    # Find class folders
    # --------------------------------------------------------

    class_folders = sorted(
        [
            folder
            for folder in DATASET_DIR.iterdir()
            if folder.is_dir()
        ],
        key=lambda x: x.name.lower()
    )

    print()
    print(f"Class folders found: {len(class_folders)}")

    # --------------------------------------------------------
    # Process each class
    # --------------------------------------------------------

    for class_folder in class_folders:

        class_name = class_folder.name

        image_files = [
            file
            for file in class_folder.rglob("*")
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        class_counts[class_name] = len(image_files)

        print(
            f"\n[{class_name}] "
            f"{len(image_files)} images"
        )

        for image_path in image_files:

            total_images += 1

            # ------------------------------------------------
            # Try opening image
            # ------------------------------------------------

            try:

                with Image.open(image_path) as image:

                    # Force image loading
                    image.load()

                    width, height = image.size

                    # ------------------------------------------------
                    # Small image check
                    # ------------------------------------------------

                    if (
                        width < MIN_WIDTH
                        or height < MIN_HEIGHT
                    ):

                        small_images.append(
                            {
                                "class": class_name,
                                "file": str(image_path),
                                "width": width,
                                "height": height
                            }
                        )

                    valid_images += 1

            except Exception as error:

                corrupted_images.append(
                    {
                        "class": class_name,
                        "file": str(image_path),
                        "error": str(error)
                    }
                )

                continue

            # ------------------------------------------------
            # Duplicate detection
            # ------------------------------------------------

            file_hash = calculate_hash(image_path)

            if file_hash:

                duplicate_groups[file_hash].append(
                    str(image_path)
                )

    # ========================================================
    # DUPLICATES
    # ========================================================

    duplicate_groups = {
        file_hash: files
        for file_hash, files in duplicate_groups.items()
        if len(files) > 1
    }

    duplicate_files_count = sum(
        len(files)
        for files in duplicate_groups.values()
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 75)
    print("AUDIT SUMMARY")
    print("=" * 75)

    print(f"Classes found             : {len(class_counts)}")
    print(f"Total images              : {total_images}")
    print(f"Readable images           : {valid_images}")
    print(f"Corrupted images          : {len(corrupted_images)}")
    print(f"Very small images         : {len(small_images)}")
    print(f"Duplicate files           : {duplicate_files_count}")
    print(f"Duplicate groups          : {len(duplicate_groups)}")

    # ========================================================
    # SAVE CLASS COUNTS
    # ========================================================

    class_report = OUTPUT_DIR / "class_counts.csv"

    with open(
        class_report,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "class_name",
                "image_count"
            ]
        )

        for class_name, count in sorted(
            class_counts.items(),
            key=lambda x: x[0].lower()
        ):

            writer.writerow(
                [
                    class_name,
                    count
                ]
            )

    # ========================================================
    # SAVE CORRUPTED FILES
    # ========================================================

    corrupted_report = OUTPUT_DIR / "corrupted_images.csv"

    with open(
        corrupted_report,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "class_name",
                "file",
                "error"
            ]
        )

        for item in corrupted_images:

            writer.writerow(
                [
                    item["class"],
                    item["file"],
                    item["error"]
                ]
            )

    # ========================================================
    # SAVE SMALL IMAGES
    # ========================================================

    small_report = OUTPUT_DIR / "small_images.csv"

    with open(
        small_report,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "class_name",
                "file",
                "width",
                "height"
            ]
        )

        for item in small_images:

            writer.writerow(
                [
                    item["class"],
                    item["file"],
                    item["width"],
                    item["height"]
                ]
            )

    # ========================================================
    # SAVE DUPLICATES
    # ========================================================

    duplicate_report = OUTPUT_DIR / "duplicate_images.txt"

    with open(
        duplicate_report,
        "w",
        encoding="utf-8"
    ) as file:

        for index, (file_hash, files) in enumerate(
            duplicate_groups.items(),
            start=1
        ):

            file.write(
                f"DUPLICATE GROUP {index}\n"
            )

            file.write(
                f"SHA256: {file_hash}\n"
            )

            for image in files:

                file.write(
                    f"  {image}\n"
                )

            file.write("\n")

    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print("=" * 75)
    print("REPORTS CREATED")
    print("=" * 75)

    print()
    print(class_report)

    print(corrupted_report)

    print(small_report)

    print(duplicate_report)

    print()
    print("=" * 75)
    print("DATASET AUDIT COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()