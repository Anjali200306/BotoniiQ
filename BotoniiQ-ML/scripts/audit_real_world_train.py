from pathlib import Path
from PIL import Image
import hashlib
from collections import defaultdict

# ============================================================
# BOTONIIQ - REAL-WORLD TRAIN DATASET AUDIT
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_DIR / "dataset" / "real_world_train"

CLASSES = [
    "Neem",
    "Tulsi",
    "Mint",
    "Aloevera",
]

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def get_image_files(folder):
    return [
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in VALID_EXTENSIONS
    ]


def calculate_md5(file_path):
    md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(chunk)

    return md5.hexdigest()


def main():

    print("=" * 70)
    print("BOTONIIQ REAL-WORLD TRAIN DATASET AUDIT")
    print("=" * 70)

    print()
    print("Dataset:")
    print(DATASET_DIR)

    if not DATASET_DIR.exists():
        print()
        print("ERROR: Dataset folder does not exist.")
        return

    # --------------------------------------------------------
    # IMAGE COUNTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("IMAGE COUNTS")
    print("=" * 70)

    all_images = []

    for class_name in CLASSES:

        class_dir = DATASET_DIR / class_name

        if not class_dir.exists():
            print(f"{class_name:10} : DIRECTORY NOT FOUND")
            continue

        images = get_image_files(class_dir)

        all_images.extend(images)

        print(f"{class_name:10} : {len(images)}")

    print("-" * 70)
    print(f"{'TOTAL':10} : {len(all_images)}")

    # --------------------------------------------------------
    # IMAGE QUALITY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("IMAGE QUALITY")
    print("=" * 70)

    readable = 0
    unreadable = []
    very_small = []

    dimensions = defaultdict(int)

    for image_path in all_images:

        try:

            with Image.open(image_path) as img:

                img.verify()

            with Image.open(image_path) as img:

                width, height = img.size

            readable += 1

            dimensions[(width, height)] += 1

            if width < 224 or height < 224:
                very_small.append(
                    (image_path, width, height)
                )

        except Exception as e:

            unreadable.append(
                (image_path, str(e))
            )

    print()
    print(f"Total images       : {len(all_images)}")
    print(f"Readable images    : {readable}")
    print(f"Unreadable images  : {len(unreadable)}")
    print(f"Very small images  : {len(very_small)}")

    # --------------------------------------------------------
    # UNREADABLE
    # --------------------------------------------------------

    if unreadable:

        print()
        print("UNREADABLE IMAGES")
        print("-" * 70)

        for path, error in unreadable:
            print(path)
            print(f"  Error: {error}")

    # --------------------------------------------------------
    # VERY SMALL
    # --------------------------------------------------------

    if very_small:

        print()
        print("VERY SMALL IMAGES")
        print("-" * 70)

        for path, width, height in very_small:

            print(
                f"{path.name} : "
                f"{width} x {height}"
            )

    # --------------------------------------------------------
    # DIMENSIONS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("IMAGE DIMENSIONS")
    print("=" * 70)

    for (width, height), count in sorted(
        dimensions.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        print(
            f"{width} x {height} : "
            f"{count}"
        )

    # --------------------------------------------------------
    # EXACT DUPLICATES
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DUPLICATE CHECK")
    print("=" * 70)

    hashes = defaultdict(list)

    for image_path in all_images:

        try:

            file_hash = calculate_md5(image_path)

            hashes[file_hash].append(image_path)

        except Exception:
            pass

    duplicate_groups = [
        paths
        for paths in hashes.values()
        if len(paths) > 1
    ]

    if duplicate_groups:

        print()
        print(
            f"Duplicate groups found: "
            f"{len(duplicate_groups)}"
        )

        for group in duplicate_groups:

            print()
            print("Duplicate group:")

            for path in group:
                print(f"  {path}")

    else:

        print()
        print("No exact duplicate images found.")

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    print()
    print(f"Total images      : {len(all_images)}")
    print(f"Readable          : {readable}")
    print(f"Unreadable        : {len(unreadable)}")
    print(f"Very small        : {len(very_small)}")
    print(f"Duplicate groups  : {len(duplicate_groups)}")

    print()

    if (
        len(all_images) > 0
        and len(unreadable) == 0
        and len(duplicate_groups) == 0
    ):

        print("STATUS: DATASET BASIC QUALITY CHECK PASSED")

    else:

        print("STATUS: REVIEW REQUIRED")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()