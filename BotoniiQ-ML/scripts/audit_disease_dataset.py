from pathlib import Path
from PIL import Image
import hashlib
from collections import Counter, defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

# IMPORTANT:
# This is ONLY the disease dataset.
DATASET_DIR = (
    PROJECT_DIR
    / "dataset"
    / "raw"
    / "Medicinal Plant Leaf Disease Dataset"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

# Images below this file size are reported as very small.
VERY_SMALL_BYTES = 10 * 1024  # 10 KB


# ============================================================
# HELPERS
# ============================================================

def calculate_md5(file_path):
    """Calculate MD5 hash of a file for duplicate detection."""
    md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            md5.update(chunk)

    return md5.hexdigest()


def find_images(dataset_dir):
    """Find all supported image files recursively."""
    return [
        path
        for path in dataset_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def get_class_name(image_path, dataset_dir):
    """
    Determine class using the folder structure.

    Example:
        Dataset/
            Neem/
                healthy/
                    image.jpg

    Class returned:
        healthy
    """

    relative_path = image_path.relative_to(dataset_dir)

    # The immediate parent folder is treated as the class.
    return relative_path.parent.name


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    print("=" * 70)
    print("BOTONIIQ - DISEASE DATASET AUDIT")
    print("=" * 70)

    print()
    print("Project directory:")
    print(PROJECT_DIR)

    print()
    print("Dataset directory:")
    print(DATASET_DIR)

    # --------------------------------------------------------
    # Check dataset directory
    # --------------------------------------------------------

    if not DATASET_DIR.exists():
        print()
        print("[ERROR] Disease dataset directory does not exist.")
        print()
        print("Expected:")
        print(DATASET_DIR)
        print()
        print("Please check the extracted dataset location.")
        return

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    print()
    print("Scanning images...")
    print()

    image_files = find_images(DATASET_DIR)

    total_images = len(image_files)

    print(f"Total images found: {total_images}")

    if total_images == 0:
        print()
        print("[WARNING] No supported image files were found.")
        return

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    class_counts = Counter()

    readable_images = []
    corrupted_images = []

    very_small_images = []

    duplicate_groups = defaultdict(list)

    image_hashes = {}

    # --------------------------------------------------------
    # Inspect every image
    # --------------------------------------------------------

    for index, image_path in enumerate(image_files, start=1):

        class_name = get_class_name(
            image_path,
            DATASET_DIR
        )

        class_counts[class_name] += 1

        # -----------------------------------------------
        # File size check
        # -----------------------------------------------

        try:
            file_size = image_path.stat().st_size

            if file_size < VERY_SMALL_BYTES:
                very_small_images.append(
                    (image_path, file_size)
                )

        except OSError:
            pass

        # -----------------------------------------------
        # Image readability check
        # -----------------------------------------------

        try:

            with Image.open(image_path) as img:

                # Verify image integrity
                img.verify()

            # Open again because verify() closes/invalidate
            # the image object.
            with Image.open(image_path) as img:

                width, height = img.size

            readable_images.append(
                (image_path, width, height)
            )

        except Exception as e:

            corrupted_images.append(
                (image_path, str(e))
            )

            continue

        # -----------------------------------------------
        # Duplicate detection
        # -----------------------------------------------

        try:

            file_hash = calculate_md5(image_path)

            image_hashes[image_path] = file_hash

            duplicate_groups[file_hash].append(
                image_path
            )

        except Exception:
            pass

        # -----------------------------------------------
        # Progress
        # -----------------------------------------------

        if index % 100 == 0 or index == total_images:
            print(
                f"Processed {index}/{total_images}",
                end="\r"
            )

    print()
    print()

    # ========================================================
    # RESULTS
    # ========================================================

    print("=" * 70)
    print("1. BASIC DATASET SUMMARY")
    print("=" * 70)

    print()
    print(f"Total images       : {total_images}")
    print(f"Readable images    : {len(readable_images)}")
    print(f"Corrupted images   : {len(corrupted_images)}")
    print(f"Very small images  : {len(very_small_images)}")
    print(f"Classes found      : {len(class_counts)}")

    # --------------------------------------------------------
    # Classes
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("2. CLASSES")
    print("=" * 70)

    for class_name in sorted(class_counts):
        print(
            f"{class_name:<40} {class_counts[class_name]}"
        )

    # --------------------------------------------------------
    # Corrupted images
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("3. CORRUPTED IMAGES")
    print("=" * 70)

    if not corrupted_images:

        print()
        print("None found.")

    else:

        print()

        for image_path, error in corrupted_images:

            print(image_path)
            print(f"  Error: {error}")

    # --------------------------------------------------------
    # Very small images
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("4. VERY SMALL IMAGES (< 10 KB)")
    print("=" * 70)

    if not very_small_images:

        print()
        print("None found.")

    else:

        print()

        for image_path, file_size in very_small_images:

            print(
                f"{image_path} "
                f"({file_size / 1024:.2f} KB)"
            )

    # --------------------------------------------------------
    # Duplicate files
    # --------------------------------------------------------

    duplicate_groups = {
        file_hash: paths
        for file_hash, paths in duplicate_groups.items()
        if len(paths) > 1
    }

    print()
    print("=" * 70)
    print("5. DUPLICATE FILES")
    print("=" * 70)

    if not duplicate_groups:

        print()
        print("No exact duplicate files found.")

    else:

        print()

        duplicate_group_number = 1

        for file_hash, paths in duplicate_groups.items():

            print(
                f"Duplicate group "
                f"{duplicate_group_number}:"
            )

            for path in paths:
                print(f"  {path}")

            print()

            duplicate_group_number += 1

    # --------------------------------------------------------
    # Cross-class duplicates
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("6. CROSS-CLASS DUPLICATES")
    print("=" * 70)

    cross_class_groups = {}

    for file_hash, paths in duplicate_groups.items():

        classes = {
            get_class_name(path, DATASET_DIR)
            for path in paths
        }

        if len(classes) > 1:

            cross_class_groups[file_hash] = (
                paths,
                classes
            )

    if not cross_class_groups:

        print()
        print("No cross-class duplicates found.")

    else:

        print()

        group_number = 1

        for file_hash, (paths, classes) in cross_class_groups.items():

            print(
                f"Cross-class duplicate group "
                f"{group_number}:"
            )

            print(
                "Classes:",
                ", ".join(sorted(classes))
            )

            for path in paths:
                print(f"  {path}")

            print()

            group_number += 1

    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("7. IMAGE DIMENSIONS")
    print("=" * 70)

    dimensions = Counter(
        (width, height)
        for _, width, height in readable_images
    )

    print()

    for (width, height), count in dimensions.most_common():

        print(
            f"{width} x {height:<6} : {count}"
        )

    # --------------------------------------------------------
    # Final assessment
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("8. AUDIT STATUS")
    print("=" * 70)

    print()

    if len(corrupted_images) == 0:

        print("[OK] No corrupted images detected.")

    else:

        print(
            f"[WARNING] "
            f"{len(corrupted_images)} corrupted images detected."
        )

    if len(cross_class_groups) == 0:

        print("[OK] No cross-class duplicates detected.")

    else:

        print(
            f"[WARNING] "
            f"{len(cross_class_groups)} "
            f"cross-class duplicate groups detected."
        )

    print()
    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()