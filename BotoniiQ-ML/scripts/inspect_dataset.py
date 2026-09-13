from pathlib import Path
from collections import Counter

# ============================================================
# BOTONIIQ - MEDICINAL LEAF DATASET INSPECTION
# ============================================================

DATASET_DIR = Path(
    r"D:\BotoniiQ\BotoniiQ-ML\dataset\raw\Medicinal Leaf dataset"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


def get_images(folder):
    """Return all supported images inside a folder."""
    return [
        file
        for file in folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]


def main():

    print("=" * 70)
    print("BOTONIIQ - MEDICINAL LEAF DATASET INSPECTION")
    print("=" * 70)

    print()
    print("Dataset location:")
    print(DATASET_DIR)

    # --------------------------------------------------------
    # Check dataset path
    # --------------------------------------------------------

    if not DATASET_DIR.exists():
        print()
        print("ERROR: Dataset folder does not exist.")
        print()
        print("Expected:")
        print(DATASET_DIR)
        return

    # --------------------------------------------------------
    # Find class folders
    # --------------------------------------------------------

    class_data = []

    for folder in sorted(DATASET_DIR.iterdir()):

        if not folder.is_dir():
            continue

        images = get_images(folder)

        if len(images) > 0:
            class_data.append(
                {
                    "name": folder.name,
                    "path": folder,
                    "count": len(images)
                }
            )

    # --------------------------------------------------------
    # Print classes
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CLASSES FOUND")
    print("=" * 70)

    for index, item in enumerate(class_data, start=1):

        print(
            f"{index:3}. "
            f"{item['name']:<35} "
            f"{item['count']:>5} images"
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_classes = len(class_data)
    total_images = sum(item["count"] for item in class_data)

    print()
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(f"Total classes : {total_classes}")
    print(f"Total images  : {total_images}")

    if class_data:

        counts = [item["count"] for item in class_data]

        print(f"Minimum images/class : {min(counts)}")
        print(f"Maximum images/class : {max(counts)}")
        print(
            f"Average images/class : "
            f"{sum(counts) / len(counts):.2f}"
        )

    # --------------------------------------------------------
    # Smallest classes
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CLASSES WITH FEWEST IMAGES")
    print("=" * 70)

    for item in sorted(
        class_data,
        key=lambda x: x["count"]
    )[:10]:

        print(
            f"{item['name']:<35} "
            f"{item['count']:>5}"
        )

    # --------------------------------------------------------
    # Largest classes
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CLASSES WITH MOST IMAGES")
    print("=" * 70)

    for item in sorted(
        class_data,
        key=lambda x: x["count"],
        reverse=True
    )[:10]:

        print(
            f"{item['name']:<35} "
            f"{item['count']:>5}"
        )

    print()
    print("=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()