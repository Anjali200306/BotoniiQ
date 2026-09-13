from pathlib import Path
from PIL import Image
from collections import Counter
import statistics

# ============================================================
# BOTONIIQ - SPECIES DATASET DOMAIN ANALYSIS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_DIR / "dataset" / "species"

TARGET_CLASSES = [
    "Neem",
    "Tulsi",
    "Mint",
    "Aloevera",
]

SPLITS = [
    "train",
    "val",
    "test",
]


def analyze_class(class_dir):
    image_files = []

    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        image_files.extend(class_dir.glob(ext))

    widths = []
    heights = []
    aspect_ratios = []
    file_sizes_kb = []

    readable = 0
    unreadable = 0

    for image_path in image_files:
        try:
            with Image.open(image_path) as img:
                width, height = img.size

                widths.append(width)
                heights.append(height)
                aspect_ratios.append(width / height)
                file_sizes_kb.append(image_path.stat().st_size / 1024)

                readable += 1

        except Exception:
            unreadable += 1

    return {
        "total": len(image_files),
        "readable": readable,
        "unreadable": unreadable,
        "width_min": min(widths) if widths else 0,
        "width_max": max(widths) if widths else 0,
        "height_min": min(heights) if heights else 0,
        "height_max": max(heights) if heights else 0,
        "width_mean": statistics.mean(widths) if widths else 0,
        "height_mean": statistics.mean(heights) if heights else 0,
        "aspect_mean": statistics.mean(aspect_ratios) if aspect_ratios else 0,
        "file_size_mean_kb": statistics.mean(file_sizes_kb)
        if file_sizes_kb else 0,
    }


def main():

    print("=" * 70)
    print("BOTONIIQ SPECIES DATASET DOMAIN ANALYSIS")
    print("=" * 70)

    print()
    print("Dataset:")
    print(DATASET_DIR)

    print()
    print("Target classes:")
    print(", ".join(TARGET_CLASSES))

    print()
    print("=" * 70)
    print("IMAGE COUNTS")
    print("=" * 70)

    for class_name in TARGET_CLASSES:

        print()
        print(f"\nCLASS: {class_name}")
        print("-" * 50)

        total_class = 0

        for split in SPLITS:

            class_dir = DATASET_DIR / split / class_name

            if not class_dir.exists():
                print(f"{split:5} : DIRECTORY NOT FOUND")
                continue

            count = 0

            for ext in ["*.jpg", "*.jpeg", "*.png",
                        "*.JPG", "*.JPEG", "*.PNG"]:
                count += len(list(class_dir.glob(ext)))

            total_class += count

            print(f"{split:5} : {count}")

        print(f"TOTAL : {total_class}")

    print()
    print("=" * 70)
    print("IMAGE PROPERTY ANALYSIS")
    print("=" * 70)

    for class_name in TARGET_CLASSES:

        print()
        print(f"\nCLASS: {class_name}")
        print("-" * 70)

        for split in SPLITS:

            class_dir = DATASET_DIR / split / class_name

            if not class_dir.exists():
                continue

            result = analyze_class(class_dir)

            print()
            print(f"[{split.upper()}]")

            print(f"Images              : {result['total']}")
            print(f"Readable            : {result['readable']}")
            print(f"Unreadable          : {result['unreadable']}")

            if result["total"] > 0:

                print(
                    f"Width range         : "
                    f"{result['width_min']} - {result['width_max']}"
                )

                print(
                    f"Height range        : "
                    f"{result['height_min']} - {result['height_max']}"
                )

                print(
                    f"Average width       : "
                    f"{result['width_mean']:.1f}"
                )

                print(
                    f"Average height      : "
                    f"{result['height_mean']:.1f}"
                )

                print(
                    f"Average aspect ratio: "
                    f"{result['aspect_mean']:.3f}"
                )

                print(
                    f"Average file size   : "
                    f"{result['file_size_mean_kb']:.1f} KB"
                )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()