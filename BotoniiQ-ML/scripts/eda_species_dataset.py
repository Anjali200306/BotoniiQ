from pathlib import Path
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# BOTONIIQ - SPECIES DATASET EDA
# ============================================================

DATASET_DIR = Path(
    r"D:\BotoniiQ\BotoniiQ-ML\dataset\species"
)

OUTPUT_DIR = Path(
    r"D:\BotoniiQ\BotoniiQ-ML\outputs\plots"
)

SPLITS = [
    "train",
    "val",
    "test"
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


# ============================================================
# GET IMAGE FILES
# ============================================================

def get_images(folder):

    return [
        file
        for file in folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]


# ============================================================
# COUNT DATASET
# ============================================================

def count_split(split):

    split_dir = DATASET_DIR / split

    counts = {}

    for class_dir in sorted(
        [
            folder
            for folder in split_dir.iterdir()
            if folder.is_dir()
        ],
        key=lambda x: x.name.lower()
    ):

        counts[class_dir.name] = len(
            get_images(class_dir)
        )

    return counts


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 75)
    print("BOTONIIQ - SPECIES DATASET EDA")
    print("=" * 75)

    all_counts = {}

    # --------------------------------------------------------
    # Count train / val / test
    # --------------------------------------------------------

    for split in SPLITS:

        counts = count_split(split)

        all_counts[split] = counts

        print()
        print(
            f"{split.upper():<15}"
            f"Classes: {len(counts):>3}"
            f"   Images: {sum(counts.values()):>5}"
        )

    # ========================================================
    # CLASS DISTRIBUTION - TRAIN
    # ========================================================

    train_counts = all_counts["train"]

    class_names = list(train_counts.keys())
    image_counts = list(train_counts.values())

    plt.figure(
        figsize=(18, 10)
    )

    plt.bar(
        range(len(class_names)),
        image_counts
    )

    plt.xticks(
        range(len(class_names)),
        class_names,
        rotation=90
    )

    plt.xlabel("Plant Species")
    plt.ylabel("Number of Training Images")
    plt.title(
        "BotoniiQ Species Dataset - Training Class Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "training_class_distribution.png",
        dpi=150
    )

    plt.close()

    # ========================================================
    # TRAIN / VAL / TEST TOTALS
    # ========================================================

    split_names = []
    split_totals = []

    for split in SPLITS:

        split_names.append(
            split.capitalize()
        )

        split_totals.append(
            sum(all_counts[split].values())
        )

    plt.figure(
        figsize=(8, 6)
    )

    plt.bar(
        split_names,
        split_totals
    )

    plt.xlabel("Dataset Split")
    plt.ylabel("Number of Images")
    plt.title(
        "BotoniiQ Dataset Split Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "dataset_split_distribution.png",
        dpi=150
    )

    plt.close()

    # ========================================================
    # IMAGE DIMENSIONS
    # ========================================================

    width_counter = Counter()
    height_counter = Counter()

    sample_count = 0

    for split in SPLITS:

        split_dir = DATASET_DIR / split

        for image_path in get_images(split_dir):

            try:

                with Image.open(image_path) as image:

                    width, height = image.size

                    width_counter[width] += 1
                    height_counter[height] += 1

                    sample_count += 1

            except Exception:
                pass

    print()
    print("=" * 75)
    print("IMAGE DIMENSIONS")
    print("=" * 75)

    print(
        f"Images inspected: {sample_count}"
    )

    print()
    print("Most common widths:")

    for width, count in width_counter.most_common(10):

        print(
            f"{width:>5} px : {count}"
        )

    print()
    print("Most common heights:")

    for height, count in height_counter.most_common(10):

        print(
            f"{height:>5} px : {count}"
        )

    # ========================================================
    # SAMPLE IMAGE GRID
    # ========================================================

    # Pick first 12 classes
    selected_classes = class_names[:12]

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(14, 11)
    )

    for index, class_name in enumerate(
        selected_classes
    ):

        class_dir = (
            DATASET_DIR
            / "train"
            / class_name
        )

        images = get_images(class_dir)

        ax = axes.flat[index]

        if images:

            try:

                image = Image.open(
                    images[0]
                )

                ax.imshow(image)

            except Exception:

                ax.text(
                    0.5,
                    0.5,
                    "Unable to load",
                    ha="center",
                    va="center"
                )

        ax.set_title(
            class_name
        )

        ax.axis("off")

    plt.suptitle(
        "BotoniiQ - Sample Training Images",
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "sample_training_images.png",
        dpi=150
    )

    plt.close()

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 75)
    print("EDA COMPLETE")
    print("=" * 75)

    print()
    print("Plots created in:")

    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()