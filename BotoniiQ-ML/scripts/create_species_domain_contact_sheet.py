from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import random
import math

# ============================================================
# BOTONIIQ - SPECIES DOMAIN CONTACT SHEET
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPECIES_DIR = PROJECT_DIR / "dataset" / "species"
REAL_WORLD_DIR = PROJECT_DIR / "dataset" / "real_world_test"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "domain_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "Neem",
    "Tulsi",
    "Mint",
    "Aloevera",
]

IMAGES_PER_CLASS = 6

IMAGE_SIZE = (220, 220)


def get_images(folder):

    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG",
    ]

    files = []

    for ext in extensions:
        files.extend(folder.glob(ext))

    return files


def create_thumbnail(image_path):

    img = Image.open(image_path).convert("RGB")

    img.thumbnail(IMAGE_SIZE)

    canvas = Image.new(
        "RGB",
        IMAGE_SIZE,
        "white"
    )

    x = (IMAGE_SIZE[0] - img.width) // 2
    y = (IMAGE_SIZE[1] - img.height) // 2

    canvas.paste(img, (x, y))

    return canvas


def create_sheet(title, groups, output_path):

    rows = len(groups)
    cols = IMAGES_PER_CLASS

    label_height = 35

    sheet_width = cols * IMAGE_SIZE[0]
    sheet_height = rows * (IMAGE_SIZE[1] + label_height)

    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for row, (class_name, image_files) in enumerate(groups):

        random.shuffle(image_files)

        selected = image_files[:IMAGES_PER_CLASS]

        y = row * (IMAGE_SIZE[1] + label_height)

        draw.text(
            (5, y + 5),
            class_name,
            fill="black"
        )

        for col, image_path in enumerate(selected):

            try:

                thumbnail = create_thumbnail(image_path)

                x = col * IMAGE_SIZE[0]

                sheet.paste(
                    thumbnail,
                    (x, y + label_height)
                )

            except Exception as e:

                print(
                    f"Could not load: {image_path}"
                )

    sheet.save(output_path)

    print()
    print("Created:")
    print(output_path)


def main():

    random.seed(42)

    print("=" * 70)
    print("BOTONIIQ SPECIES DOMAIN CONTACT SHEET")
    print("=" * 70)

    # --------------------------------------------------------
    # TRAINING DATA
    # --------------------------------------------------------

    training_groups = []

    for class_name in CLASSES:

        folder = SPECIES_DIR / "train" / class_name

        images = get_images(folder)

        print(
            f"Training {class_name}: "
            f"{len(images)} images"
        )

        training_groups.append(
            (class_name, images)
        )

    training_output = (
        OUTPUT_DIR /
        "species_training_contact_sheet.jpg"
    )

    create_sheet(
        "Training Dataset",
        training_groups,
        training_output
    )

    # --------------------------------------------------------
    # REAL WORLD DATA
    # --------------------------------------------------------

    real_world_groups = []

    for class_name in CLASSES:

        folder = REAL_WORLD_DIR / class_name

        images = get_images(folder)

        print(
            f"Real-world {class_name}: "
            f"{len(images)} images"
        )

        real_world_groups.append(
            (class_name, images)
        )

    real_world_output = (
        OUTPUT_DIR /
        "species_real_world_contact_sheet.jpg"
    )

    create_sheet(
        "Real World Dataset",
        real_world_groups,
        real_world_output
    )

    print()
    print("=" * 70)
    print("CONTACT SHEETS CREATED")
    print("=" * 70)


if __name__ == "__main__":
    main()