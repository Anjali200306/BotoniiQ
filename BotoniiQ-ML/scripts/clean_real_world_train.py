from pathlib import Path
from PIL import Image
import hashlib
import shutil

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_DIR / "dataset" / "real_world_train"
REVIEW_DIR = PROJECT_DIR / "dataset" / "real_world_train_review"

CLASSES = ["Neem", "Tulsi", "Mint", "Aloevera"]

# Images with either width OR height below this value
MIN_DIMENSION = 224

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# ============================================================
# HELPERS
# ============================================================

def md5_hash(path):
    hash_md5 = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def get_dimensions(path):
    with Image.open(path) as img:
        return img.size


def move_to_review(path, reason):
    relative = path.relative_to(DATASET_DIR)
    destination = REVIEW_DIR / relative

    destination.parent.mkdir(parents=True, exist_ok=True)

    # Avoid accidental overwrite
    if destination.exists():
        stem = destination.stem
        suffix = destination.suffix
        destination = destination.parent / f"{stem}_review{suffix}"

    shutil.move(str(path), str(destination))

    print(f"[MOVED - {reason}]")
    print(f"  From: {path}")
    print(f"  To  : {destination}")
    print()


# ============================================================
# START
# ============================================================

print("=" * 70)
print("BOTONIIQ REAL-WORLD TRAIN DATASET CLEANUP")
print("=" * 70)

print()
print(f"Dataset : {DATASET_DIR}")
print(f"Review  : {REVIEW_DIR}")
print()

if not DATASET_DIR.exists():
    print("ERROR: Dataset directory does not exist.")
    raise SystemExit(1)

REVIEW_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# COLLECT IMAGES
# ============================================================

images = []

for class_name in CLASSES:

    class_dir = DATASET_DIR / class_name

    if not class_dir.exists():
        print(f"WARNING: Missing class folder: {class_name}")
        continue

    for path in sorted(class_dir.iterdir()):

        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        try:
            width, height = get_dimensions(path)

            images.append({
                "path": path,
                "class": class_name,
                "width": width,
                "height": height,
                "pixels": width * height,
                "hash": md5_hash(path)
            })

        except Exception as e:
            print(f"[UNREADABLE] {path}")
            print(f"Reason: {e}")
            print()


print(f"Images scanned: {len(images)}")
print()


# ============================================================
# STEP 1 — FIND DUPLICATES
# ============================================================

print("=" * 70)
print("DUPLICATE CLEANUP")
print("=" * 70)

hash_groups = {}

for item in images:
    hash_groups.setdefault(item["hash"], []).append(item)

duplicate_groups = [
    group for group in hash_groups.values()
    if len(group) > 1
]

print(f"Duplicate groups found: {len(duplicate_groups)}")
print()

duplicate_moved = 0

for index, group in enumerate(duplicate_groups, start=1):

    # Keep highest-resolution copy
    group_sorted = sorted(
        group,
        key=lambda x: x["pixels"],
        reverse=True
    )

    keep = group_sorted[0]
    duplicates = group_sorted[1:]

    print(f"Duplicate group {index}")
    print("-" * 50)

    print(
        f"KEEP: {keep['path'].name} "
        f"({keep['width']} x {keep['height']})"
    )

    for item in duplicates:

        print(
            f"MOVE: {item['path'].name} "
            f"({item['width']} x {item['height']})"
        )

        move_to_review(
            item["path"],
            "DUPLICATE"
        )

        duplicate_moved += 1


# ============================================================
# STEP 2 — FIND VERY SMALL IMAGES
# ============================================================

print("=" * 70)
print("VERY SMALL IMAGE CLEANUP")
print("=" * 70)

small_images = []

for item in images:

    path = item["path"]

    # Skip files already moved as duplicates
    if not path.exists():
        continue

    if (
        item["width"] < MIN_DIMENSION
        or item["height"] < MIN_DIMENSION
    ):
        small_images.append(item)


print(
    f"Images below {MIN_DIMENSION}px in either dimension: "
    f"{len(small_images)}"
)

print()

small_moved = 0

for item in small_images:

    print(
        f"MOVE: {item['class']}\\{item['path'].name} "
        f"({item['width']} x {item['height']})"
    )

    move_to_review(
        item["path"],
        "VERY SMALL"
    )

    small_moved += 1


# ============================================================
# FINAL COUNTS
# ============================================================

print("=" * 70)
print("FINAL DATASET COUNTS")
print("=" * 70)

total_remaining = 0

for class_name in CLASSES:

    class_dir = DATASET_DIR / class_name

    count = 0

    if class_dir.exists():
        count = sum(
            1
            for p in class_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    total_remaining += count

    print(f"{class_name:<12}: {count}")

print("-" * 70)
print(f"{'TOTAL':<12}: {total_remaining}")

print()
print("=" * 70)
print("CLEANUP SUMMARY")
print("=" * 70)

print(f"Duplicate copies moved : {duplicate_moved}")
print(f"Small images moved     : {small_moved}")
print(f"Images remaining       : {total_remaining}")
print()
print(f"Review folder:")
print(REVIEW_DIR)

print()
print("STATUS: CLEANUP COMPLETED")
print("=" * 70)