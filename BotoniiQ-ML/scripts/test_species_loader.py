from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


DATASET_DIR = Path(
    r"D:\BotoniiQ\BotoniiQ-ML\dataset\species"
)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


train_dataset = datasets.ImageFolder(
    DATASET_DIR / "train",
    transform=transform
)


val_dataset = datasets.ImageFolder(
    DATASET_DIR / "val",
    transform=transform
)


test_dataset = datasets.ImageFolder(
    DATASET_DIR / "test",
    transform=transform
)


print("=" * 70)
print("BOTONIIQ - SPECIES DATASET LOADER TEST")
print("=" * 70)

print()

print("Train images :", len(train_dataset))
print("Val images   :", len(val_dataset))
print("Test images  :", len(test_dataset))

print()

print("Number of classes:", len(train_dataset.classes))

print()

print("First 10 classes:")

for index, class_name in enumerate(
    train_dataset.classes[:10]
):
    print(
        f"{index:>3} : {class_name}"
    )

print()

print("Class mapping example:")

for class_name, class_index in list(
    train_dataset.class_to_idx.items()
)[:10]:

    print(
        f"{class_name} -> {class_index}"
    )

print()

image, label = train_dataset[0]

print("Sample image tensor shape:")
print(image.shape)

print()

print("Sample label:")
print(label)

print()

print("=" * 70)
print("DATASET LOADER TEST COMPLETE")
print("=" * 70)