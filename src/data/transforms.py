"""Image transforms for training and inference.

Uses torchvision transforms v2 with a strong augmentation strategy for
training and minimal-but-correct preprocessing for validation/inference.
"""

from torchvision import transforms

from src.config import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD


def get_train_transforms() -> transforms.Compose:
    """
    Aggressive augmentation for training:
    - Random horizontal flip (mirrors left/right phone usage)
    - Random rotation ±15 degrees (camera mounting variation)
    - Color jitter (interior lighting changes)
    - Random erasing (occlusion robustness)
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE[0] + 16, IMAGE_SIZE[1] + 16)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])


def get_val_transforms() -> transforms.Compose:
    """Deterministic preprocessing for validation, test, and inference."""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def get_inference_transforms() -> transforms.Compose:
    """Alias for val transforms — used in the inference pipeline."""
    return get_val_transforms()
