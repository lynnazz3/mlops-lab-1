import os
import random
from pathlib import Path

from PIL import Image

RAW_DIR = Path("data/food11_raw")
PROCESSED_DIR = Path("data/food11_processed")
MINI_DIR = Path("data/food11_processed_mini")

SPLITS = ["training", "evaluation", "validation"]

CATEGORIES = {
    0: "Bread",
    1: "Dairy product",
    2: "Dessert",
    3: "Egg",
    4: "Fried food",
    5: "Meat",
    6: "Noodles-Pasta",
    7: "Rice",
    8: "Seafood",
    9: "Soup",
    10: "Vegetable-Fruit",
}

TARGET_SIZE = (128, 128)
MINI_MAX_PER_CATEGORY = 100


def get_category_index(filename: str) -> int:
    prefix = filename.split("_")[0]
    return int(prefix)


def process_split(split: str) -> None:
    src_split_dir = RAW_DIR / split
    if not src_split_dir.exists():
        print(f"Skipping {split}: folder not found at {src_split_dir}")
        return

    files_by_category = {idx: [] for idx in CATEGORIES}

    for filename in os.listdir(src_split_dir):
        filepath = src_split_dir / filename
        if not filepath.is_file():
            continue
        try:
            cat_idx = get_category_index(filename)
        except ValueError:
            continue
        if cat_idx not in CATEGORIES:
            continue
        files_by_category[cat_idx].append(filename)

    for cat_idx, filenames in files_by_category.items():
        cat_name = CATEGORIES[cat_idx]

        full_out_dir = PROCESSED_DIR / split / cat_name
        full_out_dir.mkdir(parents=True, exist_ok=True)

        mini_out_dir = MINI_DIR / split / cat_name
        mini_out_dir.mkdir(parents=True, exist_ok=True)

        random.shuffle(filenames)
        mini_filenames = set(filenames[:MINI_MAX_PER_CATEGORY])

        for filename in filenames:
            src_path = src_split_dir / filename
            try:
                with Image.open(src_path) as img:
                    img = img.convert("RGB")
                    img_resized = img.resize(TARGET_SIZE)

                    img_resized.save(full_out_dir / filename)

                    if filename in mini_filenames:
                        img_resized.save(mini_out_dir / filename)
            except Exception as e:
                print(f"Failed to process {src_path}: {e}")

        print(
            f"[{split}] {cat_name}: {len(filenames)} images processed "
            f"({len(mini_filenames)} kept in mini set)"
        )


def main() -> None:
    random.seed(42)
    for split in SPLITS:
        process_split(split)


if __name__ == "__main__":
    main()