import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

from src.food11.data import CLASSES

SPLITS = ["training", "validation", "evaluation"]


def validate_dataset(dataset_dir: Path, min_per_class: int = 50, tau: float = 0.05):
    results = []
    counts = {}
    hashes = {}
    duplicate_count = 0

    # Check 1: Exactly 11 class folders in each split
    for split in SPLITS:
        split_dir = dataset_dir / split

        if not split_dir.is_dir():
            results.append((f"{split}: folder exists", False))
            continue

        found_classes = {p.name for p in split_dir.iterdir() if p.is_dir()}

        valid_structure = found_classes == set(CLASSES)
        results.append((f"{split}: 11 class folders", valid_structure))

        counts[split] = Counter()

        # Check images and count samples
        for category in CLASSES:
            category_dir = split_dir / category

            if not category_dir.is_dir():
                continue

            for path in category_dir.iterdir():
                if not path.is_file():
                    continue

                counts[split][category] += 1

                # Check 2: Image opens, is RGB and 128x128
                try:
                    with Image.open(path) as img:
                        img.load()

                        valid_image = img.mode == "RGB" and img.size == (128, 128)

                except (OSError, ValueError):
                    valid_image = False

                results.append((f"Image: {path}", valid_image))

                # Check 4: Detect duplicates across splits
                file_hash = hashlib.md5(path.read_bytes()).hexdigest()

                if file_hash in hashes and hashes[file_hash] != split:
                    duplicate_count += 1
                else:
                    hashes[file_hash] = split

        # Check 3: Minimum image count
        for category in CLASSES:
            count = counts[split][category]

            results.append(
                (
                    f"{split}/{category}: >= {min_per_class} images",
                    count >= min_per_class,
                )
            )

    results.append(
        (
            "No duplicate images across splits",
            duplicate_count == 0,
        )
    )

    # Check 5: Compare training and validation distributions
    if "training" in counts and "validation" in counts:
        train_total = sum(counts["training"].values())
        val_total = sum(counts["validation"].values())

        if train_total > 0 and val_total > 0:
            for category in CLASSES:
                train_share = counts["training"][category] / train_total
                val_share = counts["validation"][category] / val_total

                results.append(
                    (
                        f"Distribution: {category}",
                        abs(train_share - val_share) <= tau,
                    )
                )
        else:
            results.append(("Distribution comparison", False))
    else:
        results.append(("Distribution comparison", False))

    # Print report
    failures = 0

    for description, passed in results:
        if passed:
            print(f"PASS: {description}")
        else:
            print(f"FAIL: {description}")
            failures += 1

    print(
        f"\nValidation finished: {len(results) - failures} passed, {failures} failed."
    )

    return failures == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.05)

    args = parser.parse_args()

    valid = validate_dataset(
        args.dataset_dir,
        min_per_class=args.min_per_class,
        tau=args.tau,
    )

    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()
