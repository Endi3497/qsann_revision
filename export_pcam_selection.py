#!/usr/bin/env python3
"""
Copy the exact PCam .npy files selected by the QSANN_revision pipeline.

It uses build_standard_loaders(dataset_choice='pcam') so that labels, sampling,
and splits match run.sh/main.py when the same seed and ratios are provided.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Sequence

from data_loader import build_standard_loaders, PCamDataset


def copy_subset(subset, split: str, dest_root: Path) -> None:
    base: PCamDataset = subset.dataset  # PCamDataset with .items storing (path, label)
    split_dir = dest_root / split
    for cname in ("class0", "class1"):
        (split_dir / cname).mkdir(parents=True, exist_ok=True)
    for idx in subset.indices:
        path, label = base.items[idx]
        out_path = split_dir / f"class{label}" / path.name
        shutil.copy2(path, out_path)


def main(args: argparse.Namespace) -> None:
    (train_loader, train_set), (val_loader, val_set), (test_loader, test_set) = build_standard_loaders(
        dataset_choice="pcam",
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        dataset_labels=[0, 1],
        samples_per_label=args.samples_per_label,
        medmnist_subset=None,
        balance_sampler=False,
        pcam_root=args.pcam_root,
    )

    dest_root = Path(args.dest)
    copy_subset(train_set, "train", dest_root)
    if len(val_set) > 0:
        copy_subset(val_set, "val", dest_root)
    copy_subset(test_set, "test", dest_root)
    print(f"Copied PCam selection to {dest_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export selected PCam .npy files used by QSANN_revision splits.")
    parser.add_argument("--pcam-root", type=Path, required=True, help="Root with class0/class1 .npy files.")
    parser.add_argument("--dest", type=Path, required=True, help="Destination root for copied files.")
    parser.add_argument("--samples-per-label", type=int, default=500, help="Per-class cap (match run settings).")
    parser.add_argument("--seed", type=int, default=42, help="Seed (match run settings).")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=28, help="Resize param (does not affect selection).")
    parser.add_argument("--batch-size", type=int, default=64, help="Loader batch size (does not affect selection).")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    main(args)
