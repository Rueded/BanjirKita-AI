"""
BanjirKita AI - Deduplicate the flood class

Removes near-duplicate images (Hamming distance <= threshold on a
difference hash) from data/flood, keeping only one copy of each. This
does NOT fix the dimension/resolution shortcut between flood and
non_flood -- that's a separate, harder problem (see the sanity check
script's dimension_only_baseline_check). This only addresses the
duplicate-image leakage issue.

Run this BEFORE re-training, on a COPY of your data if you want to
keep the originals:
    python dedupe_flood_class.py --data_dir ./data --apply

Without --apply, it only reports what WOULD be deleted (dry run).
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def simple_dhash(img, hash_size=8):
    img = img.convert("L").resize((hash_size + 1, hash_size))
    pixels = np.asarray(img, dtype=np.int16)
    return (pixels[:, 1:] > pixels[:, :-1]).flatten()


def dedupe_class(cls_dir, threshold=5, apply=False, backup_dir=None):
    cls_dir = Path(cls_dir)
    paths = sorted(
        p for p in cls_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    print(f"Scanning {len(paths)} images in {cls_dir}...")

    hashes = []
    kept_paths = []
    for p in paths:
        try:
            hashes.append(simple_dhash(Image.open(p), 8))
            kept_paths.append(p)
        except Exception as e:
            print(f"  (skipped unreadable {p.name}: {e})")

    H = np.array(hashes)
    N = H.shape[0]
    to_remove = set()

    for i in range(N):
        if i in to_remove:
            continue
        dists = np.sum(H != H[i], axis=1)
        for j in range(i + 1, N):
            if j in to_remove:
                continue
            if dists[j] <= threshold:
                to_remove.add(j)  # keep the first occurrence, drop later duplicates

    print(f"Identified {len(to_remove)} duplicate images to remove "
          f"(keeping {N - len(to_remove)} unique images).")

    if not apply:
        print("\nDRY RUN -- nothing deleted. Re-run with --apply to actually remove them.")
        print("Examples of what would be removed:")
        for idx in list(to_remove)[:10]:
            print(f"  {kept_paths[idx].name}")
        return N, len(to_remove)

    if backup_dir:
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)

    removed = 0
    for idx in to_remove:
        p = kept_paths[idx]
        if backup_dir:
            shutil.move(str(p), str(backup_dir / p.name))
        else:
            p.unlink()
        removed += 1

    print(f"Removed {removed} duplicate images"
          f"{' (moved to ' + str(backup_dir) + ')' if backup_dir else ' (deleted)'}.")
    return N, removed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate the flood class to reduce train/val leakage.")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--threshold", type=int, default=5, help="Hamming distance threshold for 'duplicate'")
    parser.add_argument("--apply", action="store_true", help="Actually remove duplicates (default: dry run only)")
    parser.add_argument("--backup_dir", default=None,
                         help="If set, move duplicates here instead of deleting them outright")
    parser.add_argument("--class_name", default="flood",
                         help="Which class folder to dedupe (default: flood, since that's where "
                              "the duplicates were found)")
    args = parser.parse_args()

    cls_dir = Path(args.data_dir) / args.class_name
    if not cls_dir.is_dir():
        raise SystemExit(f"'{cls_dir}' not found.")

    dedupe_class(cls_dir, threshold=args.threshold, apply=args.apply, backup_dir=args.backup_dir)
