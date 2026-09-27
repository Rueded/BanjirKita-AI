"""
BanjirKita AI - Dataset Sanity Check

Investigates why a MobileNetV2 classifier reaches ~99.9% val accuracy
after a single epoch on the flood/non_flood dataset -- this is NOT
normal behavior for a real-world binary image classification task and
needs to be understood before any accuracy number from this dataset
is reported anywhere in the submission.

Checks four things, in order of how likely they are to explain
"perfect from epoch 1, barely moves for 9 more epochs":
  1. Near-duplicate images leaking across the train/val split
  2. Trivial non-content shortcuts (image dimensions differ by class)
  3. Whether a dumb baseline (no real vision, just raw pixel stats)
     already achieves suspiciously high accuracy
  4. Whether a blur-based (Laplacian variance) shortcut exists, and
     whether train_flood_classifier_ex.py's resolution-debias
     mitigation actually closes it on YOUR real data
"""

import argparse
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image


def simple_dhash(img, hash_size=8):
    """Difference hash -- cheap, dependency-free near-duplicate detector."""
    img = img.convert("L").resize((hash_size + 1, hash_size))
    pixels = np.asarray(img, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return diff.flatten()


def check_duplicates(data_dir, hash_size=8, dup_threshold=5, sample_size=1500, seed=42, full=False):
    root = Path(data_dir)
    records = []
    for cls in ("flood", "non_flood"):
        cls_dir = root / cls
        if not cls_dir.is_dir():
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                records.append((p, cls))

    rng = np.random.default_rng(seed)
    total_found = len(records)
    if not full and total_found > sample_size:
        idx = rng.choice(total_found, size=sample_size, replace=False)
        records = [records[i] for i in idx]
        print(f"Sampling {sample_size} of {total_found} images for the duplicate "
              f"check (use --full to check everything -- much slower, O(n^2)).")

    hashes, kept = [], []
    for p, cls in records:
        try:
            hashes.append(simple_dhash(Image.open(p), hash_size))
            kept.append((p, cls))
        except Exception as e:
            print(f"  (skipped {p.name}: {e})")

    print(f"Hashed {len(hashes)} images.")
    if len(hashes) < 2:
        return []

    H = np.array(hashes)
    N = H.shape[0]

    near_dup_pairs = []
    for i in range(N):
        dists = np.sum(H != H[i], axis=1)
        for j in range(i + 1, N):
            if dists[j] <= dup_threshold:
                near_dup_pairs.append((kept[i][0].name, kept[i][1], kept[j][0].name, kept[j][1], int(dists[j])))

    print(f"\nFound {len(near_dup_pairs)} near-duplicate pairs (Hamming distance <= {dup_threshold}) in the sample.")
    cross_class = [p for p in near_dup_pairs if p[1] != p[3]]
    if cross_class:
        print(f"WARNING: {len(cross_class)} of those pairs have DIFFERENT labels "
              f"(near-identical image labeled flood in one place, non_flood in "
              f"another) -- this alone would explain a lot.")
    if near_dup_pairs:
        print("First 10 examples:")
        for a, ca, b, cb, d in near_dup_pairs[:10]:
            flag = "  <-- DIFFERENT LABELS" if ca != cb else ""
            print(f"  {a} ({ca})  <->  {b} ({cb})   distance={d}{flag}")
    return near_dup_pairs


def check_image_dimension_shortcut(data_dir, sample_per_class=500):
    """
    If flood/ and non_flood/ images have systematically different
    dimensions or file sizes, a model can 'cheat' using that alone,
    with zero real understanding of flood content.
    """
    root = Path(data_dir)
    print("\nImage dimension / file-size distribution by class:")
    for cls in ("flood", "non_flood"):
        cls_dir = root / cls
        if not cls_dir.is_dir():
            continue
        sizes, filesizes = [], []
        for p in list(cls_dir.iterdir())[:sample_per_class]:
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                try:
                    with Image.open(p) as img:
                        sizes.append(img.size)
                    filesizes.append(p.stat().st_size)
                except Exception:
                    pass
        if sizes:
            widths = [s[0] for s in sizes]
            heights = [s[1] for s in sizes]
            print(
                f"  {cls}: n={len(sizes)} | "
                f"width {np.mean(widths):.0f}±{np.std(widths):.0f} | "
                f"height {np.mean(heights):.0f}±{np.std(heights):.0f} | "
                f"filesize {np.mean(filesizes)/1024:.1f}±{np.std(filesizes)/1024:.1f} KB | "
                f"most common size={Counter(sizes).most_common(3)}"
            )


def dumb_baseline_check(data_dir, img_size=32, n_per_class=300, seed=42):
    """
    Trains a trivial linear classifier on nothing but flattened,
    heavily downsampled raw pixels -- no real vision model. If THIS
    alone gets suspiciously high accuracy, the classes are separable
    through superficial statistics, not genuine flood content.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    root = Path(data_dir)
    X, y = [], []
    rng = np.random.default_rng(seed)
    for cls, label in [("flood", 1), ("non_flood", 0)]:
        cls_dir = root / cls
        if not cls_dir.is_dir():
            continue
        paths = [p for p in cls_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        rng.shuffle(paths)
        for p in paths[:n_per_class]:
            try:
                img = Image.open(p).convert("RGB").resize((img_size, img_size))
                X.append(np.asarray(img, dtype=np.float32).flatten() / 255.0)
                y.append(label)
            except Exception:
                pass

    if len(set(y)) < 2:
        print("\n(dumb baseline check skipped -- couldn't load both classes)")
        return None

    X, y = np.array(X), np.array(y)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    acc = clf.score(X_val, y_val)

    print(
        f"\nDumb baseline (logistic regression on {img_size}x{img_size} raw "
        f"pixels, zero real vision): val accuracy = {acc:.4f}"
    )
    if acc > 0.90:
        print(
            "  --> HIGH. This strongly suggests the dataset has a superficial "
            "shortcut (color balance, brightness, resolution, source artifacts) "
            "that MobileNetV2 is exploiting -- not genuine flood understanding."
        )
    else:
        print("  --> Not suspiciously high on its own -- less evidence of an obvious pixel-level shortcut.")
    return acc


def dimension_only_baseline_check(data_dir, n_per_class=500, seed=42):
    """
    The most surgical dimension test: classify using ONLY
    width/height/aspect-ratio as features -- zero pixel content. If
    this alone achieves high accuracy, the two classes are trivially
    separable by image metadata.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    root = Path(data_dir)
    X, y = [], []
    rng = np.random.default_rng(seed)
    for cls, label in [("flood", 1), ("non_flood", 0)]:
        cls_dir = root / cls
        if not cls_dir.is_dir():
            continue
        paths = [p for p in cls_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        rng.shuffle(paths)
        for p in paths[:n_per_class]:
            try:
                with Image.open(p) as img:
                    w, h = img.size
                X.append([w, h, w / h])
                y.append(label)
            except Exception:
                pass

    if len(set(y)) < 2:
        print("\n(dimension-only baseline skipped -- couldn't load both classes)")
        return None

    X, y = np.array(X, dtype=np.float32), np.array(y)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    acc = clf.score(X_val, y_val)

    print(
        f"\nDimension-only baseline (width, height, aspect ratio -- ZERO pixel "
        f"content): val accuracy = {acc:.4f}"
    )
    if acc > 0.90:
        print(
            "  --> The two classes are trivially separable by image dimensions "
            "alone. A model doesn't need to understand flood content at all to "
            "score well here -- this is the strongest evidence of a dataset "
            "artifact, not a real classifier."
        )
    return acc


def blur_shortcut_baseline_check(data_dir, n_per_class=200, bottleneck=48, seed=42):
    """
    Directly tests whether train_flood_classifier_ex.py's
    _resolution_debias() mitigation is actually closing the
    dimension/blur shortcut ON YOUR REAL DATA -- not a synthetic proxy.

    Measures blur (Laplacian variance, a standard sharpness metric) for
    a sample of real images from both classes, BEFORE and AFTER applying
    the exact same debiasing transform used during training (fixed
    48px bottleneck resize + random blur + random JPEG re-compress).
    Then checks: can a trivial classifier still separate flood/non_flood
    using ONLY that single blur number, before vs. after the fix?

    Interpretation:
      - BEFORE accuracy near 1.0 confirms the shortcut exists in your
        real data.
      - AFTER accuracy close to 0.5 = shortcut essentially destroyed by
        the mitigation, on your real data.
      - AFTER accuracy still well above 0.5 (e.g. >0.65) = the
        mitigation is helping but not enough on its own; the more
        reliable fix is sourcing non_flood images that were NOT
        pre-resized to a suspiciously uniform size.
    """
    import io
    from PIL import ImageFilter
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    try:
        import cv2
    except ImportError:
        print("\n(blur_shortcut_baseline_check skipped -- needs opencv-python-headless:")
        print(" pip install opencv-python-headless)")
        return None

    def laplacian_variance(pil_img, resize_to=224):
        g = pil_img.convert("L").resize((resize_to, resize_to))
        arr = np.asarray(g)
        return cv2.Laplacian(arr, cv2.CV_64F).var()

    def resolution_debias(img, rng, bottleneck=bottleneck, target=224):
        # must match train_flood_classifier_ex.py's _resolution_debias()
        img = img.resize((bottleneck, bottleneck), Image.BILINEAR)
        img = img.resize((target, target), Image.BILINEAR)
        if rng.random() < 0.7:
            radius = float(rng.uniform(0.3, 1.2))
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        quality = int(rng.integers(40, 95))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    root = Path(data_dir)
    rng = np.random.default_rng(seed)
    paths_by_class = {}
    for cls in ("flood", "non_flood"):
        cls_dir = root / cls
        if not cls_dir.is_dir():
            print(f"\n(blur_shortcut_baseline_check skipped -- '{cls_dir}' not found)")
            return None
        paths = [p for p in cls_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        rng.shuffle(paths)
        paths_by_class[cls] = paths[:n_per_class]

    def run_pass(apply_fix):
        X, y = [], []
        local_rng = np.random.default_rng(seed + (1 if apply_fix else 0))
        for cls, label in [("flood", 1), ("non_flood", 0)]:
            for p in paths_by_class[cls]:
                try:
                    img = Image.open(p).convert("RGB")
                    if apply_fix:
                        img = resolution_debias(img, local_rng)
                    X.append([laplacian_variance(img)])
                    y.append(label)
                except Exception:
                    pass
        X, y = np.array(X, dtype=np.float32), np.array(y)
        if len(set(y)) < 2:
            return None
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y
        )
        clf = LogisticRegression().fit(X_train, y_train)
        return clf.score(X_val, y_val)

    print(f"\nRunning blur-shortcut check on {sum(len(v) for v in paths_by_class.values())} "
          f"real images (before vs. after the resolution-debias mitigation)...")
    acc_before = run_pass(apply_fix=False)
    acc_after = run_pass(apply_fix=True)

    print(f"\nBlur-only classifier accuracy, BEFORE mitigation: {acc_before:.3f}")
    print(f"Blur-only classifier accuracy, AFTER mitigation:  {acc_after:.3f}")
    print("(0.5 = shortcut destroyed / pure chance, 1.0 = shortcut fully intact)")

    if acc_after > 0.65:
        print(
            "\n  --> Mitigation helps but a real gap likely remains on this dataset. "
            "Do not expect near-perfect post-fix accuracy to mean the model has "
            "learned genuine flood content -- consider sourcing replacement "
            "non_flood images that aren't pre-resized to a fixed size, if time allows."
        )
    else:
        print("\n  --> Mitigation appears effective on your real data -- close to chance level.")

    return acc_before, acc_after


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity-check the flood/non_flood dataset for leakage or shortcuts.")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--full", action="store_true", help="Run duplicate check on the whole dataset (slow, O(n^2))")
    parser.add_argument("--skip_duplicates", action="store_true")
    parser.add_argument("--skip_blur_check", action="store_true",
                         help="Skip the blur-shortcut before/after check (needs opencv-python-headless)")
    args = parser.parse_args()

    print("=" * 70)
    print("BanjirKita AI - Dataset Sanity Check")
    print("=" * 70)

    if not args.skip_duplicates:
        check_duplicates(args.data_dir, full=args.full)
    else:
        print("(duplicate check skipped)")

    check_image_dimension_shortcut(args.data_dir)
    dimension_only_baseline_check(args.data_dir)
    dumb_baseline_check(args.data_dir)

    if not args.skip_blur_check:
        blur_shortcut_baseline_check(args.data_dir)
    else:
        print("\n(blur-shortcut before/after check skipped)")
