#!/usr/bin/env python3
"""Extract low-level, model-free image descriptors for every image of a task.

Deliberately depends on **numpy + Pillow + OpenCV only** (SciPy is used for the
DCT if present, with an OpenCV fallback): no torch, no scikit-learn. It runs on
a plain CPU box and produces the evidence for the "are these classes separable
from trivial global statistics?" analysis and for the near-duplicate audit.

Output: ``results/features_<task>.csv``, one row per image.

Columns
-------
filename, path, label, class_dir, site, hive_proxy, session, disease_code
    Identity and proxy acquisition-unit fields (see ``common/data.py``).
native_w, native_h, native_area_px
    Native JPEG pixel dimensions and w*h (**pixels**, typically 1920x1080).
file_size_bytes
    On-disk JPEG size in **bytes**.
mean_intensity_256
    Mean grayscale intensity on the 256x256 resized image, **0-255 units**.
lab_L_native, lab_a_native, lab_b_native
    CIELAB channel means at **native** resolution. L* in **0-100**, a*/b* in
    **CIE units** (OpenCV's 8-bit LAB is rescaled: L*=L*100/255, a*=a-128,
    b*=b-128).
lab_L_256, lab_a_256, lab_b_256
    The same three means computed after bilinear resize to **256x256**. These
    two blocks differ because JPEG chroma subsampling and resampling change the
    colour statistics; the manuscript's Table 4 must state which one it uses.
rgb_r_native, rgb_g_native, rgb_b_native, rgb_r_256, rgb_g_256, rgb_b_256
    Per-channel means, **0-255 units**.
otsu_threshold
    Otsu threshold on the native grayscale image, **0-255 units**.
otsu_largest_area_px
    Area of the largest Otsu foreground contour in **native pixels**.
otsu_largest_area_frac
    The same area divided by ``native_area_px`` (dimensionless).
otsu_circularity
    ``4*pi*A / P^2`` of that contour (1.0 = perfect circle, dimensionless).
otsu_aspect_ratio
    Width/height of that contour's axis-aligned bounding box (dimensionless).
md5
    MD5 of the raw file bytes, as 32 hexadecimal characters. Cheap (the file is
    read once, in 1 MiB chunks) and lets ``near_duplicates.py`` separate
    **exact** byte-identical duplicates from merely *perceptually* similar ones.
phash_hex
    64-bit perceptual hash as 16 hexadecimal characters, from the 8x8
    low-frequency block (excluding the DC term from the median) of the 2-D DCT
    of the 32x32 grayscale image.

Examples
--------
    python extract_image_features.py --data-root ./data
    python extract_image_features.py --data-root ./data --tasks you_foulbrood --limit 20
    python extract_image_features.py --data-root ./data --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
from PIL import Image

from common.data import TASKS, list_task_images

try:  # SciPy gives the reference DCT-II; OpenCV's cv2.dct is the fallback.
    from scipy.fft import dctn as _dctn

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _dctn = None
    _HAS_SCIPY = False


PHASH_IMG = 32   # DCT input size
PHASH_LOW = 8    # low-frequency block kept -> 64 bits


# --------------------------------------------------------------------------- #
def dct2(a: np.ndarray) -> np.ndarray:
    """Orthonormal 2-D DCT-II. SciPy when available, else OpenCV."""
    a = np.asarray(a, dtype=np.float64)
    if _HAS_SCIPY:
        return _dctn(a, type=2, norm="ortho")
    return cv2.dct(a.astype(np.float32)).astype(np.float64)


def phash64(gray_native: np.ndarray) -> str:
    """64-bit perceptual hash as 16 hex characters.

    Resize to 32x32, take the 2-D DCT, keep the top-left 8x8 low-frequency
    block, threshold against the **median of the block excluding the DC term**
    (the standard pHash recipe; the DC term dominates and would otherwise skew
    the median). Bit order is row-major over the 8x8 block, MSB first.
    """
    small = cv2.resize(gray_native, (PHASH_IMG, PHASH_IMG), interpolation=cv2.INTER_AREA)
    d = dct2(small.astype(np.float64))
    low = d[:PHASH_LOW, :PHASH_LOW]
    flat = low.flatten()
    med = float(np.median(flat[1:]))  # drop DC
    bits = (flat > med).astype(np.uint8)
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return f"{value:016x}"


def lab_means(bgr: np.ndarray) -> Dict[str, float]:
    """CIELAB channel means in CIE units (L* 0-100, a*/b* centred on 0)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    return {
        "L": float(lab[:, :, 0].mean()) * 100.0 / 255.0,
        "a": float(lab[:, :, 1].mean()) - 128.0,
        "b": float(lab[:, :, 2].mean()) - 128.0,
    }


def rgb_means(bgr: np.ndarray) -> Dict[str, float]:
    """Per-channel means in 0-255 units."""
    return {
        "r": float(bgr[:, :, 2].mean()),
        "g": float(bgr[:, :, 1].mean()),
        "b": float(bgr[:, :, 0].mean()),
    }


def otsu_shape(gray_native: np.ndarray) -> Dict[str, float]:
    """Otsu-threshold the native grayscale image and describe its largest blob.

    All areas are reported in **native pixels**; circularity and aspect ratio
    are dimensionless.
    """
    thr, mask = cv2.threshold(gray_native, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = {"otsu_threshold": float(thr), "otsu_largest_area_px": 0.0,
           "otsu_circularity": float("nan"), "otsu_aspect_ratio": float("nan")}
    if not contours:
        return out
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    perim = float(cv2.arcLength(c, True))
    x, y, w, h = cv2.boundingRect(c)
    out["otsu_largest_area_px"] = area
    out["otsu_circularity"] = (4.0 * math.pi * area / (perim ** 2)) if perim > 0 else float("nan")
    out["otsu_aspect_ratio"] = (float(w) / float(h)) if h > 0 else float("nan")
    return out


FEATURE_COLUMNS = [
    "filename", "path", "label", "class_dir", "site", "hive_proxy", "session", "disease_code",
    "native_w", "native_h", "native_area_px", "file_size_bytes",
    "mean_intensity_256",
    "lab_L_native", "lab_a_native", "lab_b_native",
    "lab_L_256", "lab_a_256", "lab_b_256",
    "rgb_r_native", "rgb_g_native", "rgb_b_native",
    "rgb_r_256", "rgb_g_256", "rgb_b_256",
    "otsu_threshold", "otsu_largest_area_px", "otsu_largest_area_frac",
    "otsu_circularity", "otsu_aspect_ratio",
    "md5", "phash_hex",
]


def md5_of_file(path: str, chunk: int = 1 << 20) -> str:
    """MD5 of the raw file bytes, streamed so large JPEGs never load whole."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def features_for_image(sample) -> Optional[Dict[str, object]]:
    """Compute every descriptor for one :class:`common.data.Sample`."""
    path = sample.path
    try:
        with Image.open(path) as im:
            native_w, native_h = im.size
            bgr_native = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception as exc:
        print(f"[warn] unreadable image {path}: {exc}")
        return None

    gray_native = cv2.cvtColor(bgr_native, cv2.COLOR_BGR2GRAY)
    bgr_256 = cv2.resize(bgr_native, (256, 256), interpolation=cv2.INTER_LINEAR)
    gray_256 = cv2.cvtColor(bgr_256, cv2.COLOR_BGR2GRAY)

    lab_n = lab_means(bgr_native)
    lab_s = lab_means(bgr_256)
    rgb_n = rgb_means(bgr_native)
    rgb_s = rgb_means(bgr_256)
    shape = otsu_shape(gray_native)
    area_px = int(native_w) * int(native_h)

    row: Dict[str, object] = {
        "filename": sample.filename,
        "path": path,
        "label": sample.label,
        "class_dir": sample.class_dir,
        "site": sample.site,
        "hive_proxy": sample.hive_proxy,
        "session": sample.session,
        "disease_code": sample.disease_code or "",
        "native_w": int(native_w),
        "native_h": int(native_h),
        "native_area_px": area_px,
        "file_size_bytes": int(os.path.getsize(path)),
        "mean_intensity_256": float(gray_256.mean()),
        "lab_L_native": lab_n["L"], "lab_a_native": lab_n["a"], "lab_b_native": lab_n["b"],
        "lab_L_256": lab_s["L"], "lab_a_256": lab_s["a"], "lab_b_256": lab_s["b"],
        "rgb_r_native": rgb_n["r"], "rgb_g_native": rgb_n["g"], "rgb_b_native": rgb_n["b"],
        "rgb_r_256": rgb_s["r"], "rgb_g_256": rgb_s["g"], "rgb_b_256": rgb_s["b"],
        "otsu_largest_area_frac": (shape["otsu_largest_area_px"] / area_px) if area_px else float("nan"),
        "md5": md5_of_file(path),
        "phash_hex": phash64(gray_native),
    }
    row.update(shape)
    return row


# --------------------------------------------------------------------------- #
def list_pool_images(root: str, extensions=(".jpg", ".jpeg", ".png")):
    """Recursively list every image below ``root`` as a normal (label 0) Sample."""
    from common.data import Sample, parse_filename
    out = []
    for dirpath, _dirs, files in os.walk(root):
        for f in sorted(files):
            if not f.lower().endswith(extensions):
                continue
            p = os.path.join(dirpath, f)
            info = parse_filename(p)
            out.append(Sample(path=p, filename=info["filename"], label=0, class_dir="normal",
                              site=str(info["site"]), hive_proxy=str(info["hive_proxy"]),
                              session=str(info["session"]), disease_code=info.get("disease_code")))
    out.sort(key=lambda s: s.filename)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument("--out-dir", default="./results")
    p.add_argument("--pool-dir", default=None,
                   help="Scan this folder RECURSIVELY for normal (label 0) images of any layout and "
                        "write features_normal_pool.csv (candidate pool for rebuild_matched_subset.py). "
                        "When given, --data-root/--tasks are ignored.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N images per task (smoke testing).")
    p.add_argument("--dry-run", action="store_true",
                   help="Process at most 4 images per task and print rather than write.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    limit = 4 if args.dry_run else args.limit

    jobs = list(args.tasks)
    if args.pool_dir:
        jobs = ["normal_pool"]

    for task in jobs:
        if args.pool_dir:
            samples = list_pool_images(args.pool_dir)
        else:
            samples = list_task_images(args.data_root, task)
        if not samples:
            print(f"[warn] no images for task '{task}' under {args.data_root}")
            continue
        if limit is not None:
            samples = samples[:limit]

        rows: List[Dict[str, object]] = []
        for i, s in enumerate(samples, 1):
            r = features_for_image(s)
            if r is not None:
                rows.append(r)
            if i % 200 == 0:
                print(f"[{task}] {i}/{len(samples)}")

        if args.dry_run:
            print(f"[dry-run] {task}: {len(rows)} rows, first row:")
            if rows:
                for k in FEATURE_COLUMNS:
                    print(f"    {k:26s} {rows[0][k]}")
            continue

        out = os.path.join(args.out_dir, f"features_{task}.csv")
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FEATURE_COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in FEATURE_COLUMNS})
        print(f"[{task}] wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
