"""Dataset listing, AI-Hub filename parsing, cross-validation splitting and manifests.

Data layout expected by every script in this package::

    <data-root>/<task>/normal/*.jpg
    <data-root>/<task>/abnormal/*.jpg

with ``task`` in {``you_chalk_brood``, ``you_foulbrood``}.

Filename convention (AI-Hub honeybee dataset No. 71667), e.g.::

    B_001_018_20230819135434_001_004_000_002.jpg
    f1 f2  f3  f4             f5  f6  f7  f8

The data provider does not document the semantics of f1, f2, f3, f5, f6, f7.
Empirically ``f1_f2`` and ``f1_f2_f3`` behave like nested acquisition
identifiers (many images share them, and they co-vary with capture date), so we
use them strictly as **proxy acquisition-unit identifiers**:

* ``site``       = ``f1_f2``            (coarse proxy)
* ``hive_proxy`` = ``f1_f2_f3``         (grouping unit for grouped CV)
* ``session``    = ``f1_f2_f3`` + date  (first 8 characters of f4)

``f4`` is a ``YYYYMMDDhhmmss`` timestamp and ``f8`` is a disease code
(``000`` normal, ``002`` chalkbrood, ``003`` foulbrood).

PyTorch is imported behind a guard: everything except ``ImageFolderDataset``
works without it, so the CPU-only analysis scripts can import this module.
"""

from __future__ import annotations

import csv
import glob
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
try:  # scikit-learn is only needed by the split builders, not by the feature extractor.
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split
except Exception:  # pragma: no cover
    StratifiedKFold = StratifiedGroupKFold = train_test_split = None  # type: ignore[assignment]

try:  # pragma: no cover - only exercised when torch is installed
    import torch
    from torch.utils.data import Dataset as _TorchDataset

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TorchDataset = object  # type: ignore[misc,assignment]
    _HAS_TORCH = False


TASKS = ("you_chalk_brood", "you_foulbrood")
CLASS_DIRS = ("normal", "abnormal")
LABEL_OF_CLASS_DIR = {"normal": 0, "abnormal": 1}

#: Disease codes as documented by the provider.
DISEASE_CODES = {"000": "normal", "002": "chalkbrood", "003": "foulbrood"}

#: Expected positive disease code per task (used only for sanity reporting).
TASK_POSITIVE_CODE = {"you_chalk_brood": "002", "you_foulbrood": "003"}

_FIELD_RE = re.compile(r"^(?P<f1>[^_]+)_(?P<f2>[^_]+)_(?P<f3>[^_]+)_(?P<f4>[^_]+)_"
                       r"(?P<f5>[^_]+)_(?P<f6>[^_]+)_(?P<f7>[^_]+)_(?P<f8>[^_.]+)$")


# --------------------------------------------------------------------------- #
# Filename parsing
# --------------------------------------------------------------------------- #
def parse_filename(path_or_name: str) -> Dict[str, Optional[str]]:
    """Parse an AI-Hub honeybee filename into its proxy grouping identifiers.

    Parameters
    ----------
    path_or_name:
        Full path or bare file name. The extension is stripped.

    Returns
    -------
    dict
        Keys: ``filename``, ``f1``..``f8``, ``site``, ``hive_proxy``,
        ``session``, ``date``, ``datetime``, ``disease_code``,
        ``disease_name``, ``parsed``.

    Non-conforming names do not raise: they come back with ``parsed=False`` and
    the whole stem used as ``site``/``hive_proxy``/``session`` so that grouped
    CV degrades to "one group per odd file" rather than crashing.
    """
    name = os.path.basename(path_or_name)
    stem = os.path.splitext(name)[0]
    out: Dict[str, Optional[str]] = {"filename": name, "parsed": False}

    m = _FIELD_RE.match(stem)
    if m is None:
        out.update(
            {f"f{i}": None for i in range(1, 9)}
        )
        out.update(
            site=stem, hive_proxy=stem, session=stem, date=None,
            datetime=None, disease_code=None, disease_name=None,
        )
        return out

    g = m.groupdict()
    out.update(g)
    f4 = g["f4"] or ""
    date = f4[:8] if len(f4) >= 8 else None
    out.update(
        parsed=True,
        site=f"{g['f1']}_{g['f2']}",
        hive_proxy=f"{g['f1']}_{g['f2']}_{g['f3']}",
        session=f"{g['f1']}_{g['f2']}_{g['f3']}_{date}" if date else f"{g['f1']}_{g['f2']}_{g['f3']}",
        date=date,
        datetime=f4 if len(f4) == 14 else None,
        disease_code=g["f8"],
        disease_name=DISEASE_CODES.get(g["f8"] or "", "unknown"),
    )
    return out


# --------------------------------------------------------------------------- #
# Dataset listing
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    """One image and everything derived from its path/name."""

    path: str
    filename: str
    label: int
    class_dir: str
    site: str
    hive_proxy: str
    session: str
    disease_code: Optional[str]

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def list_task_images(data_root: str, task: str, extensions: Sequence[str] = (".jpg", ".jpeg", ".png")) -> List[Sample]:
    """List every image of one task as :class:`Sample` records, sorted by filename.

    Sorting is deterministic (by class directory then filename) so that split
    assignment does not depend on filesystem ordering.
    """
    samples: List[Sample] = []
    for class_dir in CLASS_DIRS:
        d = os.path.join(data_root, task, class_dir)
        if not os.path.isdir(d):
            continue
        paths: List[str] = []
        for ext in extensions:
            paths.extend(glob.glob(os.path.join(d, "*" + ext)))
            paths.extend(glob.glob(os.path.join(d, "*" + ext.upper())))
        for p in sorted(set(paths)):
            info = parse_filename(p)
            samples.append(
                Sample(
                    path=p,
                    filename=info["filename"],  # type: ignore[arg-type]
                    label=LABEL_OF_CLASS_DIR[class_dir],
                    class_dir=class_dir,
                    site=str(info["site"]),
                    hive_proxy=str(info["hive_proxy"]),
                    session=str(info["session"]),
                    disease_code=info["disease_code"],
                )
            )
    samples.sort(key=lambda s: (s.class_dir, s.filename))
    return samples


def samples_to_arrays(samples: Sequence[Sample]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Return ``(paths, labels, groups)`` where groups are the hive-proxy ids."""
    paths = [s.path for s in samples]
    labels = np.asarray([s.label for s in samples], dtype=int)
    groups = np.asarray([s.hive_proxy for s in samples], dtype=object)
    return paths, labels, groups


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def make_stratified_folds(
    files: Sequence[str],
    labels: Sequence[int],
    n_splits: int = 3,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Stratified K-fold indices.

    Replaces the original code's ``KFold(3, shuffle=True, random_state=42)``,
    which was *not* stratified. Returns a list of ``(train_idx, test_idx)``.
    """
    y = np.asarray(labels, dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in skf.split(np.zeros(len(y)), y)]


def make_grouped_folds(
    files: Sequence[str],
    labels: Sequence[int],
    groups: Sequence[str],
    n_splits: int = 3,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Hive-proxy-grouped, class-balanced folds via :class:`StratifiedGroupKFold`.

    No hive-proxy identifier appears in both the training and the test portion
    of a fold, so near-duplicate frames from the same acquisition unit cannot
    leak across the split.
    """
    y = np.asarray(labels, dtype=int)
    g = np.asarray(groups, dtype=object)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in sgkf.split(np.zeros(len(y)), y, groups=g)]


def make_val_split(
    train_idx: Sequence[int],
    labels: Sequence[int],
    val_fraction: float = 0.25,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified validation carve-out from a fold's training portion.

    Replaces the original "last 25% of the shuffled training indices" rule with
    ``train_test_split(..., stratify=y_train, random_state=seed)``.

    Returns ``(inner_train_idx, val_idx)`` as indices into the *original* array.
    """
    train_idx = np.asarray(train_idx, dtype=int)
    y = np.asarray(labels, dtype=int)[train_idx]
    inner_tr, inner_va = train_test_split(
        train_idx, test_size=val_fraction, stratify=y, random_state=seed, shuffle=True
    )
    return np.sort(np.asarray(inner_tr)), np.sort(np.asarray(inner_va))


def build_manifest_rows(
    samples: Sequence[Sample],
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    val_fraction: float = 0.25,
    seed: int = 42,
) -> List[Dict[str, object]]:
    """Expand folds into one manifest row per (image, fold) with role train/val/test."""
    labels = [s.label for s in samples]
    rows: List[Dict[str, object]] = []
    for k, (tr, te) in enumerate(folds):
        inner_tr, inner_va = make_val_split(tr, labels, val_fraction=val_fraction, seed=seed)
        role = {}
        for i in inner_tr:
            role[int(i)] = "train"
        for i in inner_va:
            role[int(i)] = "val"
        for i in te:
            role[int(i)] = "test"
        for i in range(len(samples)):
            if i not in role:
                continue
            s = samples[i]
            rows.append(
                {
                    "filename": s.filename,
                    "path": s.path,
                    "label": s.label,
                    "class_dir": s.class_dir,
                    "group": s.hive_proxy,
                    "site": s.site,
                    "session": s.session,
                    "fold": k,
                    "role": role[int(i)],
                }
            )
    return rows


MANIFEST_COLUMNS = [
    "filename", "path", "label", "class_dir", "group", "site", "session", "fold", "role",
]


def write_manifest(rows: Sequence[Dict[str, object]], out_csv: str) -> str:
    """Write a split manifest CSV (``filename,label,group,fold,role`` plus extras)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in MANIFEST_COLUMNS})
    return out_csv


def read_manifest(path: str) -> List[Dict[str, str]]:
    """Read a split manifest CSV back as a list of dicts (all values as str)."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def manifest_fold_indices(
    rows: Sequence[Dict[str, str]], fold: int
) -> Dict[str, List[str]]:
    """Filenames per role for one fold of a manifest."""
    out: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for r in rows:
        if int(r["fold"]) != int(fold):
            continue
        out.setdefault(r["role"], []).append(r["filename"])
    return out


# --------------------------------------------------------------------------- #
# Torch dataset
# --------------------------------------------------------------------------- #
class ImageFolderDataset(_TorchDataset):  # type: ignore[misc]
    """Lazy image dataset: resize to ``size`` x ``size``, scale to [0, 1], NCHW float32.

    Matches the original pipeline exactly: bilinear resize to 256x256, divide by
    255, **no augmentation**, no ImageNet mean/std normalisation.

    Parameters
    ----------
    paths, labels:
        Parallel sequences.
    size:
        Square output size (256 in the paper; some baselines override it inside
        their wrapper, e.g. CrossViT at 240).
    """

    def __init__(self, paths: Sequence[str], labels: Sequence[int], size: int = 256):
        if not _HAS_TORCH:
            raise ImportError(
                "ImageFolderDataset requires PyTorch. Install torch (>=2.9.0) or use the "
                "CPU-only analysis scripts, which do not touch this class."
            )
        self.paths = list(paths)
        self.labels = [int(v) for v in labels]
        self.size = int(size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        from PIL import Image

        img = Image.open(self.paths[idx]).convert("RGB").resize(
            (self.size, self.size), Image.BILINEAR
        )
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC in [0, 1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return tensor, self.labels[idx], os.path.basename(self.paths[idx])


def make_dataloader(paths, labels, size=256, batch_size=32, shuffle=False, num_workers=4, seed=42):
    """Convenience DataLoader factory with deterministic worker seeding."""
    if not _HAS_TORCH:
        raise ImportError("make_dataloader requires PyTorch.")
    from torch.utils.data import DataLoader

    from .seed import worker_init_fn

    ds = ImageFolderDataset(paths, labels, size=size)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=g,
        # BatchNorm1d 는 배치 1개로 학습 불가 -> 나머지가 1인 학습 로더에서만 마지막 배치 제외
        drop_last=(bool(shuffle) and len(ds) % batch_size == 1),
    )


#: Human-readable task names for tables and figure panels.
TASK_LABEL = {"you_chalk_brood": "Chalkbrood", "you_foulbrood": "Foulbrood"}


def task_label(task: str) -> str:
    """Pretty task name for LaTeX tables/figures, falling back to the raw key."""
    return TASK_LABEL.get(task, task)
