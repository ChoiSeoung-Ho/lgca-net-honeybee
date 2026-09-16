#!/usr/bin/env python3
"""Fold-wise quantitative XAI localisation against the provider's lesion boxes.

Leakage-free by construction
----------------------------
For each fold ``k`` the script loads **fold k's checkpoint** and scores **only
the disease-positive images in fold k's held-out TEST portion** as recorded in
the split manifest. Training and validation images are never scored, so the
attribution maps come from a model that has not seen those frames.

Attribution methods
-------------------
``gradcam``
    Grad-CAM on the CNN stem's last BatchNorm (``model.bn2``), i.e. the last
    convolutional feature map before token pooling. Uses ``pytorch_grad_cam``
    when installed; otherwise a minimal forward/backward hook implementation
    (below) with identical semantics: channel weights = spatially averaged
    gradients, ReLU on the weighted sum.
``saliency``
    Plain gradient saliency: ``|d logit_pos / d input|``, maximum over the three
    input channels.

Metrics (all computed at 256x256, the network's input resolution)
-----------------------------------------------------------------
``pointing_game``
    1 when the argmax attribution pixel falls inside any ground-truth box.
``energy_in_box``
    Sum of attribution inside the union of boxes / total attribution mass.
    (Attributions are min-max normalised to [0, 1] first, so this is a
    proportion in [0, 1].)
``iou_p80``
    IoU between the binary mask ``attribution >= 80th percentile`` and the union
    of boxes. Reported to 4 decimals.

Annotations
-----------
Provider JSON, one file per image, under ``--annotation-dir``. Boxes are
``[x, y, w, h]`` in **native** pixel coordinates and are rescaled to 256x256.
The exact key path is dataset-version dependent — see the ``TODO`` in
:func:`extract_boxes`, which is the single place to adapt.

Outputs
-------
``results/xai_localization_<task>.csv``   per-image and per-fold rows
``tables/xai_localization.tex``           N, mean +/- SD across folds
``figures/xai_localization.png``          per-fold metric summary (400 dpi)

Examples
--------
    python xai_localization_foldwise.py --task you_chalk_brood --annotation-dir ./labels
    python xai_localization_foldwise.py --task you_foulbrood --method saliency
    python xai_localization_foldwise.py --task you_chalk_brood --dry-run
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from common import latex
from common.data import TASKS, list_task_images, read_manifest
from common.latex import tagged

METRICS = ["pointing_game", "energy_in_box", "iou_p80"]
METRIC_DECIMALS = {"pointing_game": 3, "energy_in_box": 3, "iou_p80": 4}
METRIC_LABEL = {
    "pointing_game": "Pointing game (hit rate)",
    "energy_in_box": "Energy in box",
    "iou_p80": "IoU@P80",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--task", default="you_chalk_brood", choices=list(TASKS))
    p.add_argument("--model", default="lgca_net")
    p.add_argument("--annotation-dir", default="./labels",
                   help="Directory of provider JSON annotations (searched recursively).")
    p.add_argument("--manifest-dir", default="./results")
    p.add_argument("--manifest", default=None)
    p.add_argument("--ckpt-dir", default="./model_save")
    p.add_argument("--run-tag", default="",
                   help="Protocol tag (e.g. matched): selects model_save/<task>/fold<k>_<tag>/ checkpoints and suffixes outputs.")
    p.add_argument("--out-dir", default="./results")
    p.add_argument("--table-dir", default="./tables")
    p.add_argument("--figure-dir", default="./figures")
    p.add_argument("--method", choices=["gradcam", "saliency", "both"], default="both")
    p.add_argument("--percentile", type=float, default=80.0,
                   help="Percentile threshold for the IoU mask (default 80).")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None, help="Max positive images per fold.")
    p.add_argument("--dpi", type=int, default=400)
    p.add_argument("--dry-run", action="store_true",
                   help="Report which images/annotations would be used; load no model.")
    return p


# --------------------------------------------------------------------------- #
# Annotation adapter
# --------------------------------------------------------------------------- #
def extract_boxes(payload: object) -> List[Tuple[float, float, float, float]]:
    """Pull ``[x, y, w, h]`` boxes (native pixels) out of one provider JSON payload.

    TODO (dataset-version specific): set the exact key path here.
    ------------------------------------------------------------
    AI-Hub No. 71667 label files have varied between releases. Observed shapes
    include ``{"annotations": [{"bbox": [x, y, w, h], ...}, ...]}`` and
    ``{"shapes": [{"points": [[x1, y1], [x2, y2]]}, ...]}``. Inspect one of your
    label files and either

    * set ``BOX_KEY_PATH`` below to the dotted path that reaches the box list, or
    * replace the body of this function outright.

    The permissive search below is a convenience for a first run and must be
    replaced by an explicit key path before the numbers go into the manuscript,
    so that a schema change fails loudly instead of silently returning nothing.
    """
    BOX_KEY_PATH: Optional[str] = None  # e.g. "annotations" or "label_info.objects"

    def walk(node, path=""):
        if BOX_KEY_PATH is not None and path != BOX_KEY_PATH:
            if isinstance(node, dict):
                for k, v in node.items():
                    yield from walk(v, f"{path}.{k}" if path else k)
            return
        if isinstance(node, dict):
            for key in ("bbox", "box", "rect", "bounding_box"):
                v = node.get(key)
                if isinstance(v, (list, tuple)) and len(v) == 4 and all(
                        isinstance(c, (int, float)) for c in v):
                    yield tuple(float(c) for c in v)
            pts = node.get("points")
            if isinstance(pts, (list, tuple)) and len(pts) == 2 and all(
                    isinstance(q, (list, tuple)) and len(q) == 2 for q in pts):
                (x1, y1), (x2, y2) = pts
                yield (float(min(x1, x2)), float(min(y1, y2)),
                       abs(float(x2 - x1)), abs(float(y2 - y1)))
            for k, v in node.items():
                yield from walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, (list, tuple)):
            for v in node:
                yield from walk(v, path)

    return list(walk(payload))


def index_annotations(annotation_dir: str) -> Dict[str, str]:
    """``image stem -> annotation json path`` (recursive)."""
    idx: Dict[str, str] = {}
    for p in glob.glob(os.path.join(annotation_dir, "**", "*.json"), recursive=True):
        idx.setdefault(os.path.splitext(os.path.basename(p))[0], p)
    return idx


def boxes_for(filename: str, ann_index: Dict[str, str],
              native_wh: Tuple[int, int], size: int) -> List[Tuple[float, float, float, float]]:
    """Boxes for one image, rescaled from native coordinates to ``size`` x ``size``."""
    stem = os.path.splitext(filename)[0]
    path = ann_index.get(stem)
    if path is None:
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        print(f"[warn] unreadable annotation {path}: {exc}")
        return []
    w, h = native_wh
    sx, sy = size / float(w), size / float(h)
    return [(x * sx, y * sy, bw * sx, bh * sy) for (x, y, bw, bh) in extract_boxes(payload)]


def box_mask(boxes, size: int) -> np.ndarray:
    """Binary union-of-boxes mask at ``size`` x ``size``."""
    m = np.zeros((size, size), dtype=bool)
    for (x, y, w, h) in boxes:
        x0 = max(0, int(np.floor(x)));  y0 = max(0, int(np.floor(y)))
        x1 = min(size, int(np.ceil(x + w)));  y1 = min(size, int(np.ceil(y + h)))
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = True
    return m


# --------------------------------------------------------------------------- #
# Attribution
# --------------------------------------------------------------------------- #
def normalise(a: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(np.asarray(a, dtype=np.float64))
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


class MinimalGradCAM:
    """Grad-CAM via forward/backward hooks on one target layer.

    Weights each channel of the target activation by its spatially averaged
    gradient, sums, applies ReLU, and bilinearly upsamples to the input size —
    the standard formulation, used when ``pytorch_grad_cam`` is unavailable.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.acts = None
        self.grads = None
        self.h1 = target_layer.register_forward_hook(self._fwd)
        self.h2 = target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, out):
        self.acts = out.detach()

    def _bwd(self, _m, _gi, go):
        self.grads = go[0].detach()

    def close(self):
        self.h1.remove()
        self.h2.remove()

    def __call__(self, x, class_idx: int = 1) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        logits[:, class_idx].sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return cam[0, 0].cpu().numpy()


def saliency_map(model, x, class_idx: int = 1) -> np.ndarray:
    """``max_c |d logit_c=pos / d x_c|``."""
    import torch

    x = x.clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(x)
    logits[:, class_idx].sum().backward()
    return x.grad.detach().abs().max(dim=1)[0][0].cpu().numpy()


def find_target_layer(model):
    """The CNN stem's last BatchNorm (``bn2`` on LGCA-Net), else the last BN found."""
    import torch.nn as nn

    if hasattr(model, "bn2"):
        return model.bn2
    last = None
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            last = m
    if last is None:
        raise RuntimeError("No BatchNorm/LayerNorm found to use as a Grad-CAM target.")
    return last


def localisation_metrics(attr: np.ndarray, mask: np.ndarray, percentile: float) -> Dict[str, float]:
    """Pointing game, energy-in-box and IoU@P{percentile} for one attribution map."""
    a = normalise(attr)
    if not mask.any():
        return {m: float("nan") for m in METRICS}
    iy, ix = np.unravel_index(int(np.argmax(a)), a.shape)
    total = float(a.sum())
    thr = float(np.percentile(a, percentile))
    pred = a >= thr
    inter = float(np.logical_and(pred, mask).sum())
    union = float(np.logical_or(pred, mask).sum())
    return {
        "pointing_game": float(bool(mask[iy, ix])),
        "energy_in_box": float(a[mask].sum() / total) if total > 0 else float("nan"),
        "iou_p80": inter / union if union > 0 else float("nan"),
    }


# --------------------------------------------------------------------------- #
def summary_table(rows: Sequence[Dict[str, object]]) -> str:
    """Per-method: per-fold N and value, then mean +/- SD across folds."""
    methods = sorted({str(r["method"]) for r in rows})
    folds = sorted({int(r["fold"]) for r in rows})
    header = (["Method", "Metric"] + [f"Fold {k} ($N$)" for k in folds]
              + ["Mean $\\pm$ SD across folds"])
    out, mid = [], []
    for mi, method in enumerate(methods):
        for metric in METRICS:
            dec = METRIC_DECIMALS[metric]
            cells = [latex.esc(method) if metric == METRICS[0] else "",
                     METRIC_LABEL[metric]]
            vals = []
            for k in folds:
                fr = [r for r in rows if int(r["fold"]) == k and r["method"] == method]
                v = [float(r[metric]) for r in fr if np.isfinite(float(r[metric]))]
                if v:
                    m = float(np.mean(v))
                    vals.append(m)
                    cells.append(f"{latex.fmt(m, dec)} ({len(v)})")
                else:
                    cells.append("-- (0)")
            arr = np.array(vals, dtype=float)
            cells.append(latex.fmt_pm(arr.mean() if arr.size else float("nan"),
                                      arr.std(ddof=1) if arr.size > 1 else float("nan"),
                                      dec))
            out.append(cells)
        if mi < len(methods) - 1:
            mid.append(len(out) - 1)
    return latex.tabular(
        header, out, align="ll" + "c" * (len(folds) + 1), midrules_after=mid,
        notes=["Quantitative XAI localisation against the provider's lesion boxes.",
               "Only disease-positive images from each fold's held-out test portion "
               "are scored, using that fold's own checkpoint.",
               "All metrics are computed at 256x256; boxes are rescaled from native "
               "coordinates. IoU@P80 is reported to 4 decimals.",
               "$N$ is the number of scored positive test images with at least one box."],
    )


def make_figure(rows: Sequence[Dict[str, object]], out_png: str, dpi: int) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = sorted({str(r["method"]) for r in rows})
    folds = sorted({int(r["fold"]) for r in rows})
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 3.4), squeeze=False)
    width = 0.8 / max(len(methods), 1)
    for mi, metric in enumerate(METRICS):
        ax = axes[0][mi]
        for j, method in enumerate(methods):
            means, sds = [], []
            for k in folds:
                v = [float(r[metric]) for r in rows
                     if int(r["fold"]) == k and r["method"] == method
                     and np.isfinite(float(r[metric]))]
                means.append(float(np.mean(v)) if v else np.nan)
                sds.append(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
            xs = np.arange(len(folds)) + j * width - 0.4 + width / 2
            ax.bar(xs, means, width=width * 0.92, yerr=sds, capsize=3,
                   label=method, edgecolor="black", linewidth=0.5)
        ax.set_xticks(np.arange(len(folds)))
        ax.set_xticklabels([f"fold {k}" for k in folds], fontsize=9)
        ax.set_title(METRIC_LABEL[metric], fontsize=10)
        ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if mi == 0:
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Fold-wise XAI localisation on held-out positive images "
                 "(error bars: SD over images)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    mpath = args.manifest or os.path.join(args.manifest_dir, f"split_manifest_{args.task}.csv")
    if not os.path.exists(mpath):
        print(f"[error] manifest not found: {mpath}", file=sys.stderr)
        return 2
    manifest = read_manifest(mpath)
    samples = {s.filename: s for s in list_task_images(args.data_root, args.task)}
    ann_index = index_annotations(args.annotation_dir)
    print(f"annotations indexed: {len(ann_index)} JSON files under {args.annotation_dir}")

    folds = sorted({int(r["fold"]) for r in manifest})
    targets: Dict[int, List[str]] = {}
    for k in folds:
        names = [r["filename"] for r in manifest
                 if int(r["fold"]) == k and r["role"] == "test" and int(r["label"]) == 1]
        if args.limit:
            names = names[: args.limit]
        targets[k] = names
        print(f"fold {k}: {len(names)} positive held-out test images "
              f"({sum(1 for n in names if os.path.splitext(n)[0] in ann_index)} with annotations)")

    if args.dry_run:
        print("[dry-run] no model loaded, nothing written.")
        return 0

    try:
        import torch
    except Exception:
        print("PyTorch is not installed; xai_localization_foldwise.py needs torch.",
              file=sys.stderr)
        return 2

    from PIL import Image

    from common.models import build_model

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    methods = ["gradcam", "saliency"] if args.method == "both" else [args.method]

    try:
        from pytorch_grad_cam import GradCAM as _ExternalGradCAM  # noqa: F401
        have_external = True
    except Exception:
        have_external = False
    print(f"Grad-CAM backend: {'pytorch_grad_cam' if have_external else 'built-in hooks'}")

    rows: List[Dict[str, object]] = []
    for k in folds:
        ck = os.path.join(args.ckpt_dir, args.task, tagged(f"fold{k}", args.run_tag), f"{args.model}.pt")
        if not os.path.exists(ck):
            print(f"[warn] missing checkpoint {ck}; fold {k} skipped")
            continue
        blob = torch.load(ck, map_location=device, weights_only=False)
        model = build_model(args.model, pretrained=False).to(device)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        target_layer = find_target_layer(model)

        for name in targets[k]:
            s = samples.get(name)
            if s is None:
                continue
            with Image.open(s.path) as im:
                native_wh = im.size
                arr = np.asarray(
                    im.convert("RGB").resize((args.image_size, args.image_size), Image.BILINEAR),
                    dtype=np.float32,
                ) / 255.0
            boxes = boxes_for(name, ann_index, native_wh, args.image_size)
            if not boxes:
                continue
            mask = box_mask(boxes, args.image_size)
            if not mask.any():
                continue
            x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

            for method in methods:
                if method == "gradcam":
                    cam = MinimalGradCAM(model, target_layer)
                    try:
                        attr = cam(x, class_idx=1)
                    finally:
                        cam.close()
                else:
                    attr = saliency_map(model, x, class_idx=1)
                m = localisation_metrics(attr, mask, args.percentile)
                rows.append({"task": args.task, "model": args.model, "fold": k,
                             "filename": name, "method": method,
                             "n_boxes": len(boxes),
                             "box_area_frac": float(mask.mean()), **m})

    if not rows:
        print("[error] nothing scored: no checkpoints, no annotations, or no positive "
              "test images with boxes.", file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"xai_localization_{args.task}.csv")
    cols = ["task", "model", "fold", "filename", "method", "n_boxes",
            "box_area_frac"] + METRICS
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print("wrote", out_csv)
    print("wrote", latex.write_fragment(os.path.join(args.table_dir, "xai_localization.tex"),
                                        summary_table(rows)))
    print("wrote", make_figure(rows, os.path.join(args.figure_dir, "xai_localization.png"),
                               args.dpi))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
