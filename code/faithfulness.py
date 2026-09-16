#!/usr/bin/env python3
"""Faithfulness of the saliency explanations: perturbation curves, Average Drop, Increase in Confidence.

Localisation (``xai_localization_foldwise.py``) asks whether an attribution map
lands on the annotated lesion. **Faithfulness** asks a different and
complementary question: does the map actually identify the pixels the model
relies on? A map can point at the lesion and still be unfaithful, and vice
versa, so both are reported.

As in the localisation script, each fold is scored with **that fold's own
checkpoint on that fold's held-out test images**, never on training data.

Metrics
-------
``Positive perturbation AUC x100`` (lower is better)
    Delete (set to 0) the fraction ``x`` of *highest*-attribution pixels for
    ``x`` in ``{0.1, ..., 0.9}`` and record accuracy with respect to the
    **originally predicted class**. A faithful map removes the evidence quickly,
    so accuracy collapses and the area under the accuracy-vs-``x`` curve is
    small. AUC is the trapezoidal integral over the ``x`` grid divided by its
    range (so it stays on the accuracy scale), times 100.
``Negative perturbation AUC x100`` (higher is better)
    The same, deleting the *lowest*-attribution pixels first. A faithful map
    keeps accuracy high, because only irrelevant pixels are being destroyed.
``Average Drop %`` (lower is better)
    Chattopadhay et al. (Grad-CAM++, WACV 2018):
    ``mean_i max(0, Y_i - O_i) / Y_i x 100``, where ``Y_i`` is the model's score
    for the target class on the original image and ``O_i`` the score on the
    **explanation map**: the image multiplied element-wise by the min-max
    normalised attribution.
``Increase in Confidence %`` (higher is better)
    Same reference: ``mean_i 1[O_i > Y_i] x 100``.

The target class is the model's originally predicted class, so the measurement
never depends on the ground-truth label.

Attribution methods
-------------------
``gradcam``   Grad-CAM on the CNN stem's last BatchNorm (``bn2``).
``saliency``  ``max_c |d logit / d x_c|``.
``lime``      LIME image explanations via the optional ``lime`` package
              (guarded import; skipped with a clear message when absent, since
              it is not in ``requirements.txt``).

Outputs
-------
``results/faithfulness_<task>.csv``   per fold x method, plus the per-x curves
``tables/faithfulness.tex``           rows disease x method, averaged over folds

Examples
--------
    python faithfulness.py --task you_chalk_brood
    python faithfulness.py --task you_foulbrood --methods gradcam saliency --limit 50
    python faithfulness.py --task you_chalk_brood --dry-run
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from common import latex
from common.data import TASKS, list_task_images, read_manifest, task_label

#: perturbation fractions
DEFAULT_X_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

METHOD_LABEL = {"gradcam": "Grad-CAM", "saliency": "Saliency", "lime": "LIME"}

#: column key -> (header, arrow, decimals)
FAITHFULNESS_COLUMNS = [
    ("pos_auc_x100", "Pos. perturbation AUC $\\times$100", "$\\downarrow$", 2),
    ("neg_auc_x100", "Neg. perturbation AUC $\\times$100", "$\\uparrow$", 2),
    ("average_drop_pct", "Average Drop \\%", "$\\downarrow$", 2),
    ("increase_confidence_pct", "Increase in Confidence \\%", "$\\uparrow$", 2),
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--task", default="you_chalk_brood", choices=list(TASKS))
    p.add_argument("--model", default="lgca_net")
    p.add_argument("--manifest-dir", default="./results")
    p.add_argument("--manifest", default=None)
    p.add_argument("--ckpt-dir", default="./model_save")
    p.add_argument("--out-dir", default="./results")
    p.add_argument("--table-dir", default="./tables")
    p.add_argument("--methods", nargs="+", default=["gradcam", "saliency", "lime"],
                   choices=["gradcam", "saliency", "lime"])
    p.add_argument("--x-grid", nargs="+", type=float, default=list(DEFAULT_X_GRID),
                   help="Perturbation fractions (default 0.1 ... 0.9).")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None,
                   help="Max held-out test images scored per fold.")
    p.add_argument("--lime-samples", type=int, default=1000,
                   help="LIME perturbation samples per image (LIME is slow).")
    p.add_argument("--run-tag", default="",
                   help="Suffix identifying this protocol; selects the tagged "
                        "checkpoints and suffixes the outputs.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="Report which images would be scored; load no model.")
    return p


# --------------------------------------------------------------------------- #
# Attribution
# --------------------------------------------------------------------------- #
def normalise(a: np.ndarray) -> np.ndarray:
    """Min-max normalise an attribution map to [0, 1]."""
    a = np.nan_to_num(np.asarray(a, dtype=np.float64))
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


class MinimalGradCAM:
    """Grad-CAM via forward/backward hooks on one target layer (see xai_localization_foldwise)."""

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

    def __call__(self, x, class_idx: int) -> np.ndarray:
        import torch.nn.functional as F

        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        logits[:, class_idx].sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return cam[0, 0].cpu().numpy()


def saliency_map(model, x, class_idx: int) -> np.ndarray:
    """``max_c |d logit_class / d x_c|``."""
    x = x.clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    model(x)[:, class_idx].sum().backward()
    return x.grad.detach().abs().max(dim=1)[0][0].cpu().numpy()


def lime_map(model, arr_hwc: np.ndarray, class_idx: int, device,
             n_samples: int, seed: int) -> Optional[np.ndarray]:
    """LIME image explanation as a per-pixel map, or None when ``lime`` is absent."""
    try:
        from lime import lime_image
    except Exception:
        return None
    import torch

    def predict(batch: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(batch), 32):
                chunk = torch.from_numpy(
                    batch[i:i + 32].astype(np.float32)
                ).permute(0, 3, 1, 2).to(device)
                out.append(torch.softmax(model(chunk), dim=1).cpu().numpy())
        return np.concatenate(out, axis=0)

    explainer = lime_image.LimeImageExplainer(random_state=seed)
    exp = explainer.explain_instance(
        arr_hwc.astype(np.double), predict, labels=(class_idx,), top_labels=None,
        hide_color=0, num_samples=int(n_samples), random_seed=seed,
    )
    segments = exp.segments
    weights = dict(exp.local_exp[class_idx])
    out = np.zeros(segments.shape, dtype=np.float64)
    for seg_id, w in weights.items():
        out[segments == seg_id] = w
    # LIME weights are signed; faithfulness ranks *evidence for* the class, so
    # negative-contribution superpixels are floored at zero rather than ranked
    # by magnitude (which would conflate evidence for and against).
    return np.maximum(out, 0.0)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def perturbation_curves(
    model, x, attr: np.ndarray, target: int, x_grid: Sequence[float], device,
) -> Tuple[List[float], List[float]]:
    """Accuracy w.r.t. ``target`` after deleting the top-x / bottom-x attribution pixels.

    Returns ``(positive_hits, negative_hits)``: for each fraction, 1.0 when the
    perturbed image is still classified as ``target``, else 0.0.
    """
    import torch

    a = normalise(attr).ravel()
    n = a.size
    order_desc = np.argsort(-a, kind="stable")   # highest attribution first
    order_asc = np.argsort(a, kind="stable")     # lowest attribution first

    pos_hits, neg_hits = [], []
    for frac in x_grid:
        k = int(round(float(frac) * n))
        for order, sink in ((order_desc, pos_hits), (order_asc, neg_hits)):
            mask = np.ones(n, dtype=np.float32)
            if k > 0:
                mask[order[:k]] = 0.0
            m = torch.from_numpy(mask.reshape(attr.shape)).to(device)
            with torch.no_grad():
                pred = int(model(x * m[None, None, :, :]).argmax(dim=1).item())
            sink.append(1.0 if pred == target else 0.0)
    return pos_hits, neg_hits


def curve_auc(values: Sequence[float], x_grid: Sequence[float]) -> float:
    """Trapezoidal AUC over the x grid, normalised by its range (accuracy scale)."""
    xs = np.asarray(x_grid, dtype=float)
    ys = np.asarray(values, dtype=float)
    if xs.size < 2:
        return float(ys.mean()) if ys.size else float("nan")
    span = float(xs[-1] - xs[0])
    if span <= 0:
        return float(ys.mean())
    return float(np.trapezoid(ys, xs) / span)


def drop_and_increase(model, x, attr: np.ndarray, target: int, device) -> Tuple[float, float]:
    """Average-Drop contribution and confidence-increase indicator for one image.

    The explanation map is the input multiplied element-wise by the min-max
    normalised attribution (Chattopadhay et al., Grad-CAM++).
    """
    import torch

    a = torch.from_numpy(normalise(attr).astype(np.float32)).to(device)
    with torch.no_grad():
        y = float(torch.softmax(model(x), dim=1)[0, target])
        o = float(torch.softmax(model(x * a[None, None, :, :]), dim=1)[0, target])
    drop = max(0.0, y - o) / y if y > 0 else float("nan")
    return drop, 1.0 if o > y else 0.0


# --------------------------------------------------------------------------- #
# Table
# --------------------------------------------------------------------------- #
def faithfulness_table(rows: Sequence[Dict[str, object]]) -> str:
    """Rows = task x method, values averaged over folds."""
    tasks = list(dict.fromkeys(str(r["task"]) for r in rows))
    methods = [m for m in ("gradcam", "saliency", "lime")
               if any(str(r["method"]) == m for r in rows)]
    header = (["Disease", "Method"]
              + [f"{h} {arrow}" for _, h, arrow, _ in FAITHFULNESS_COLUMNS]
              + ["Folds"])
    out: List[List[str]] = []
    mid: List[int] = []
    for ti, task in enumerate(tasks):
        for method in methods:
            sub = [r for r in rows if r["task"] == task and r["method"] == method]
            if not sub:
                continue
            cells = [latex.esc(task_label(task)), METHOD_LABEL.get(method, method)]
            for key, _, _, dec in FAITHFULNESS_COLUMNS:
                vals = np.array([float(r[key]) for r in sub
                                 if r.get(key) not in (None, "")], dtype=float)
                vals = vals[np.isfinite(vals)]
                cells.append(latex.fmt(vals.mean(), dec) if vals.size else "--")
            cells.append(str(len({int(r["fold"]) for r in sub})))
            out.append(cells)
        if ti < len(tasks) - 1:
            mid.append(len(out) - 1)
    return latex.tabular(
        header, out, align="ll" + "c" * (len(FAITHFULNESS_COLUMNS) + 1),
        midrules_after=mid,
        notes=["Faithfulness of the saliency explanations, averaged over the "
               "cross-validation folds; each fold is scored with its own checkpoint "
               "on its own held-out test images.",
               "Perturbation AUCs: accuracy w.r.t. the originally predicted class "
               "after deleting the top (positive) or bottom (negative) fraction "
               "$x \\in \\{0.1, \\dots, 0.9\\}$ of attribution pixels; trapezoidal "
               "area over $x$, normalised by its range, $\\times 100$.",
               "Average Drop and Increase in Confidence follow Chattopadhay et al. "
               "(Grad-CAM++), computed on the image multiplied by the min-max "
               "normalised attribution map.",
               "Arrows give the preferred direction."],
    )


# --------------------------------------------------------------------------- #
# Legacy CSV ingestion (used by make_tables.py)
# --------------------------------------------------------------------------- #
def _is_int(value: object) -> bool:
    try:
        int(str(value).strip())
        return True
    except (TypeError, ValueError):
        return False


def _norm_method(name: str) -> str:
    """Map a legacy method label onto one of gradcam/saliency/lime."""
    key = str(name).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if "gradcam" in key or key == "cam":
        return "gradcam"
    if "salien" in key or "grad" == key:
        return "saliency"
    if "lime" in key:
        return "lime"
    return key


def load_legacy_faithfulness(pos_neg_dir: str = "./results_POS_NEG",
                             drop_inc_dir: str = "./results_DROP_INC",
                             tasks: Sequence[str] = TASKS) -> List[Dict[str, object]]:
    """Rebuild faithfulness rows from the author's older per-fold CSV exports.

    Reads ``<pos_neg_dir>/<task>_pos_neg_perturb_auc.csv`` (columns
    ``Fold,Method,...,PosAUC_x100,NegAUC_x100``) and
    ``<drop_inc_dir>/<task>_grad_saliency_drop_inc.csv`` /
    ``<task>_lime_shap_drop_inc.csv`` (columns ``Fold,Method,
    AverageDropPercent,IncreaseInConfidencePercent``).

    Only rows whose ``Fold`` parses as an integer are kept -- the legacy files
    carry trailing ``mean``/``std`` summary rows that must not be averaged in
    again.
    """
    merged: Dict[Tuple[str, int, str], Dict[str, object]] = {}

    for task in tasks:
        path = os.path.join(pos_neg_dir, f"{task}_pos_neg_perturb_auc.csv")
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if not _is_int(r.get("Fold")):
                    continue
                fold = int(str(r["Fold"]).strip())
                method = _norm_method(r.get("Method", ""))
                key = (task, fold, method)
                row = merged.setdefault(key, {"task": task, "fold": fold,
                                              "method": method, "source": "legacy"})
                for src, dst in (("PosAUC_x100", "pos_auc_x100"),
                                 ("NegAUC_x100", "neg_auc_x100")):
                    if r.get(src) not in (None, ""):
                        row[dst] = float(r[src])
                # fall back to the un-scaled columns when only those exist
                for src, dst in (("PosAUC", "pos_auc_x100"), ("NegAUC", "neg_auc_x100")):
                    if dst not in row and r.get(src) not in (None, ""):
                        row[dst] = 100.0 * float(r[src])

    for task in tasks:
        for stem in (f"{task}_grad_saliency_drop_inc.csv", f"{task}_lime_shap_drop_inc.csv"):
            path = os.path.join(drop_inc_dir, stem)
            if not os.path.exists(path):
                continue
            with open(path, newline="", encoding="utf-8-sig") as fh:
                for r in csv.DictReader(fh):
                    if not _is_int(r.get("Fold")):
                        continue
                    fold = int(str(r["Fold"]).strip())
                    method = _norm_method(r.get("Method", ""))
                    key = (task, fold, method)
                    row = merged.setdefault(key, {"task": task, "fold": fold,
                                                  "method": method, "source": "legacy"})
                    if r.get("AverageDropPercent") not in (None, ""):
                        row["average_drop_pct"] = float(r["AverageDropPercent"])
                    if r.get("IncreaseInConfidencePercent") not in (None, ""):
                        row["increase_confidence_pct"] = float(r["IncreaseInConfidencePercent"])

    return [merged[k] for k in sorted(merged)]


# --------------------------------------------------------------------------- #
def write_csv(path: str, rows: Sequence[Dict[str, object]],
              columns: Optional[Sequence[str]] = None) -> str:
    if not rows:
        return path
    cols = list(columns) if columns else sorted({k for r in rows for k in r})
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return path


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    mpath = args.manifest or os.path.join(args.manifest_dir, f"split_manifest_{args.task}.csv")
    if not os.path.exists(mpath):
        print(f"[error] manifest not found: {mpath}", file=sys.stderr)
        return 2
    manifest = read_manifest(mpath)
    samples = {s.filename: s for s in list_task_images(args.data_root, args.task)}
    for r in manifest:
        if r["filename"] not in samples and (r.get("path") or "").strip():
            pass  # path resolved below from the manifest
    folds = sorted({int(r["fold"]) for r in manifest})

    targets: Dict[int, List[Dict[str, str]]] = {}
    for k in folds:
        rows = [r for r in manifest if int(r["fold"]) == k and r["role"] == "test"]
        if args.limit:
            rows = rows[: args.limit]
        targets[k] = rows
        print(f"fold {k}: {len(rows)} held-out test images to score")

    if args.dry_run:
        print("[dry-run] no model loaded, nothing written.")
        return 0

    try:
        import torch
    except Exception:
        print("PyTorch is not installed; faithfulness.py needs torch>=2.9.0.", file=sys.stderr)
        return 2

    from PIL import Image

    from common.latex import tagged
    from common.models import build_model
    from common.seed import set_seed

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    methods = list(args.methods)
    if "lime" in methods:
        try:
            import lime  # noqa: F401
        except Exception:
            print("[warn] the 'lime' package is not installed (pip install lime); "
                  "the LIME rows are skipped.")
            methods = [m for m in methods if m != "lime"]

    per_image: List[Dict[str, object]] = []
    for k in folds:
        ck = os.path.join(args.ckpt_dir, args.task,
                          tagged(f"fold{k}", args.run_tag), f"{args.model}.pt")
        if not os.path.exists(ck):
            print(f"[warn] missing checkpoint {ck}; fold {k} skipped")
            continue
        blob = torch.load(ck, map_location=device, weights_only=False)
        model = build_model(args.model, pretrained=False).to(device)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        target_layer = getattr(model, "bn2", None)
        if target_layer is None:
            import torch.nn as nn
            target_layer = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)][-1]

        for row in targets[k]:
            name = row["filename"]
            s = samples.get(name)
            path = s.path if s is not None else (row.get("path") or "")
            if not path or not os.path.exists(path):
                continue
            with Image.open(path) as im:
                arr = np.asarray(
                    im.convert("RGB").resize((args.image_size, args.image_size), Image.BILINEAR),
                    dtype=np.float32,
                ) / 255.0
            x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                target = int(model(x).argmax(dim=1).item())

            for method in methods:
                if method == "gradcam":
                    cam = MinimalGradCAM(model, target_layer)
                    try:
                        attr = cam(x, target)
                    finally:
                        cam.close()
                elif method == "saliency":
                    attr = saliency_map(model, x, target)
                else:
                    attr = lime_map(model, arr, target, device, args.lime_samples, args.seed)
                    if attr is None:
                        continue
                pos, neg = perturbation_curves(model, x, attr, target, args.x_grid, device)
                drop, inc = drop_and_increase(model, x, attr, target, device)
                per_image.append({
                    "task": args.task, "model": args.model, "fold": k, "filename": name,
                    "method": method, "target_class": target,
                    "pos_auc_x100": 100.0 * curve_auc(pos, args.x_grid),
                    "neg_auc_x100": 100.0 * curve_auc(neg, args.x_grid),
                    "average_drop_pct": 100.0 * drop,
                    "increase_confidence_pct": 100.0 * inc,
                    **{f"pos_x{int(round(100 * xx))}": v for xx, v in zip(args.x_grid, pos)},
                    **{f"neg_x{int(round(100 * xx))}": v for xx, v in zip(args.x_grid, neg)},
                })
        print(f"  fold {k}: scored {sum(1 for r in per_image if r['fold'] == k)} "
              f"image-method rows")

    if not per_image:
        print("[error] nothing scored: no checkpoints, or no resolvable test images.",
              file=sys.stderr)
        return 1

    # aggregate to fold x method (the table then averages over folds)
    fold_rows: List[Dict[str, object]] = []
    for k in sorted({int(r["fold"]) for r in per_image}):
        for method in methods:
            sub = [r for r in per_image if int(r["fold"]) == k and r["method"] == method]
            if not sub:
                continue
            agg: Dict[str, object] = {"task": args.task, "model": args.model,
                                      "fold": k, "method": method, "n_images": len(sub),
                                      "source": "computed"}
            for key, _, _, _ in FAITHFULNESS_COLUMNS:
                vals = np.array([float(r[key]) for r in sub], dtype=float)
                vals = vals[np.isfinite(vals)]
                agg[key] = float(vals.mean()) if vals.size else float("nan")
            fold_rows.append(agg)

    tag = args.run_tag
    os.makedirs(args.out_dir, exist_ok=True)
    print("wrote", write_csv(
        os.path.join(args.out_dir, tagged(f"faithfulness_{args.task}", tag) + ".csv"), fold_rows))
    print("wrote", write_csv(
        os.path.join(args.out_dir, tagged(f"faithfulness_perimage_{args.task}", tag) + ".csv"),
        per_image))
    print("wrote", latex.write_fragment(
        os.path.join(args.table_dir, tagged("faithfulness", tag) + ".tex"),
        faithfulness_table(fold_rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
