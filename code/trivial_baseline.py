#!/usr/bin/env python3
"""Trivial, model-free baselines on global image statistics, plus Table-4 univariate stats.

Motivation
----------
If a single global colour statistic already separates "normal" from "abnormal"
at high balanced accuracy, then a deep model's headline number is not evidence
of lesion recognition. This script quantifies exactly that, using the same
folds as the deep models (from the split manifest) so the comparison is fair.

Baselines
---------
(a) ``threshold_L``
    One-dimensional decision stump on the mean CIELAB L*. The threshold **and
    the polarity** are chosen on each fold's training portion (train + val) by
    maximising balanced accuracy, then applied unchanged to the held-out test
    portion. The "score" used for AUROC is the polarity-signed L*.
(b) ``logreg``
    ``StandardScaler`` + L2 logistic regression on
    ``[L*, a*, b*, native_w, native_h, file_size_bytes]``.
(c) ``gbm``
    ``GradientBoostingClassifier`` on the same six features.

Both fold-wise and pooled out-of-fold BA/AUROC are reported, with seeded
bootstrap confidence intervals on the pooled predictions.

Outputs
-------
``results/trivial_baseline_<task>.csv``     fold-wise + pooled metrics
``results/trivial_oof_<task>_<model>.csv``  per-image out-of-fold predictions
``tables/trivial_baseline.tex``             pooled summary, both tasks
``tables/image_statistics.tex``             Welch t per feature, native **and**
                                            256x256 resolution, units stated

Examples
--------
    python trivial_baseline.py --features-dir ./results --manifest-dir ./results
    python trivial_baseline.py --resolution native --n-boot 500
    python trivial_baseline.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import latex
from common.data import TASKS, read_manifest, task_label
from common.metrics import (
    auroc,
    balanced_accuracy,
    bootstrap_ci,
    hedges_g,
    holm_bonferroni,
    welch_t_from_raw,
)

#: features used by the logistic-regression and gradient-boosting baselines,
#: with the units the manuscript must quote.
FEATURE_UNITS = {
    "lab_L": ("CIELAB $L^*$", "0--100"),
    "lab_a": ("CIELAB $a^*$", "CIE units"),
    "lab_b": ("CIELAB $b^*$", "CIE units"),
    "native_w": ("Image width", "px"),
    "native_h": ("Image height", "px"),
    "file_size_bytes": ("JPEG file size", "bytes"),
    "mean_intensity_256": ("Mean grey level", "0--255"),
    "rgb_r": ("Mean red", "0--255"),
    "rgb_g": ("Mean green", "0--255"),
    "rgb_b": ("Mean blue", "0--255"),
    "otsu_largest_area_px": ("Largest Otsu blob area", "native px"),
    "otsu_largest_area_frac": ("Largest Otsu blob area", "fraction of frame"),
    "otsu_circularity": ("Largest Otsu blob circularity", "dimensionless"),
    "otsu_aspect_ratio": ("Largest Otsu blob aspect ratio", "dimensionless"),
}

#: features whose column name carries a ``_native`` / ``_256`` suffix
RESOLUTION_DEPENDENT = ("lab_L", "lab_a", "lab_b", "rgb_r", "rgb_g", "rgb_b")

CLASSIFIER_FEATURES = ["lab_L", "lab_a", "lab_b", "native_w", "native_h", "file_size_bytes"]

#: features reported in the Table-4-style univariate comparison
UNIVARIATE_FEATURES = [
    "lab_L", "lab_a", "lab_b", "rgb_r", "rgb_g", "rgb_b",
    "mean_intensity_256", "native_w", "native_h", "file_size_bytes",
    "otsu_largest_area_px", "otsu_largest_area_frac",
    "otsu_circularity", "otsu_aspect_ratio",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features-dir", default="./results")
    p.add_argument("--manifest-dir", default="./results")
    p.add_argument("--out-dir", default="./results")
    p.add_argument("--table-dir", default="./tables")
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument("--manifest-prefix", default="split_manifest")
    p.add_argument("--resolution", choices=["256", "native"], default="256",
                   help="Which resolution's colour statistics feed the classifiers.")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="Use 50 bootstrap replicates and write nothing.")
    return p


# --------------------------------------------------------------------------- #
def column_for(feature: str, resolution: str) -> str:
    """Map a logical feature name to the CSV column at the requested resolution."""
    if feature in RESOLUTION_DEPENDENT:
        return f"{feature}_{resolution}"
    return feature


def load_features(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return {r["filename"]: r for r in csv.DictReader(fh)}


def feature_matrix(rows: Sequence[Dict[str, str]], features: Sequence[str],
                   resolution: str) -> np.ndarray:
    cols = [column_for(f, resolution) for f in features]
    return np.array([[float(r[c]) for c in cols] for r in rows], dtype=float)


# --------------------------------------------------------------------------- #
def fit_threshold(x: np.ndarray, y: np.ndarray) -> Tuple[float, int, float]:
    """Best (threshold, polarity, training BA) for a 1-D decision stump.

    Polarity ``+1`` predicts positive when ``x >= t``; ``-1`` predicts positive
    when ``x <= t``. Candidate thresholds are the midpoints between consecutive
    unique values.
    """
    uniq = np.unique(x)
    if uniq.size < 2:
        return float(uniq[0] if uniq.size else 0.0), 1, 0.5
    cands = (uniq[:-1] + uniq[1:]) / 2.0
    best = (float(cands[0]), 1, -1.0)
    for t in cands:
        for pol in (1, -1):
            pred = (x >= t).astype(int) if pol == 1 else (x <= t).astype(int)
            ba = balanced_accuracy(y, pred)
            if np.isfinite(ba) and ba > best[2]:
                best = (float(t), pol, float(ba))
    return best


def apply_threshold(x: np.ndarray, t: float, pol: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(predictions, scores)`` where scores are polarity-signed values."""
    pred = (x >= t).astype(int) if pol == 1 else (x <= t).astype(int)
    return pred, (x if pol == 1 else -x).astype(float)


def run_task(
    task: str,
    features_by_name: Dict[str, Dict[str, str]],
    manifest_rows: Sequence[Dict[str, str]],
    resolution: str,
    seed: int,
    n_boot: int,
) -> Tuple[List[Dict[str, object]], Dict[str, List[Dict[str, object]]]]:
    """Fold-wise + pooled metrics and per-image OOF predictions for all three baselines."""
    folds = sorted({int(r["fold"]) for r in manifest_rows})
    per_model_oof: Dict[str, List[Dict[str, object]]] = {"threshold_L": [], "logreg": [], "gbm": []}
    fold_metrics: List[Dict[str, object]] = []

    lcol = column_for("lab_L", resolution)

    for k in folds:
        rows_k = [r for r in manifest_rows if int(r["fold"]) == k]
        tr_rows = [features_by_name[r["filename"]] for r in rows_k
                   if r["role"] in ("train", "val") and r["filename"] in features_by_name]
        te_rows = [features_by_name[r["filename"]] for r in rows_k
                   if r["role"] == "test" and r["filename"] in features_by_name]
        if not tr_rows or not te_rows:
            print(f"[warn] fold {k}: empty train or test portion; skipped")
            continue
        y_tr = np.array([int(r["label"]) for r in tr_rows])
        y_te = np.array([int(r["label"]) for r in te_rows])
        names_te = [r["filename"] for r in te_rows]

        # ---- (a) single-threshold on mean L* -----------------------------
        x_tr = np.array([float(r[lcol]) for r in tr_rows])
        x_te = np.array([float(r[lcol]) for r in te_rows])
        t, pol, ba_tr = fit_threshold(x_tr, y_tr)
        pred, score = apply_threshold(x_te, t, pol)
        preds = {"threshold_L": (pred, score)}

        # ---- (b)/(c) multivariate ----------------------------------------
        X_tr = feature_matrix(tr_rows, CLASSIFIER_FEATURES, resolution)
        X_te = feature_matrix(te_rows, CLASSIFIER_FEATURES, resolution)
        models = {
            "logreg": Pipeline([("sc", StandardScaler()),
                                ("clf", LogisticRegression(max_iter=2000, random_state=seed))]),
            "gbm": GradientBoostingClassifier(random_state=seed),
        }
        for mname, clf in models.items():
            if len(np.unique(y_tr)) < 2:
                continue
            clf.fit(X_tr, y_tr)
            prob = clf.predict_proba(X_te)[:, 1]
            preds[mname] = ((prob >= 0.5).astype(int), prob)

        for mname, (pred_m, score_m) in preds.items():
            fold_metrics.append({
                "task": task, "model": mname, "fold": k, "scope": "fold",
                "n_test": len(y_te),
                "BA": balanced_accuracy(y_te, pred_m),
                "AUROC": auroc(y_te, score_m),
                "threshold": t if mname == "threshold_L" else "",
                "polarity": pol if mname == "threshold_L" else "",
                "train_BA": ba_tr if mname == "threshold_L" else "",
            })
            for n, yt, pm, sm in zip(names_te, y_te, pred_m, score_m):
                per_model_oof[mname].append(
                    {"filename": n, "fold": k, "y_true": int(yt),
                     "score": float(sm), "pred": int(pm)}
                )

    # ---- pooled out-of-fold ------------------------------------------------
    pooled: List[Dict[str, object]] = []
    for mname, rows in per_model_oof.items():
        if not rows:
            continue
        y = np.array([r["y_true"] for r in rows])
        s = np.array([r["score"] for r in rows], dtype=float)
        pr = np.array([r["pred"] for r in rows])
        # the threshold stump's score is an L* value, not a probability, so
        # rank-normalise it before the bootstrap (AUROC is rank-invariant; the
        # BA still comes from the stored hard predictions).
        s_norm = (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else np.full_like(s, 0.5)
        ci = bootstrap_ci(y, s_norm, pr, n_boot=n_boot, seed=seed)
        pooled.append({
            "task": task, "model": mname, "fold": "pooled", "scope": "pooled",
            "n_test": len(y),
            "BA": ci["BA"][0], "BA_lo": ci["BA"][1], "BA_hi": ci["BA"][2],
            "AUROC": ci["AUROC"][0], "AUROC_lo": ci["AUROC"][1], "AUROC_hi": ci["AUROC"][2],
            "Sensitivity": ci["Sensitivity"][0], "Specificity": ci["Specificity"][0],
            "F1": ci["F1"][0],
        })
        # fold-wise mean +/- SD
        fm = [r for r in fold_metrics if r["model"] == mname]
        for metric in ("BA", "AUROC"):
            vals = np.array([float(r[metric]) for r in fm], dtype=float)
            vals = vals[np.isfinite(vals)]
            pooled[-1][f"{metric}_foldmean"] = float(vals.mean()) if vals.size else float("nan")
            pooled[-1][f"{metric}_foldsd"] = float(vals.std(ddof=1)) if vals.size > 1 else float("nan")

    return fold_metrics + pooled, per_model_oof


# --------------------------------------------------------------------------- #
def univariate_stats(features_by_name: Dict[str, Dict[str, str]], task: str) -> List[Dict[str, object]]:
    """Welch t-test per feature per resolution, Hedges' g, Holm-adjusted p."""
    rows = list(features_by_name.values())
    y = np.array([int(r["label"]) for r in rows])
    out: List[Dict[str, object]] = []
    for feat in UNIVARIATE_FEATURES:
        resolutions = ["native", "256"] if feat in RESOLUTION_DEPENDENT else ["native"]
        for res in resolutions:
            col = column_for(feat, res)
            if not rows or col not in rows[0]:
                continue
            x = np.array([float(r[col]) for r in rows], dtype=float)
            a, b = x[y == 1], x[y == 0]
            if a.size < 2 or b.size < 2:
                continue
            w = welch_t_from_raw(a, b)
            label, unit = FEATURE_UNITS.get(feat, (feat, ""))
            out.append({
                "task": task, "feature": feat, "label": label, "unit": unit,
                "resolution": ("--" if feat not in RESOLUTION_DEPENDENT else
                               ("native" if res == "native" else "256$\\times$256")),
                "mean_abnormal": float(a.mean()), "sd_abnormal": float(a.std(ddof=1)),
                "mean_normal": float(b.mean()), "sd_normal": float(b.std(ddof=1)),
                "t": w["t"], "df": w["df"], "p": w["p"],
                "hedges_g": hedges_g(float(a.mean()), float(a.std(ddof=1)), a.size,
                                     float(b.mean()), float(b.std(ddof=1)), b.size),
            })
    padj, _ = holm_bonferroni([r["p"] for r in out])
    for r, pa in zip(out, padj):
        r["p_holm"] = float(pa)
    return out


def image_statistics_table(all_rows: Sequence[Dict[str, object]]) -> str:
    header = ["Task", "Statistic", "Unit", "Resolution",
              "Abnormal (mean $\\pm$ SD)", "Normal (mean $\\pm$ SD)",
              "$t$", "$p$", "$p_{\\mathrm{Holm}}$", "Hedges' $g$"]
    rows = []
    mid = []
    last_task = None
    for r in all_rows:
        if last_task is not None and r["task"] != last_task:
            mid.append(len(rows) - 1)
        last_task = r["task"]
        big = abs(float(r["mean_abnormal"])) >= 1000 or abs(float(r["mean_normal"])) >= 1000
        dec = 0 if big else 3
        rows.append([
            latex.esc(task_label(str(r["task"]))), r["label"], r["unit"], r["resolution"],
            f"{latex.fmt(r['mean_abnormal'], dec)} $\\pm$ {latex.fmt(r['sd_abnormal'], dec)}",
            f"{latex.fmt(r['mean_normal'], dec)} $\\pm$ {latex.fmt(r['sd_normal'], dec)}",
            latex.fmt(r["t"]), latex.fmt_p(r["p"]), latex.fmt_p(r.get("p_holm")),
            latex.fmt(r["hedges_g"]),
        ])
    return latex.tabular(
        header, rows, align="llllcccccc", midrules_after=mid,
        notes=["Univariate class comparison of global image statistics (Welch's "
               "unequal-variance t-test).",
               "'Resolution' is native (typically 1920x1080) vs. the 256x256 input "
               "actually seen by the networks; '--' marks resolution-independent statistics.",
               "Holm-Bonferroni correction is applied across every row of this table."],
    )


def trivial_table(pooled_rows: Sequence[Dict[str, object]], n_boot: int = 2000) -> str:
    display = {"threshold_L": "Single threshold on mean $L^*$",
               "logreg": "Logistic regression (6 global features)",
               "gbm": "Gradient boosting (6 global features)"}
    header = ["Task", "Trivial baseline", "$N$", "BA (pooled, 95\\% CI)",
              "AUROC (pooled, 95\\% CI)", "BA (fold mean $\\pm$ SD)",
              "AUROC (fold mean $\\pm$ SD)"]
    rows = []
    mid = []
    last = None
    for r in pooled_rows:
        if last is not None and r["task"] != last:
            mid.append(len(rows) - 1)
        last = r["task"]
        rows.append([
            latex.esc(task_label(str(r["task"]))), display.get(str(r["model"]), latex.esc(r["model"])),
            str(r["n_test"]),
            latex.fmt_ci(r["BA"], r["BA_lo"], r["BA_hi"]),
            latex.fmt_ci(r["AUROC"], r["AUROC_lo"], r["AUROC_hi"]),
            latex.fmt_pm(r.get("BA_foldmean"), r.get("BA_foldsd")),
            latex.fmt_pm(r.get("AUROC_foldmean"), r.get("AUROC_foldsd")),
        ])
    return latex.tabular(
        header, rows, align="llccccc", midrules_after=mid,
        notes=["Model-free baselines on the *same* folds as the deep models.",
               "The L* threshold and its polarity are fitted on each fold's "
               "training portion only.",
               f"CIs are seeded stratified percentile bootstraps ({n_boot} replicates) "
               "on the pooled out-of-fold predictions."],
    )


def write_csv(path: str, rows: Sequence[Dict[str, object]], columns: Optional[Sequence[str]] = None) -> str:
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


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    n_boot = 50 if args.dry_run else args.n_boot

    pooled_all: List[Dict[str, object]] = []
    univ_all: List[Dict[str, object]] = []

    for task in args.tasks:
        fpath = os.path.join(args.features_dir, f"features_{task}.csv")
        mpath = os.path.join(args.manifest_dir, f"{args.manifest_prefix}_{task}.csv")
        if not os.path.exists(fpath) or not os.path.exists(mpath):
            print(f"[warn] missing {fpath} or {mpath}; skipping {task}")
            continue
        feats = load_features(fpath)
        manifest = read_manifest(mpath)

        rows, oof = run_task(task, feats, manifest, args.resolution, args.seed, n_boot)
        pooled = [r for r in rows if r["scope"] == "pooled"]
        pooled_all.extend(pooled)
        for r in pooled:
            print(f"[{task}] {r['model']:<12s} pooled BA={r['BA']:.3f} "
                  f"({r['BA_lo']:.3f}-{r['BA_hi']:.3f})  AUROC={r['AUROC']:.3f} "
                  f"({r['AUROC_lo']:.3f}-{r['AUROC_hi']:.3f})")

        univ = univariate_stats(feats, task)
        univ_all.extend(univ)

        if args.dry_run:
            continue
        write_csv(os.path.join(args.out_dir, f"trivial_baseline_{task}.csv"), rows)
        for mname, orows in oof.items():
            write_csv(os.path.join(args.out_dir, f"trivial_oof_{task}_{mname}.csv"), orows,
                      ["filename", "fold", "y_true", "score", "pred"])
        write_csv(os.path.join(args.out_dir, f"image_statistics_{task}.csv"), univ)

    if not args.dry_run:
        if pooled_all:
            print("wrote", latex.write_fragment(
                os.path.join(args.table_dir, "trivial_baseline.tex"), trivial_table(pooled_all, n_boot)))
        if univ_all:
            print("wrote", latex.write_fragment(
                os.path.join(args.table_dir, "image_statistics.tex"), image_statistics_table(univ_all)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
