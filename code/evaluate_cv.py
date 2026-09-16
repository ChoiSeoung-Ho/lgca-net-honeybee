#!/usr/bin/env python3
"""Evaluate cross-validation OOF predictions: pooled CIs, fold-wise spread, paired tests.

Reads ``results/oof_<task>_<model>.csv`` (written by ``train_cv.py``) and emits

``tables/cv_metrics.tex``
    Pooled out-of-fold metrics with seeded bootstrap 95% CIs: BA, Sensitivity,
    Specificity, Precision, F1 (macro), AUROC, Loss. There is deliberately **no
    Recall column** — for a binary task it duplicates Sensitivity.
``tables/cv_foldwise.tex``
    The same metrics as mean +/- SD across the three folds, which is the honest
    dispersion estimate; pooling across folds hides it.
``tables/cv_pairwise_foldwise.tex``
    Proposed vs. each baseline, **within each fold** (each model scored on its
    own held-out fold, so the pairing is valid): DeLong test on the correlated
    AUCs and McNemar's test on paired correctness, with Holm-Bonferroni applied
    across every comparison in the table.

Also writes the same numbers as CSVs under ``--out-dir``.

Examples
--------
    python evaluate_cv.py --task you_chalk_brood
    python evaluate_cv.py --task you_foulbrood --models lgca_net resnet50 coatnet
    python evaluate_cv.py --task you_chalk_brood --mcnemar midp --n-boot 500
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from common import latex
from common.data import TASKS, task_label
from common.metrics import (
    METRIC_ORDER,
    bootstrap_ci,
    compute_metrics,
    delong_roc_test,
    holm_bonferroni,
    mcnemar_test,
)
from common.latex import tagged
from common.models import display_name


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--out-dir", default="./results")
    p.add_argument("--table-dir", default="./tables")
    p.add_argument("--task", default=None, choices=list(TASKS),
                   help="Restrict to one task (default: every task with OOF files).")
    p.add_argument("--models", nargs="+", default=None,
                   help="Restrict/order the models (default: whatever OOF files exist).")
    p.add_argument("--reference", default="lgca_net",
                   help="Model key used as the reference in the pairwise tests.")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--mcnemar", choices=["exact", "midp"], default="exact")
    p.add_argument("--run-tag", default="",
                   help="Suffix identifying this protocol (e.g. 'original', "
                        "'matched'). It selects the matching oof_*_<tag>.csv inputs "
                        "and suffixes every output: --run-tag original writes "
                        "tables/cv_metrics_original.tex. Empty = current names.")
    p.add_argument("--ablation-models", nargs="+",
                   default=["lgca_net", "lgca_net_no_cross_attn", "lgca_net_cnn_only"],
                   help="Models forming the ablation fragment (full model first).")
    p.add_argument("--dry-run", action="store_true",
                   help="200 bootstrap replicates, print only, write nothing.")
    return p


# --------------------------------------------------------------------------- #
def discover(results_dir: str, task: Optional[str], run_tag: str = "") -> Dict[str, List[str]]:
    """``task -> [model, ...]`` from the OOF filenames on disk.

    With a ``run_tag`` only ``oof_<task>_<model>_<tag>.csv`` files are considered
    and the tag is stripped from the model name; without one, files are taken as
    ``oof_<task>_<model>.csv``. Mixing tagged and untagged runs in a single
    results directory is therefore unambiguous only when the tag is supplied.
    """
    found: Dict[str, List[str]] = {}
    suffix = f"_{run_tag}" if run_tag else ""
    for path in sorted(glob.glob(os.path.join(results_dir, "oof_*.csv"))):
        base = os.path.basename(path)[len("oof_"):-len(".csv")]
        for t in TASKS:
            if not base.startswith(t + "_"):
                continue
            if task and t != task:
                break
            model = base[len(t) + 1:]
            if suffix:
                if not model.endswith(suffix):
                    break
                model = model[: -len(suffix)]
            found.setdefault(t, []).append(model)
            break
    return found


def oof_path(results_dir: str, task: str, model: str, run_tag: str = "") -> str:
    """Path of one model's OOF predictions for a task/run-tag combination."""
    return os.path.join(results_dir, tagged(f"oof_{task}_{model}", run_tag) + ".csv")


def load_oof(path: str) -> Dict[str, np.ndarray]:
    names, folds, y, p, pred, loss = [], [], [], [], [], []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            names.append(r["filename"])
            folds.append(int(r["fold"]))
            y.append(int(r["y_true"]))
            p.append(float(r["prob_pos"]))
            pred.append(int(r["pred"]))
            loss.append(float(r["loss"]) if r.get("loss") not in (None, "") else np.nan)
    return {"filename": np.array(names, dtype=object), "fold": np.array(folds),
            "y_true": np.array(y), "prob_pos": np.array(p, dtype=float),
            "pred": np.array(pred), "loss": np.array(loss, dtype=float)}


def align(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Index arrays aligning two OOF tables on filename (intersection, sorted)."""
    ia = {n: i for i, n in enumerate(a["filename"])}
    ib = {n: i for i, n in enumerate(b["filename"])}
    common = sorted(set(ia) & set(ib))
    return (np.array([ia[n] for n in common], dtype=int),
            np.array([ib[n] for n in common], dtype=int))


def name_of(model: str) -> str:
    """Manuscript-facing model name (resolves registry aliases)."""
    return display_name(model)


# --------------------------------------------------------------------------- #
def pooled_rows(task: str, data: Dict[str, Dict[str, np.ndarray]],
                n_boot: int, seed: int) -> List[Dict[str, object]]:
    rows = []
    for model, d in data.items():
        losses = d["loss"] if np.isfinite(d["loss"]).all() else None
        ci = bootstrap_ci(d["y_true"], d["prob_pos"], d["pred"], loss=losses,
                          n_boot=n_boot, seed=seed)
        row: Dict[str, object] = {"task": task, "model": model, "n": int(len(d["y_true"]))}
        for m in METRIC_ORDER:
            row[m], row[f"{m}_lo"], row[f"{m}_hi"] = ci[m]
        rows.append(row)
    return rows


def foldwise_rows(task: str, data: Dict[str, Dict[str, np.ndarray]]) -> List[Dict[str, object]]:
    rows = []
    for model, d in data.items():
        per_fold: Dict[str, List[float]] = {m: [] for m in METRIC_ORDER}
        for k in sorted(set(d["fold"].tolist())):
            s = d["fold"] == k
            losses = d["loss"][s] if np.isfinite(d["loss"][s]).all() else None
            m = compute_metrics(d["y_true"][s], d["prob_pos"][s], d["pred"][s], loss=losses)
            for key, v in m.items():
                per_fold[key].append(v)
        row: Dict[str, object] = {"task": task, "model": model,
                                  "n_folds": len(set(d["fold"].tolist()))}
        for key, vals in per_fold.items():
            arr = np.array(vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            row[f"{key}_mean"] = float(arr.mean()) if arr.size else float("nan")
            row[f"{key}_sd"] = float(arr.std(ddof=1)) if arr.size > 1 else float("nan")
        rows.append(row)
    return rows


def pairwise_foldwise_rows(task: str, data: Dict[str, Dict[str, np.ndarray]],
                           reference: str, mcnemar_method: str) -> List[Dict[str, object]]:
    """Per-fold DeLong and McNemar, reference vs. each other model."""
    if reference not in data:
        print(f"[warn] reference model '{reference}' has no OOF file for {task}; "
              "pairwise tests skipped")
        return []
    ref = data[reference]
    rows: List[Dict[str, object]] = []
    for model, d in data.items():
        if model == reference:
            continue
        ia, ib = align(ref, d)
        if ia.size == 0:
            print(f"[warn] {task}: no overlapping filenames between {reference} and {model}")
            continue
        folds = sorted(set(ref["fold"][ia].tolist()))
        for k in folds:
            s = ref["fold"][ia] == k
            ya = ref["y_true"][ia][s]
            pa = ref["prob_pos"][ia][s]
            pb = d["prob_pos"][ib][s]
            ra = ref["pred"][ia][s]
            rb = d["pred"][ib][s]
            dl = delong_roc_test(ya, pa, pb)
            mc = mcnemar_test(ya, ra, rb, method=mcnemar_method)
            rows.append({
                "task": task, "fold": k, "reference": reference, "model": model,
                "n": int(s.sum()),
                "auc_ref": dl["auc_a"], "auc_model": dl["auc_b"],
                "auc_diff": dl["diff"], "delong_z": dl["z"], "delong_p": dl["p"],
                "mcnemar_b": mc["b"], "mcnemar_c": mc["c"],
                "mcnemar_n_discordant": mc["n_discordant"], "mcnemar_p": mc["p"],
            })
    # Holm across every comparison in the table, per test family combined:
    # DeLong and McNemar p-values enter the same family (they are separate
    # hypotheses about the same set of comparisons).
    pvals = [r["delong_p"] for r in rows] + [r["mcnemar_p"] for r in rows]
    padj, rej = holm_bonferroni(pvals)
    n = len(rows)
    for i, r in enumerate(rows):
        r["delong_p_holm"] = float(padj[i])
        r["delong_sig"] = int(bool(rej[i]))
        r["mcnemar_p_holm"] = float(padj[n + i])
        r["mcnemar_sig"] = int(bool(rej[n + i]))
    return rows


# --------------------------------------------------------------------------- #
def metrics_table(rows: Sequence[Dict[str, object]], n_boot: int = 2000) -> str:
    header = ["Task", "Model", "$N$"] + [f"{m} (95\\% CI)" for m in METRIC_ORDER]
    out, mid, last = [], [], None
    for r in rows:
        if last is not None and r["task"] != last:
            mid.append(len(out) - 1)
        last = r["task"]
        cells = [latex.esc(task_label(str(r["task"]))), latex.esc(name_of(str(r["model"]))), str(r["n"])]
        for m in METRIC_ORDER:
            cells.append(latex.fmt_ci(r[m], r[f"{m}_lo"], r[f"{m}_hi"]))
        out.append(cells)
    return latex.tabular(
        header, out, align="llc" + "c" * len(METRIC_ORDER), midrules_after=mid,
        notes=["Pooled out-of-fold metrics over the 3 CV folds.",
               f"CIs are seeded stratified percentile bootstraps ({n_boot} replicates) "
               "over the pooled out-of-fold predictions.",
               "Sensitivity is the positive-class recall; no separate Recall column "
               "is reported because it would be identical.",
               "Loss is the mean unweighted cross-entropy in nats."],
    )


def foldwise_table(rows: Sequence[Dict[str, object]]) -> str:
    header = ["Task", "Model", "Folds"] + [f"{m} (mean $\\pm$ SD)" for m in METRIC_ORDER]
    out, mid, last = [], [], None
    for r in rows:
        if last is not None and r["task"] != last:
            mid.append(len(out) - 1)
        last = r["task"]
        cells = [latex.esc(task_label(str(r["task"]))), latex.esc(name_of(str(r["model"]))), str(r["n_folds"])]
        for m in METRIC_ORDER:
            cells.append(latex.fmt_pm(r[f"{m}_mean"], r[f"{m}_sd"]))
        out.append(cells)
    return latex.tabular(
        header, out, align="llc" + "c" * len(METRIC_ORDER), midrules_after=mid,
        notes=["Fold-wise dispersion: mean $\\pm$ SD of each metric across the 3 CV folds.",
               "With only 3 folds the SD is a coarse estimate and is reported for "
               "transparency, not for inference."],
    )


def pairwise_table(rows: Sequence[Dict[str, object]], method: str) -> str:
    header = ["Task", "Fold", "Comparison", "$\\Delta$AUC", "DeLong $z$",
              "DeLong $p$", "DeLong $p_{\\mathrm{Holm}}$",
              "McNemar $b/c$", "McNemar $p$", "McNemar $p_{\\mathrm{Holm}}$"]
    out, mid, last = [], [], None
    for r in rows:
        key = (r["task"], r["model"])
        if last is not None and key != last:
            mid.append(len(out) - 1)
        last = key
        dp = latex.fmt_p(r.get("delong_p_holm"))
        mp = latex.fmt_p(r.get("mcnemar_p_holm"))
        if r.get("delong_sig"):
            dp = latex.bold(dp)
        if r.get("mcnemar_sig"):
            mp = latex.bold(mp)
        out.append([
            latex.esc(task_label(str(r["task"]))), str(r["fold"]),
            f"{latex.esc(name_of(str(r['reference'])))} vs.\\ {latex.esc(name_of(str(r['model'])))}",
            latex.fmt(r["auc_diff"]), latex.fmt(r["delong_z"]),
            latex.fmt_p(r["delong_p"]), dp,
            f"{r['mcnemar_b']}/{r['mcnemar_c']}",
            latex.fmt_p(r["mcnemar_p"]), mp,
        ])
    return latex.tabular(
        header, out, align="llrccccccc", midrules_after=mid,
        notes=["Fold-wise paired comparisons: each model is scored on its own "
               "held-out fold, so the two score vectors in a row are paired by image.",
               "DeLong's test compares the two correlated AUCs; McNemar's test "
               f"({method}) compares paired correctness ($b$ = proposed correct / "
               "baseline wrong, $c$ = the reverse).",
               "Holm--Bonferroni is applied across every DeLong and McNemar "
               "p-value in this table. Bold marks significance at $\\alpha = 0.05$."],
    )


ABLATION_METRICS = ["BA", "F1", "AUROC"]


def ablation_table(
    pooled: Sequence[Dict[str, object]],
    foldwise: Sequence[Dict[str, object]],
    pairwise: Sequence[Dict[str, object]],
    models: Sequence[str],
    method: str,
) -> str:
    """Ablation fragment: pooled CIs + fold mean/SD + fold-wise DeLong/McNemar vs the full model.

    The fold-wise tests give one p-value per fold, so they are shown as
    slash-separated triples (fold 0/1/2) rather than exploding the row count.
    """
    full = models[0]
    by_task: Dict[str, List[str]] = {}
    for r in pooled:
        by_task.setdefault(str(r["task"]), [])
        if str(r["model"]) in models and str(r["model"]) not in by_task[str(r["task"])]:
            by_task[str(r["task"])].append(str(r["model"]))

    header = (["Task", "Variant"]
              + [f"{m} (95\% CI)" for m in ABLATION_METRICS]
              + [f"{m} (fold $\pm$ SD)" for m in ABLATION_METRICS]
              + ["DeLong $p_{\mathrm{Holm}}$ (per fold)",
                 "McNemar $p_{\mathrm{Holm}}$ (per fold)"])
    group = [("", 2), ("Pooled out-of-fold", len(ABLATION_METRICS)),
             ("Across folds", len(ABLATION_METRICS)), ("vs.\ full model", 2)]
    rows: List[List[str]] = []
    mid: List[int] = []
    tasks = list(by_task)
    for ti, task in enumerate(tasks):
        for model in models:
            pr = next((r for r in pooled if r["task"] == task and r["model"] == model), None)
            if pr is None:
                continue
            fr = next((r for r in foldwise if r["task"] == task and r["model"] == model), None)
            cells = [latex.esc(task_label(task)), latex.esc(name_of(model))]
            for m in ABLATION_METRICS:
                cells.append(latex.fmt_ci(pr[m], pr[f"{m}_lo"], pr[f"{m}_hi"]))
            for m in ABLATION_METRICS:
                cells.append(latex.fmt_pm(fr[f"{m}_mean"], fr[f"{m}_sd"]) if fr else "--")
            if model == full:
                cells += ["--", "--"]
            else:
                comp = sorted((r for r in pairwise
                               if r["task"] == task and r["model"] == model),
                              key=lambda r: int(r["fold"]))
                if comp:
                    dl = "/".join(latex.fmt_p(c.get("delong_p_holm")) for c in comp)
                    mc = "/".join(latex.fmt_p(c.get("mcnemar_p_holm")) for c in comp)
                    if any(c.get("delong_sig") for c in comp):
                        dl = latex.bold(dl)
                    if any(c.get("mcnemar_sig") for c in comp):
                        mc = latex.bold(mc)
                    cells += [dl, mc]
                else:
                    cells += ["--", "--"]
            rows.append(cells)
        if ti < len(tasks) - 1:
            mid.append(len(rows) - 1)

    return latex.tabular(
        header, rows, align="ll" + "c" * (2 * len(ABLATION_METRICS) + 2),
        midrules_after=mid, group_header=group,
        notes=["Ablation of the proposed architecture. The full model keeps both the "
               "transformer branch and cross-attention fusion; 'w/o cross-attn.' "
               "replaces the fusion with an additive residual; 'CNN only' drops the "
               "transformer branch entirely.",
               "Pooled CIs are seeded stratified percentile bootstraps over the "
               "out-of-fold predictions.",
               "The last two columns give the fold-wise p-values against the full "
               "model as fold 0/1/2, from DeLong's test on the correlated AUCs and "
               f"McNemar's {method} test on paired correctness.",
               "Holm--Bonferroni is applied across every fold-wise comparison in the "
               "parent cross-validation table, so these p-values are directly "
               "comparable with it. Bold marks at least one significant fold."],
    )


def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> str:
    if not rows:
        return path
    cols = list(rows[0])
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
    n_boot = 200 if args.dry_run else args.n_boot

    found = discover(args.results_dir, args.task, args.run_tag)
    if not found:
        print(f"[error] no oof_*.csv under {args.results_dir}; run train_cv.py first.")
        return 1

    all_pooled, all_fold, all_pair = [], [], []
    for task, models in found.items():
        if args.models:
            models = [m for m in args.models if m in models]
        data = {m: load_oof(oof_path(args.results_dir, task, m, args.run_tag))
                for m in models}
        if not data:
            continue
        print(f"[{task}] models: {', '.join(data)}")

        pr = pooled_rows(task, data, n_boot, args.seed)
        fr = foldwise_rows(task, data)
        pw = pairwise_foldwise_rows(task, data, args.reference, args.mcnemar)
        all_pooled.extend(pr)
        all_fold.extend(fr)
        all_pair.extend(pw)
        for r in pr:
            print(f"  {name_of(str(r['model'])):<24s} BA={r['BA']:.3f} "
                  f"({r['BA_lo']:.3f}-{r['BA_hi']:.3f})  AUROC={r['AUROC']:.3f}")

    if args.dry_run:
        print(metrics_table(all_pooled, n_boot))
        return 0

    tag = args.run_tag
    print("wrote", write_csv(os.path.join(args.out_dir, tagged("cv_metrics_pooled", tag) + ".csv"),
                             all_pooled))
    print("wrote", write_csv(os.path.join(args.out_dir, tagged("cv_metrics_foldwise", tag) + ".csv"),
                             all_fold))
    print("wrote", write_csv(os.path.join(args.out_dir, tagged("cv_pairwise_foldwise", tag) + ".csv"),
                             all_pair))
    print("wrote", latex.write_fragment(
        os.path.join(args.table_dir, tagged("cv_metrics", tag) + ".tex"),
        metrics_table(all_pooled, n_boot)))
    print("wrote", latex.write_fragment(
        os.path.join(args.table_dir, tagged("cv_foldwise", tag) + ".tex"),
        foldwise_table(all_fold)))
    if all_pair:
        print("wrote", latex.write_fragment(
            os.path.join(args.table_dir, tagged("cv_pairwise_foldwise", tag) + ".tex"),
            pairwise_table(all_pair, args.mcnemar)))

    # ---- ablation fragment -------------------------------------------------
    abl_models = [m for m in args.ablation_models
                  if any(str(r["model"]) == m for r in all_pooled)]
    if len(abl_models) >= 2:
        print("wrote", latex.write_fragment(
            os.path.join(args.table_dir, tagged("ablation", tag) + ".tex"),
            ablation_table(all_pooled, all_fold, all_pair, abl_models, args.mcnemar)))
    else:
        present = sorted({str(r["model"]) for r in all_pooled})
        print(f"[warn] ablation fragment skipped: need at least 2 of "
              f"{args.ablation_models}, found {present}. Run train_cv.py with "
              "--models lgca_net lgca_net_no_cross_attn lgca_net_cnn_only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
