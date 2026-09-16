#!/usr/bin/env python3
"""Across-seed statistics: Welch t-tests, Hedges' g, Holm correction, and a forest plot.

Input (in order of preference, per task)
----------------------------------------
1. **Raw per-seed CSV** ``results/seeds_<task>.csv`` written by ``train_seeds.py``
   with columns ``model, seed, BA, sensitivity, specificity, precision, F1,
   AUROC, loss``. Preferred: the SDs are exact.
2. **Legacy summary CSV** ``seed_table_<task>_95CI.csv`` with the header

       Model,Accuracy (95% CI),BA (95% CI),Recall (95% CI),Precision (95% CI),F1 (95% CI),AUROC (95% CI),Loss (95% CI)

   and cells like ``0.9280 (0.8729-0.9831)``. These CIs were produced as
   ``mean +/- t_{0.975,4} * SD / sqrt(5)`` over ``n = 5`` seeds, so the SD is
   recovered as ``SD = half_width * sqrt(5) / t_{0.975,4}``
   (``t_{0.975,4} = 2.7764``). The row named ``Proposed`` is the reference.

Outputs
-------
``tables/seed_table.tex``     per-model mean (95% CI) for BA / F1 / AUROC / Loss, both tasks
``tables/seed_pairwise.tex``  Proposed vs. each baseline: difference, Welch t, raw p,
                              Holm-adjusted p, Hedges' g, CI overlap
``figures/seed_forest.png``   400 dpi, two panels ((a) chalkbrood, (b) foulbrood),
                              three metric columns (BA, F1, AUROC), 95% CI bars
``results/seed_pairwise_<task>.csv``  the same pairwise numbers as CSV

Holm-Bonferroni is applied across **all** comparisons in the table
(models x metrics x tasks) unless ``--holm-scope`` narrows the family.

Examples
--------
    python seed_stats.py --summary-csv you_chalk_brood=/path/a.csv you_foulbrood=/path/b.csv
    python seed_stats.py --results-dir ./results          # uses raw seeds_<task>.csv
    python seed_stats.py --summary-csv ... --holm-scope task
    python seed_stats.py --results-dir results/matched --run-tag matched
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from common import latex
from common.data import TASKS, TASK_LABEL
from common.latex import tagged
from common.metrics import ci_overlap, hedges_g, holm_bonferroni, sd_from_ci, welch_t_from_summary

#: metrics compared between the proposed model and each baseline
COMPARE_METRICS = ["BA", "F1", "AUROC"]
#: metrics shown in the descriptive seed table
TABLE_METRICS = ["BA", "F1", "AUROC", "Loss"]

REFERENCE_MODEL = "Proposed"
DEFAULT_N_SEEDS = 5

#: legacy header column -> canonical metric name
LEGACY_COLUMNS = OrderedDict([
    ("Accuracy (95% CI)", "Accuracy"),
    ("BA (95% CI)", "BA"),
    ("Recall (95% CI)", "Recall"),
    ("Precision (95% CI)", "Precision"),
    ("F1 (95% CI)", "F1"),
    ("AUROC (95% CI)", "AUROC"),
    ("Loss (95% CI)", "Loss"),
])

#: ``0.9280 (0.8729-0.9831)`` and ``3.7488 (-5.0747-12.5723)``
_CI_RE = re.compile(
    r"^\s*(?P<mean>-?\d+(?:\.\d+)?)\s*"
    r"\(\s*(?P<lo>-?\d+(?:\.\d+)?)\s*-\s*(?P<hi>-?\d+(?:\.\d+)?)\s*\)\s*$"
)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="./results",
                   help="Searched for raw seeds_<task>.csv files.")
    p.add_argument("--summary-csv", nargs="*", default=[],
                   help="Fallback legacy summary CSVs as task=path (repeatable).")
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument("--table-dir", default="./tables")
    p.add_argument("--figure-dir", default="./figures")
    p.add_argument("--out-dir", default="./results")
    p.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS,
                   help="Seed count assumed when recovering SDs from legacy CIs.")
    p.add_argument("--reference", default=REFERENCE_MODEL)
    p.add_argument("--holm-scope", choices=["all", "task", "metric"], default="all",
                   help="Family over which Holm-Bonferroni is applied (default: all comparisons).")
    p.add_argument("--run-tag", default="",
                   help="Suffix identifying this protocol (e.g. 'matched'). It "
                        "selects results/seeds_<task>_<tag>.csv when present and "
                        "suffixes the outputs: --run-tag matched writes "
                        "tables/seed_table_matched.tex and seed_pairwise_matched.tex. "
                        "Empty = current names.")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--dpi", type=int, default=400)
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the parsed summaries and the pairwise table; write nothing.")
    return p


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_ci_cell(cell: str) -> Tuple[float, float, float]:
    """``'0.9280 (0.8729-0.9831)'`` -> ``(mean, lo, hi)``; NaNs on failure."""
    m = _CI_RE.match(str(cell))
    if m is None:
        return float("nan"), float("nan"), float("nan")
    return float(m.group("mean")), float(m.group("lo")), float(m.group("hi"))


def load_legacy_summary(path: str, n_seeds: int) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Legacy ``mean (lo-hi)`` CSV -> ``{model: {metric: {mean, lo, hi, sd, n}}}``.

    ``utf-8-sig`` handles the BOM the original export writes.
    """
    out: Dict[str, Dict[str, Dict[str, float]]] = OrderedDict()
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fields = [f.strip() for f in (reader.fieldnames or [])]
        if "Model" not in fields:
            raise ValueError(f"{path}: no 'Model' column (found {fields})")
        for raw in reader:
            row = { (k.strip() if k else k): v for k, v in raw.items() }
            model = (row.get("Model") or "").strip()
            if not model:
                continue
            per_metric: Dict[str, Dict[str, float]] = {}
            for col, metric in LEGACY_COLUMNS.items():
                if col not in row:
                    continue
                mean, lo, hi = parse_ci_cell(row[col])
                per_metric[metric] = {
                    "mean": mean, "lo": lo, "hi": hi,
                    "sd": sd_from_ci(mean, lo, hi, n=n_seeds),
                    "n": float(n_seeds),
                    "source": "legacy_ci",
                }
            out[model] = per_metric
    return out


def load_raw_seeds(path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Raw per-seed CSV -> the same nested summary structure (exact SDs)."""
    alias = {"sensitivity": "Sensitivity", "specificity": "Specificity",
             "precision": "Precision", "loss": "Loss", "ba": "BA",
             "f1": "F1", "auroc": "AUROC", "accuracy": "Accuracy"}
    values: Dict[str, Dict[str, List[float]]] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            model = (row.get("model") or row.get("Model") or "").strip()
            if not model:
                continue
            bucket = values.setdefault(model, {})
            for k, v in row.items():
                if k is None:
                    continue
                key = alias.get(k.strip().lower())
                if key is None:
                    continue
                try:
                    bucket.setdefault(key, []).append(float(v))
                except (TypeError, ValueError):
                    pass

    out: Dict[str, Dict[str, Dict[str, float]]] = OrderedDict()
    for model, per_metric in values.items():
        summ: Dict[str, Dict[str, float]] = {}
        for metric, vals in per_metric.items():
            arr = np.asarray([v for v in vals if math.isfinite(v)], dtype=float)
            if arr.size == 0:
                continue
            n = arr.size
            mean = float(arr.mean())
            sd = float(arr.std(ddof=1)) if n > 1 else 0.0
            half = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n) if n > 1 else 0.0
            summ[metric] = {"mean": mean, "lo": mean - half, "hi": mean + half,
                            "sd": sd, "n": float(n), "source": "raw",
                            "values": arr.tolist()}
        out[model] = summ
    return out


def load_task(task: str, results_dir: str, summary_map: Dict[str, str], n_seeds: int,
              run_tag: str = ""):
    """Prefer the raw per-seed CSV; fall back to the legacy summary.

    With a ``run_tag`` the tagged file ``seeds_<task>_<tag>.csv`` is preferred, so
    a matched-protocol run and the original run can share a results directory.
    """
    candidates = []
    if run_tag:
        candidates.append(os.path.join(results_dir, tagged(f"seeds_{task}", run_tag) + ".csv"))
    candidates.append(os.path.join(results_dir, f"seeds_{task}.csv"))
    for raw in candidates:
        if os.path.exists(raw):
            print(f"[{task}] using raw per-seed results: {raw}")
            return load_raw_seeds(raw), "raw"
    path = summary_map.get(task)
    if path and os.path.exists(path):
        print(f"[{task}] using legacy summary CSV: {path} (SDs recovered from the 95% CIs, n={n_seeds})")
        return load_legacy_summary(path, n_seeds), "legacy"
    return None, None


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def pairwise_rows(
    task: str,
    summary: Dict[str, Dict[str, Dict[str, float]]],
    reference: str,
    metrics: Sequence[str] = COMPARE_METRICS,
) -> List[Dict[str, object]]:
    """Proposed-vs-baseline Welch t, Hedges' g and CI overlap for every metric."""
    if reference not in summary:
        raise KeyError(f"[{task}] reference model '{reference}' not found "
                       f"(available: {', '.join(summary)})")
    ref = summary[reference]
    rows: List[Dict[str, object]] = []
    for model, per_metric in summary.items():
        if model == reference:
            continue
        for metric in metrics:
            if metric not in ref or metric not in per_metric:
                continue
            r, b = ref[metric], per_metric[metric]
            w = welch_t_from_summary(r["mean"], r["sd"], int(r["n"]),
                                     b["mean"], b["sd"], int(b["n"]))
            g = hedges_g(r["mean"], r["sd"], int(r["n"]),
                         b["mean"], b["sd"], int(b["n"]))
            rows.append({
                "task": task, "metric": metric, "reference": reference, "model": model,
                "mean_ref": r["mean"], "sd_ref": r["sd"], "n_ref": int(r["n"]),
                "mean_model": b["mean"], "sd_model": b["sd"], "n_model": int(b["n"]),
                "diff": w["diff"], "t": w["t"], "df": w["df"], "p_raw": w["p"],
                "hedges_g": g,
                "ci_overlap": int(ci_overlap(r["lo"], r["hi"], b["lo"], b["hi"])),
                "ref_lo": r["lo"], "ref_hi": r["hi"],
                "model_lo": b["lo"], "model_hi": b["hi"],
            })
    return rows


def apply_holm(rows: List[Dict[str, object]], scope: str, alpha: float) -> None:
    """Add ``p_holm``/``significant``/``holm_family`` in place."""
    if scope == "all":
        families = {"all": list(range(len(rows)))}
    elif scope == "task":
        families = {}
        for i, r in enumerate(rows):
            families.setdefault(str(r["task"]), []).append(i)
    else:  # metric
        families = {}
        for i, r in enumerate(rows):
            families.setdefault(f"{r['task']}|{r['metric']}", []).append(i)

    for fam, idxs in families.items():
        padj, rej = holm_bonferroni([rows[i]["p_raw"] for i in idxs], alpha=alpha)
        for j, i in enumerate(idxs):
            rows[i]["p_holm"] = float(padj[j])
            rows[i]["significant"] = int(bool(rej[j]))
            rows[i]["holm_family"] = fam
            rows[i]["holm_family_size"] = len(idxs)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def seed_table(summaries: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
               reference: str, n_seeds_note: str) -> str:
    header = ["Task", "Model"] + [f"{m} (mean, 95\\% CI)" for m in TABLE_METRICS]
    rows: List[List[str]] = []
    mid: List[int] = []
    for ti, (task, summary) in enumerate(summaries.items()):
        for model, per_metric in summary.items():
            name = latex.esc(model)
            if model == reference:
                name = latex.bold(name)
            cells = [latex.esc(TASK_LABEL.get(task, task)), name]
            for metric in TABLE_METRICS:
                d = per_metric.get(metric)
                cells.append("--" if d is None else latex.fmt_ci(d["mean"], d["lo"], d["hi"]))
            rows.append(cells)
        if ti < len(summaries) - 1:
            mid.append(len(rows) - 1)
    return latex.tabular(
        header, rows, align="ll" + "c" * len(TABLE_METRICS), midrules_after=mid,
        notes=[f"Across-seed results on the fixed held-out split. {n_seeds_note}",
               "CI = mean $\\pm t_{0.975,\\,n-1}\\,\\mathrm{SD}/\\sqrt{n}$.",
               "BA = balanced accuracy; F1 is macro-averaged; Loss is the mean "
               "unweighted cross-entropy (nats)."],
    )


def pairwise_table(rows: Sequence[Dict[str, object]], reference: str, scope: str) -> str:
    header = ["Task", "Metric", "Comparison", "$\\Delta$ (Proposed $-$ baseline)",
              "$t$", "df", "$p$", "$p_{\\mathrm{Holm}}$", "Hedges' $g$", "CIs overlap"]
    out: List[List[str]] = []
    mid: List[int] = []
    prev = None
    tasks = list(dict.fromkeys(str(r["task"]) for r in rows))
    rows = sorted(
        rows,
        key=lambda r: (tasks.index(str(r["task"])),
                       COMPARE_METRICS.index(str(r["metric"]))
                       if str(r["metric"]) in COMPARE_METRICS else 99,
                       str(r["model"])),
    )
    for r in rows:
        key = (r["task"], r["metric"])
        if prev is not None and key != prev:
            mid.append(len(out) - 1)
        prev = key
        sig = r.get("significant", 0)
        padj = latex.fmt_p(r.get("p_holm"))
        if sig:
            padj = latex.bold(padj)
        out.append([
            latex.esc(TASK_LABEL.get(str(r["task"]), str(r["task"]))),
            latex.esc(r["metric"]),
            f"{latex.esc(reference)} vs.\\ {latex.esc(r['model'])}",
            latex.fmt(r["diff"]),
            latex.fmt(r["t"]), latex.fmt(r["df"], 1),
            latex.fmt_p(r["p_raw"]), padj,
            latex.fmt(r["hedges_g"]),
            "yes" if r["ci_overlap"] else "no",
        ])
    fam = {"all": "all comparisons in this table",
           "task": "all comparisons within each task",
           "metric": "all comparisons within each task and metric"}[scope]
    return latex.tabular(
        header, out, align="lllccccccc", midrules_after=mid,
        notes=["Welch's unequal-variance t-tests across seeds, proposed model vs. each baseline.",
               f"Holm--Bonferroni correction applied over {fam}.",
               "Bold $p_{Holm}$ marks significance at $\\alpha = 0.05$. "
               "Positive $\\Delta$ favours the proposed model."],
    )


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def forest_plot(summaries, reference: str, out_png: str, dpi: int = 400) -> str:
    """Two-panel forest plot: rows = models, columns = BA / F1 / AUROC, 95% CI bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = list(summaries)
    metrics = COMPARE_METRICS
    n_rows = len(tasks)
    height = 1.2 + 0.42 * max(len(s) for s in summaries.values()) * n_rows
    fig, axes = plt.subplots(n_rows, len(metrics),
                             figsize=(3.4 * len(metrics), height),
                             squeeze=False)
    panel_letters = "abcdefgh"

    for ti, task in enumerate(tasks):
        summary = summaries[task]
        models = list(summary)
        # reference last so it plots at the top of an inverted axis
        models = [m for m in models if m != reference] + [m for m in models if m == reference]
        ypos = np.arange(len(models))
        for mi, metric in enumerate(metrics):
            ax = axes[ti][mi]
            for y, model in zip(ypos, models):
                d = summary[model].get(metric)
                if d is None or not math.isfinite(d["mean"]):
                    continue
                is_ref = model == reference
                color = "#1f4e79" if is_ref else "#7f7f7f"
                lo, hi = d["lo"], d["hi"]
                if math.isfinite(lo) and math.isfinite(hi):
                    ax.plot([lo, hi], [y, y], color=color,
                            lw=2.2 if is_ref else 1.4, solid_capstyle="butt", zorder=2)
                    for edge in (lo, hi):
                        ax.plot([edge, edge], [y - 0.16, y + 0.16], color=color,
                                lw=2.2 if is_ref else 1.4, zorder=2)
                ax.plot([d["mean"]], [y], marker="D" if is_ref else "o",
                        ms=6.5 if is_ref else 5.0, color=color, zorder=3)
            ref_mean = summary.get(reference, {}).get(metric, {}).get("mean")
            if ref_mean is not None and math.isfinite(ref_mean):
                ax.axvline(ref_mean, color="#1f4e79", ls="--", lw=0.8, alpha=0.6, zorder=1)

            ax.set_yticks(ypos)
            ax.set_yticklabels(models if mi == 0 else [""] * len(models), fontsize=8)
            ax.invert_yaxis()
            ax.set_ylim(len(models) - 0.4, -0.6)
            ax.grid(axis="x", ls=":", lw=0.5, alpha=0.6)
            ax.tick_params(axis="x", labelsize=8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if ti == 0:
                ax.set_title(metric, fontsize=10)
            if ti == n_rows - 1:
                ax.set_xlabel("value (95% CI)", fontsize=9)
            if mi == 0:
                ax.set_ylabel(f"({panel_letters[ti]}) {TASK_LABEL.get(task, task)}",
                              fontsize=10, labelpad=8)

    fig.suptitle("Across-seed performance on the fixed held-out split (n = 5 seeds)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- #
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


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    summary_map: Dict[str, str] = {}
    for item in args.summary_csv:
        if "=" not in item:
            raise SystemExit(f"--summary-csv entries must be task=path, got '{item}'")
        t, p = item.split("=", 1)
        summary_map[t.strip()] = p.strip()

    summaries: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = OrderedDict()
    sources = set()
    for task in args.tasks:
        summary, src = load_task(task, args.results_dir, summary_map, args.n_seeds,
                                 args.run_tag)
        if summary is None:
            print(f"[warn] no seed results for '{task}' "
                  f"(looked for results/seeds_{task}.csv and --summary-csv)")
            continue
        summaries[task] = summary
        sources.add(src)

    if not summaries:
        print("[error] nothing to do: no seed results found.")
        return 1

    all_rows: List[Dict[str, object]] = []
    for task, summary in summaries.items():
        all_rows.extend(pairwise_rows(task, summary, args.reference))
    apply_holm(all_rows, args.holm_scope, args.alpha)

    tcrit = float(stats.t.ppf(0.975, args.n_seeds - 1))
    note = ("SDs recovered from the reported 95% CIs "
            f"(n = {args.n_seeds}, $t_{{0.975,{args.n_seeds - 1}}} = {tcrit:.4f}$)."
            if "legacy" in sources else
            f"Computed from raw per-seed results (n = {args.n_seeds}).")

    tbl = seed_table(summaries, args.reference, note)
    pw = pairwise_table(all_rows, args.reference, args.holm_scope)

    if args.dry_run:
        print(tbl)
        print(pw)
        return 0

    tag = args.run_tag
    print("wrote", latex.write_fragment(
        os.path.join(args.table_dir, tagged("seed_table", tag) + ".tex"), tbl))
    print("wrote", latex.write_fragment(
        os.path.join(args.table_dir, tagged("seed_pairwise", tag) + ".tex"), pw))
    for task in summaries:
        rows = [r for r in all_rows if r["task"] == task]
        print("wrote", write_csv(
            os.path.join(args.out_dir, tagged(f"seed_pairwise_{task}", tag) + ".csv"), rows))
    if not args.no_figure:
        print("wrote", forest_plot(summaries, args.reference,
                                   os.path.join(args.figure_dir,
                                                tagged("seed_forest", tag) + ".png"),
                                   dpi=args.dpi))

    n_sig = sum(1 for r in all_rows if r.get("significant"))
    print(f"\n{len(all_rows)} comparisons, {n_sig} significant after "
          f"Holm correction (alpha={args.alpha}, scope={args.holm_scope}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
