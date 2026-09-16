#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate every analysis figure as a vector PDF.

Model-agnostic; bar widths and label sizes adapt to the number of models, so a
run with 13 baselines produces the same figures as a run with 9.  Every figure
that plots a mean carries a fold-wise standard-deviation error bar.

    python make_manuscript_figures.py --results-dir ./results --out-dir ./results \
        --figures-dir ./manuscript
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402
import pandas as pd                    # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_io import TASKS, base_parser, ensure_dir, read_features  # noqa: E402
from model_registry import (ABLATION_ORDER, REFERENCE_MODEL, common_models,  # noqa: E402
                            short_name)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "axes.labelsize": 8.5,
    "axes.titlesize": 9, "legend.fontsize": 7.5, "figure.dpi": 400,
    "savefig.dpi": 400, "axes.spines.top": False, "axes.spines.right": False})

C_NORM, C_DIS, C_ORIG, C_MATCH = "#4C72B0", "#C44E52", "#8C8C8C", "#55A868"
TRIVIAL_SHORT = {"threshold_L": "$L^*$ threshold",
                 "logreg": "Logistic reg.\n(global stats)",
                 "gbm": "GBM\n(global stats)"}


def _xticks(ax, keys, n):
    ax.set_xticks(np.arange(len(keys)))
    size = 6.5 if n <= 10 else (5.6 if n <= 14 else 4.8)
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=size)


def main() -> int:
    ap = base_parser(__doc__)
    a = ap.parse_args()
    F = ensure_dir(a.figures_dir) + os.sep
    report = json.load(open(os.path.join(a.out_dir, "matched_report.json"), encoding="utf-8"))
    models = common_models(a.results_dir, a.tasks, a.run_tag)
    tag = f"_{a.run_tag}" if a.run_tag else ""
    fold = pd.read_csv(os.path.join(a.results_dir, f"cv_metrics_foldwise{tag}.csv"))
    fold["m"] = fold.model.str.replace(tag, "", regex=False) if tag else fold.model

    # ------------------------------------------------------------ audit figure
    fig, axes = plt.subplots(len(a.tasks), 3, figsize=(7.1, 2.1 * len(a.tasks)), squeeze=False)
    for i, task in enumerate(a.tasks):
        f = read_features(a.results_dir, task)
        m = pd.read_csv(os.path.join(a.out_dir, f"matched_{task}.csv"))
        bins = np.linspace(5, 80, 46)
        for j, (data, title) in enumerate([(f, "before matching"), (m, "after matching")]):
            ax = axes[i][j]
            ax.hist(data.loc[data.label == 0, "lab_L_256"], bins=bins, color=C_NORM,
                    alpha=.75, lw=0, label="Normal")
            ax.hist(data.loc[data.label == 1, "lab_L_256"], bins=bins, color=C_DIS,
                    alpha=.75, lw=0, label="Disease")
            ax.set_title(f"({chr(97 + i * 3 + j)}) {TASKS[task]}: {title}", loc="left")
            ax.set_xlabel("CIELAB $L^*$"); ax.set_ylabel("Images")
            if i == 0 and j == 0:
                ax.legend(frameon=False, loc="upper right")
        ax = axes[i][2]
        ct = pd.crosstab(f.site, f.label).reindex(sorted(f.site.unique()))
        x = np.arange(len(ct))
        ax.bar(x - .2, ct.get(0, 0), .4, color=C_NORM)
        ax.bar(x + .2, ct.get(1, 0), .4, color=C_DIS)
        ax.set_xticks(x); ax.set_xticklabels(ct.index, rotation=90, fontsize=5.5)
        ax.set_title(f"({chr(99 + i * 3)}) {TASKS[task]}: images per site", loc="left")
        ax.set_ylabel("Images")
    fig.tight_layout(pad=.6); fig.savefig(F + "fig_audit.pdf"); plt.close(fig)

    # ------------------------------------------------- performance comparison
    triv_keys = list(TRIVIAL_SHORT)
    keys = models + triv_keys
    fig, axes = plt.subplots(len(a.tasks), 1, figsize=(7.1, 2.7 * len(a.tasks)),
                             sharex=True, squeeze=False)
    for i, task in enumerate(a.tasks):
        ax = axes[i][0]; r = report[task]; xs = np.arange(len(keys))
        om, os_, km, ks = [], [], [], []
        for k in keys:
            if k in models:
                fr = fold[(fold.task == task) & (fold.m == k)].iloc[0]
                om.append(fr.BA_mean); os_.append(fr.BA_sd)
                km.append(r["deep"][k]["BA_foldmean"]); ks.append(r["deep"][k]["BA_foldsd"])
            else:
                o = r["trivial_original"][k]; mm = r["trivial"][k]
                om.append(o["BA_foldmean"]); os_.append(o["BA_foldsd"])
                km.append(mm["BA_foldmean"]); ks.append(mm["BA_foldsd"])
        ek = dict(lw=.8)
        ax.bar(xs - .2, om, .4, yerr=os_, color=C_ORIG, ecolor="0.25", capsize=2,
               error_kw=ek, label="Original subset")
        ax.bar(xs + .2, km, .4, yerr=ks, color=C_MATCH, ecolor="0.25", capsize=2,
               error_kw=ek, label="Acquisition-matched subset")
        ax.axhline(.5, color="k", ls=":", lw=.8)
        ax.axvline(len(models) - .5, color="0.6", lw=.8, ls="--")
        ax.axvspan(-.5, .5, color="#FFD966", alpha=.20, zorder=0)
        ax.set_ylim(.3, 1.04); ax.set_ylabel("Balanced accuracy")
        ax.set_title(f"({chr(97 + i)}) {TASKS[task]}", loc="left")
        if i == 0:
            ax.legend(frameon=False, ncol=2, loc="lower left")
    _xticks(axes[-1][0], [short_name(k) if k in models else TRIVIAL_SHORT[k] for k in keys],
            len(keys))
    fig.tight_layout(pad=.6); fig.savefig(F + "fig_performance.pdf"); plt.close(fig)

    # ------------------------------------------------------------- ablation
    abl = [m for m in ABLATION_ORDER if m in models]
    if len(abl) >= 2:
        labels = {"lgca_net_cnn_only": "CNN branch only",
                  "lgca_net_no_cross_attn": "+ Transformer branch\n(position-wise addition)",
                  "lgca_net": "+ cross-attention fusion\n(LGCA-Net)"}
        fig, axes = plt.subplots(1, len(a.tasks), figsize=(7.1, 2.7), sharey=True, squeeze=False)
        for i, task in enumerate(a.tasks):
            ax = axes[0][i]; r = report[task]; xs = np.arange(len(abl))
            om = [fold[(fold.task == task) & (fold.m == k)].iloc[0].BA_mean for k in abl]
            os_ = [fold[(fold.task == task) & (fold.m == k)].iloc[0].BA_sd for k in abl]
            km = [r["deep"][k]["BA_foldmean"] for k in abl]
            ks = [r["deep"][k]["BA_foldsd"] for k in abl]
            ax.bar(xs - .2, om, .4, yerr=os_, color=C_ORIG, ecolor="0.25", capsize=2,
                   error_kw=dict(lw=.8), label="Original subset")
            ax.bar(xs + .2, km, .4, yerr=ks, color=C_MATCH, ecolor="0.25", capsize=2,
                   error_kw=dict(lw=.8), label="Matched subset")
            ax.set_xticks(xs); ax.set_xticklabels([labels.get(k, short_name(k)) for k in abl],
                                                  fontsize=6.5)
            ax.set_title(f"({chr(97 + i)}) {TASKS[task]}", loc="left"); ax.set_ylim(.4, 1.02)
            if i == 0:
                ax.set_ylabel("Balanced accuracy"); ax.legend(frameon=False, loc="lower left")
        fig.tight_layout(pad=.6); fig.savefig(F + "fig_ablation.pdf"); plt.close(fig)

    # -------------------------------------------------- confound reliance
    cpath = os.path.join(a.out_dir, "confound_reliance.csv")
    if os.path.isfile(cpath):
        c = pd.read_csv(cpath)
        fig, axes = plt.subplots(1, len(a.tasks), figsize=(7.1, 3.0), sharey=True, squeeze=False)
        for i, task in enumerate(a.tasks):
            ax = axes[0][i]; xs = np.arange(len(models))
            for j, (scope, col, lab) in enumerate(
                    [("original", C_ORIG, "Original subset"),
                     ("matched", C_MATCH, "Acquisition-matched subset")]):
                g = c[(c.task == task) & (c.scope == scope)].set_index("model")
                if g.empty:
                    continue
                ax.bar(xs + (j - .5) * .4, [g.loc[m].surrogate_ba for m in models], .4,
                       color=col, label=lab if i == 0 else None)
                ref = g.loc["__reference_label__"].surrogate_ba
                ax.hlines(ref, -.62, len(models) - .38, color=col, ls=(0, (4, 2)), lw=1.1, zorder=5)
                ax.annotate(f"disease-label reference\n({lab.split()[0].lower()})",
                            xy=(len(models) - .45, ref),
                            xytext=(3, 2 if scope == "original" else -11),
                            textcoords="offset points", ha="right", va="bottom",
                            fontsize=5.6, color=col, zorder=6)
            _xticks(ax, [short_name(m) for m in models], len(models))
            ax.set_title(f"({chr(97 + i)}) {TASKS[task]}", loc="left")
            ax.set_ylim(.5, 1.06); ax.set_xlim(-.62, len(models) - .38)
            ax.axvspan(-.5, .5, color="#FFD966", alpha=.20, zorder=0)
            if i == 0:
                ax.set_ylabel("Balanced accuracy of the surrogate")
                ax.legend(frameon=False, loc="lower left", fontsize=6.5)
        fig.tight_layout(pad=.6); fig.savefig(F + "fig_confound.pdf"); plt.close(fig)

    print(f"figures written to {F}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
