#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate every LaTeX table fragment used by the manuscript.

Model-agnostic: the rows come from whatever models have out-of-fold predictions,
so adding Conformer / Mobile-Former / LSNet-T / MambaVision changes every table
without editing this file.  Parameter counts and the pretraining column are read
from ``results/model_provenance.json`` (written by verify_external_models.py);
a model with no measured entry prints ``--`` rather than a quoted figure.

    python make_manuscript_tables.py --results-dir ./results --tables-dir ./tables
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_io import BALANCE_FEATS, TASKS, base_parser, ensure_dir  # noqa: E402
from model_registry import (ABLATION_ORDER, REFERENCE_MODEL, common_models,  # noqa: E402
                            display_name, load_provenance, params_millions,
                            pretraining_label)

TRIVIAL_NAMES = {"threshold_L": r"Single threshold on mean $L^*$",
                 "logreg": "Logistic regression on six global statistics",
                 "gbm": "Gradient boosting on six global statistics"}
FEAT_LABEL = {"lab_L_256": r"CIELAB $L^*$", "lab_a_256": r"CIELAB $a^*$",
              "lab_b_256": r"CIELAB $b^*$", "rgb_r_256": "Mean red",
              "rgb_g_256": "Mean green", "rgb_b_256": "Mean blue",
              "file_size_bytes": "JPEG file size",
              "otsu_largest_area_frac": "Largest Otsu blob area"}
ABL_LABEL = {"lgca_net_cnn_only": "CNN branch only (no Transformer path)",
             "lgca_net_no_cross_attn": "CNN + Transformer, position-wise additive fusion",
             "lgca_net": "CNN + Transformer, cross-attention fusion (LGCA-Net)"}


def wrap(body: str, colspec: str, header: str) -> str:
    return (f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header}\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}\n")


def block(label: str, ncol: int) -> str:
    return f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{{label}}}}} \\\\"


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--provenance", default=None)
    a = ap.parse_args()
    T = ensure_dir(a.tables_dir) + os.sep
    R = a.results_dir
    prov = load_provenance(a.provenance or os.path.join(R, "model_provenance.json"))
    models = common_models(R, a.tasks, a.run_tag)
    report = json.load(open(os.path.join(a.out_dir, "matched_report.json"), encoding="utf-8"))
    tag = f"_{a.run_tag}" if a.run_tag else ""

    def w(name, text):
        open(T + name, "w", encoding="utf-8").write(text)
        print(f"  wrote {T + name}")

    # ---------------------------------------------------- performance, original
    pool = pd.read_csv(os.path.join(R, f"cv_metrics_pooled{tag}.csv"))
    fold = pd.read_csv(os.path.join(R, f"cv_metrics_foldwise{tag}.csv"))
    for d in (pool, fold):
        d["m"] = d.model.str.replace(tag, "", regex=False) if tag else d.model
    lines = []
    for task in a.tasks:
        lines.append(block(f"{TASKS[task]} ($n={int(pool[pool.task == task].n.iloc[0]):,}$)".replace(",", "{,}"), 9))
        for mod in models:
            p = pool[(pool.task == task) & (pool.m == mod)]
            f = fold[(fold.task == task) & (fold.m == mod)]
            if p.empty or f.empty:
                continue
            p, f = p.iloc[0], f.iloc[0]
            pm = params_millions(mod, prov)
            lines.append(
                f"\\quad {display_name(mod, latex=True)} & {f'{pm:.2f}' if pm is not None else '--'} & "
                f"{pretraining_label(mod, prov)} & {p.BA:.3f} ({p.BA_lo:.3f}--{p.BA_hi:.3f}) & "
                f"{f.BA_mean:.3f} ({f.BA_sd:.3f}) & {p.Sensitivity:.3f} & {p.Specificity:.3f} & "
                f"{p.F1:.3f} & {p.AUROC:.3f} ({p.AUROC_lo:.3f}--{p.AUROC_hi:.3f}) \\\\")
    w("cv_metrics_original.tex", wrap("\n".join(lines), "@{}lllccccc c@{}",
        "Model & Param. & Pre- & BA & BA fold & Sens. & Spec. & F$_1$ & AUROC \\\\\n"
        " & (M) & training & (95\\% CI) & mean (SD) & & & & (95\\% CI) \\\\"))

    # ----------------------------------------------------- performance, matched
    lines = []
    for task in a.tasks:
        r = report[task]
        lines.append(block(f"{TASKS[task]} ($n={r['after']['n']}$, {r['after']['n_pairs']} matched pairs)", 6))
        for mod in models:
            if mod not in r["ci"]:
                continue
            c, d = r["ci"][mod], r["deep"][mod]
            lines.append(
                f"\\quad {display_name(mod, latex=True)} & {c['BA']:.3f} ({c['BA_lo']:.3f}--{c['BA_hi']:.3f}) & "
                f"{d['BA_foldmean']:.3f} ({d['BA_foldsd']:.3f}) & {d['F1']:.3f} & "
                f"{c['AUROC']:.3f} ({c['AUROC_lo']:.3f}--{c['AUROC_hi']:.3f}) & "
                f"{d['AUROC_foldmean']:.3f} ({d['AUROC_foldsd']:.3f}) \\\\")
        for k, name in TRIVIAL_NAMES.items():
            v = r["trivial"][k]
            lines.append(f"\\quad \\textit{{{name}}} & {v['BA']:.3f} & "
                         f"{v['BA_foldmean']:.3f} ({v['BA_foldsd']:.3f}) & -- & "
                         f"{v['AUROC']:.3f} & {v['AUROC_foldmean']:.3f} ({v['AUROC_foldsd']:.3f}) \\\\")
    w("cv_metrics_matched.tex", wrap("\n".join(lines), "@{}lccccc@{}",
        "Model & BA (95\\% CI) & BA fold mean (SD) & F$_1$ & AUROC (95\\% CI) & AUROC fold mean (SD) \\\\"))

    # ------------------------------------------ model-free, original (audited folds)
    lines = []
    for task in a.tasks:
        lines.append(block(TASKS[task], 5))
        for k, name in TRIVIAL_NAMES.items():
            v = report[task]["trivial_original"][k]
            lines.append(f"\\quad {name} & {v['BA']:.3f} & {v['BA_foldmean']:.3f} ({v['BA_foldsd']:.3f}) & "
                         f"{v['AUROC']:.3f} & {v['AUROC_foldmean']:.3f} ({v['AUROC_foldsd']:.3f}) \\\\")
    w("trivial_baseline.tex", wrap("\n".join(lines), "@{}lcccc@{}",
        "Classifier & BA & BA fold mean (SD) & AUROC & AUROC fold mean (SD) \\\\"))

    # ------------------------------------------------------------------ ablation
    lines = []
    for task in a.tasks:
        r = report[task]
        lines.append(block(TASKS[task], 6))
        for mod in ABLATION_ORDER:
            p = pool[(pool.task == task) & (pool.m == mod)]
            f = fold[(fold.task == task) & (fold.m == mod)]
            if p.empty or mod not in r["deep"]:
                continue
            p, f, d = p.iloc[0], f.iloc[0], r["deep"][mod]
            pm = params_millions(mod, prov)
            lines.append(f"\\quad {ABL_LABEL.get(mod, display_name(mod))} & "
                         f"{f'{pm:.2f}' if pm is not None else '--'} & {f.BA_mean:.3f} ({f.BA_sd:.3f}) & "
                         f"{p.AUROC:.3f} & {d['BA_foldmean']:.3f} ({d['BA_foldsd']:.3f}) & {d['AUROC']:.3f} \\\\")
    w("ablation.tex", wrap("\n".join(lines), "@{}lccccc@{}",
        " & & \\multicolumn{2}{c}{Original subset} & \\multicolumn{2}{c}{Matched subset} \\\\\n"
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
        "Configuration & Param.\\ (M) & BA fold mean (SD) & AUROC & BA fold mean (SD) & AUROC \\\\"))

    # ------------------------------------------------------- covariate balance
    lines = []
    for task in a.tasks:
        r = report[task]; b, af = r["before"], r["after"]
        lines.append(block(
            f"{TASKS[task]}: {b['n_pos']}+{b['n_neg']} images $\\rightarrow$ {af['n_pairs']} matched pairs "
            f"({af['n']} images) from {af['n_sites']} site(s), {af['n_hives']} hives", 4))
        lines.append(f"\\quad {FEAT_LABEL['lab_L_256']} & {b['L_smd']:+.3f} & {af['L_smd']:+.3f} & "
                     f"{'yes' if abs(af['L_smd']) < .25 else 'no'} \\\\")
        for c in BALANCE_FEATS:
            v = r["balance"][c]
            lines.append(f"\\quad {FEAT_LABEL.get(c, c)} & {v['smd_before']:+.3f} & {v['smd_after']:+.3f} & "
                         f"{'yes' if abs(v['smd_after']) < .25 else 'no'} \\\\")
    w("matched_subset_balance.tex", wrap("\n".join(lines), "@{}lccc@{}",
        "Global statistic & SMD before matching & SMD after matching & $|$SMD$|<0.25$ \\\\"))

    # --------------------------------------------------------- pairwise summary
    po = pd.read_csv(os.path.join(R, f"cv_pairwise_foldwise{tag}.csv"))
    lines = []
    for task in a.tasks:
        so = po[po.task == task]
        pmpath = os.path.join(a.out_dir, f"matched_pairwise_{task}.csv")
        pm = pd.read_csv(pmpath) if os.path.isfile(pmpath) else pd.DataFrame()
        lines.append(f"{TASKS[task]} & {len(so)} & {(so.delong_p_holm < .05).sum()} & "
                     f"{(so.mcnemar_p_holm < .05).sum()} & {len(pm)} & "
                     f"{(pm.delong_p_holm < .05).sum() if len(pm) else 0} & "
                     f"{(pm.mcnemar_p_holm < .05).sum() if len(pm) else 0} \\\\")
    w("pairwise_summary.tex", wrap("\n".join(lines), "@{}lcccccc@{}",
        " & \\multicolumn{3}{c}{Original subset} & \\multicolumn{3}{c}{Matched subset} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
        "Task & Comparisons & DeLong & McNemar & Comparisons & DeLong & McNemar \\\\"))

    # --------------------------------------------------- full pairwise (supplement)
    def pw_table(df, fname):
        lines = []
        for task in a.tasks:
            s = df[df.task == task]
            if s.empty:
                continue
            lines.append(block(TASKS[task], 9))
            for _, r in s.iterrows():
                pv = lambda x: "$<10^{-4}$" if x < 1e-4 else f"{x:.3f}"
                lines.append(f"\\quad {display_name(r.model)} & {int(r.fold)} & {int(r.n)} & "
                             f"{r.auc_ref:.3f} & {r.auc_model:.3f} & {pv(r.delong_p_holm)} & "
                             f"{int(r.mcnemar_b)} & {int(r.mcnemar_c)} & {pv(r.mcnemar_p_holm)} \\\\")
        w(fname, wrap("\n".join(lines), "@{}llrccccrc@{}",
            "Comparator & Fold & $n$ & AUROC & AUROC & DeLong & \\multicolumn{2}{c}{Discordant pairs} & McNemar \\\\\n"
            "\\cmidrule(lr){7-8}\n"
            f" & & & {display_name(REFERENCE_MODEL)} & comparator & $p_{{\\mathrm{{Holm}}}}$ & "
            f"{display_name(REFERENCE_MODEL)} only & comp.\\ only & $p_{{\\mathrm{{Holm}}}}$ \\\\"))
    pw_table(po, "cv_pairwise_original.tex")
    mp = [pd.read_csv(os.path.join(a.out_dir, f"matched_pairwise_{t}.csv"))
          for t in a.tasks if os.path.isfile(os.path.join(a.out_dir, f"matched_pairwise_{t}.csv"))]
    if mp:
        pw_table(pd.concat(mp), "cv_pairwise_matched.tex")

    # ------------------------------------------------------- confound reliance
    cpath = os.path.join(a.out_dir, "confound_reliance.csv")
    if os.path.isfile(cpath):
        c = pd.read_csv(cpath)
        lines = []
        for task in a.tasks:
            for scope, sl in [("original", "original subset"),
                              ("matched", "acquisition-matched subset")]:
                g = c[(c.task == task) & (c.scope == scope)]
                if g.empty:
                    continue
                g = g.set_index("model")
                ref = g.loc["__reference_label__"]
                lines.append(block(f"{TASKS[task]}, {sl} ($n={int(ref.n)}$)", 5))
                lines.append(f"\\quad \\textit{{Reference: the disease label itself}} & "
                             f"{ref.surrogate_auc:.3f} & {ref.surrogate_ba:.3f} & -- & -- \\\\")
                for mod in models:
                    if mod not in g.index:
                        continue
                    r = g.loc[mod]
                    lines.append(f"\\quad {display_name(mod, latex=True)} & {r.surrogate_auc:.3f} & "
                                 f"{r.surrogate_ba:.3f} ({r.surrogate_ba - ref.surrogate_ba:+.3f}) & "
                                 f"{r.kappa_threshold:.2f} & {r.spearman_L:+.2f} \\\\")
        w("confound_reliance.tex", wrap("\n".join(lines), "@{}lcccc@{}",
            " & \\multicolumn{2}{c}{Reproducible from six global statistics} & Agreement with & Spearman $\\rho$ \\\\\n"
            "\\cmidrule(lr){2-3}\n"
            "Target of the surrogate & AUROC & BA (gap to reference) & $L^*$ rule ($\\kappa$) & with $L^*$ \\\\"))
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
