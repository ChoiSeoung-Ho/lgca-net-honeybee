#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tables added in revision 7 (Ecological Informatics).

* ``training_cost.tex``       -- Table: epochs, wall-clock minutes and s/epoch per
                                 network and task, read from train_log_*.csv
* ``confusion_matrices.tex``  -- Supplementary Table S5: TP/TN/FP/FN at threshold
                                 0.5 for every network on the original and the
                                 acquisition-matched subsets, with the balanced
                                 accuracy and positive-class F1 recomputed from them
* ``training_cost.csv``, ``confusion_matrices.csv`` -- the same numbers as CSV

Pure Python (csv/glob only), so it runs anywhere.

    python make_extra_tables.py --results-dir ./results --tables-dir ./tables
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

TASKS = [("you_chalk_brood", "Chalkbrood"), ("you_foulbrood", "Foulbrood")]
MODELS = ["lgca_net", "lgca_net_no_cross_attn", "lgca_net_cnn_only",
          "resnet50", "densenet121", "efficientnetv2",
          "coatnet", "crossvit", "nextvit_small", "conformer", "lsnet_t", "mambavision"]
NAME = {"lgca_net": r"\textbf{LGCA-Net (proposed)}",
        "lgca_net_no_cross_attn": "LGCA-Net w/o cross-attention",
        "lgca_net_cnn_only": "LGCA-Net CNN branch only",
        "resnet50": "ResNet-50", "densenet121": "DenseNet-121",
        "efficientnetv2": "EfficientNetV2-S", "coatnet": "CoAtNet-0",
        "crossvit": "CrossViT-S", "nextvit_small": "Next-ViT-S",
        "conformer": "Conformer-S", "lsnet_t": "LSNet-T", "mambavision": "MambaVision-T"}
PARAMS_M = {"lgca_net": 1.46, "lgca_net_no_cross_attn": 1.20, "lgca_net_cnn_only": 0.29,
            "resnet50": 23.51, "densenet121": 6.96, "efficientnetv2": 20.18,
            "coatnet": 26.67, "crossvit": 26.28, "nextvit_small": 30.74,
            "conformer": 36.27, "lsnet_t": 11.03, "mambavision": 31.15}


def wrap(body: str, colspec: str, header: str) -> str:
    return (f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header}\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}\n")


def block(label: str, ncol: int) -> str:
    return f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{{label}}}}} \\\\"


def find_log(R: str, task: str, mod: str, fold: int) -> str | None:
    for pat in (f"train_log_{task}_{mod}_fold{fold}_original.csv",
                f"train_log_{task}_{mod}_fold{fold}.csv"):
        p = os.path.join(R, pat)
        if os.path.isfile(p):
            return p
    return None


def find_oof(R: str, task: str, mod: str) -> str | None:
    for pat in (f"oof_{task}_{mod}_original.csv", f"oof_{task}_{mod}.csv"):
        p = os.path.join(R, pat)
        if os.path.isfile(p):
            return p
    return None


def training_cost(R: str, T: str) -> None:
    rows, lines = [], []
    for task, lab in TASKS:
        lines.append(block(lab, 6))
        for mod in MODELS:
            ep, sec, per_fold = 0, 0.0, []
            for f in range(3):
                p = find_log(R, task, mod, f)
                if p is None:
                    continue
                r = list(csv.DictReader(open(p, encoding="utf-8")))
                s = sum(float(x["seconds"]) for x in r)
                ep += len(r); sec += s; per_fold.append(f"{len(r)}")
            if ep == 0:
                continue
            rows.append(dict(task=task, model=mod, params_M=PARAMS_M.get(mod, ""),
                             epochs_total=ep, epochs_per_fold="/".join(per_fold),
                             wallclock_min=round(sec / 60, 1), sec_per_epoch=round(sec / ep, 1)))
            lines.append(f"\\quad {NAME[mod]} & {PARAMS_M.get(mod, '--'):.2f} & "
                         f"{'/'.join(per_fold)} & {ep} & {sec / 60:.1f} & {sec / ep:.1f} \\\\")
    hdr = ("Model & Param.\\ (M) & Epochs per fold & Epochs (total) & "
           "Wall-clock (min) & s / epoch \\\\")
    open(os.path.join(T, "training_cost.tex"), "w", encoding="utf-8").write(
        wrap("\n".join(lines), "@{}lrcrrr@{}", hdr))
    with open(os.path.join(R, "training_cost.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("  wrote", os.path.join(T, "training_cost.tex"))


def confusion(R: str, T: str) -> None:
    rows, lines = [], []
    for task, lab in TASKS:
        mp = os.path.join(R, f"matched_{task}.csv")
        matched = {r["filename"] for r in csv.DictReader(open(mp, encoding="utf-8"))} if os.path.isfile(mp) else set()
        for setting, keep in (("original", None), ("matched", matched)):
            if keep is not None and not keep:
                continue
            lines.append(block(f"{lab}, {setting} subset", 8))
            for mod in MODELS:
                p = find_oof(R, task, mod)
                if p is None:
                    continue
                d = [r for r in csv.DictReader(open(p, encoding="utf-8"))
                     if keep is None or r["filename"] in keep]
                y = [int(r["y_true"]) for r in d]; q = [int(r["pred"]) for r in d]
                tp = sum(a == 1 and b == 1 for a, b in zip(y, q))
                tn = sum(a == 0 and b == 0 for a, b in zip(y, q))
                fp = sum(a == 0 and b == 1 for a, b in zip(y, q))
                fn = sum(a == 1 and b == 0 for a, b in zip(y, q))
                sens, spec = tp / (tp + fn), tn / (tn + fp)
                ba = (sens + spec) / 2
                f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
                rows.append(dict(task=task, setting=setting, model=mod, n=len(d), TP=tp, TN=tn, FP=fp, FN=fn,
                                 BA=round(ba, 4), F1_positive=round(f1, 4)))
                lines.append(f"\\quad {NAME[mod]} & {len(d)} & {tp} & {tn} & {fp} & {fn} & {ba:.3f} & {f1:.3f} \\\\")
    hdr = r"Model & $n$ & TP & TN & FP & FN & BA & F$_1$ (positive class) \\"
    open(os.path.join(T, "confusion_matrices.tex"), "w", encoding="utf-8").write(
        wrap("\n".join(lines), "@{}lrrrrrcc@{}", hdr))
    with open(os.path.join(R, "confusion_matrices.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("  wrote", os.path.join(T, "confusion_matrices.tex"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--tables-dir", default="./tables")
    a = ap.parse_args()
    os.makedirs(a.tables_dir, exist_ok=True)
    training_cost(a.results_dir, a.tables_dir)
    confusion(a.results_dir, a.tables_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
