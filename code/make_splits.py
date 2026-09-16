#!/usr/bin/env python3
"""Write the cross-validation split manifests and the group-structure tables.

For every task this script produces

* ``results/split_manifest_<task>.csv``          - stratified 3-fold CV (seed 42)
* ``results/split_manifest_grouped_<task>.csv``  - hive-proxy-grouped 3-fold CV
* ``results/fold_class_counts_<task>.csv``       - per-fold, per-role class counts
* ``results/group_structure_<task>.csv``         - site / hive-proxy / session counts
* ``tables/fold_class_counts.tex``               - LaTeX fragment (both tasks)
* ``tables/group_structure.tex``                 - LaTeX fragment (both tasks)

Manifest columns: ``filename, path, label, class_dir, group, site, session,
fold, role`` with ``role`` in {train, val, test}.

The validation split is a **stratified** 25% carve-out of each fold's training
portion (``train_test_split(stratify=y, random_state=42)``), replacing the
original "last 25% of the shuffled training indices" rule.

Examples
--------
    python make_splits.py --data-root ./data
    python make_splits.py --data-root ./data --tasks you_chalk_brood --dry-run
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Sequence

from common import latex
from common.data import (
    TASKS,
    Sample,
    build_manifest_rows,
    list_task_images,
    make_grouped_folds,
    make_stratified_folds,
    samples_to_arrays,
    task_label,
    write_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="./data", help="Directory containing <task>/{normal,abnormal}")
    p.add_argument("--tasks", nargs="+", default=list(TASKS), help="Tasks to process")
    p.add_argument("--out-dir", default="./results", help="Where the CSVs go")
    p.add_argument("--table-dir", default="./tables", help="Where the LaTeX fragments go")
    p.add_argument("--n-splits", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be written; do not write manifests.")
    return p


# --------------------------------------------------------------------------- #
def fold_class_counts(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Per (fold, role) class counts."""
    agg: Dict[tuple, Dict[str, int]] = defaultdict(lambda: {"normal": 0, "abnormal": 0})
    for r in rows:
        agg[(int(r["fold"]), str(r["role"]))][str(r["class_dir"])] += 1
    out = []
    for (fold, role), c in sorted(agg.items(), key=lambda kv: (kv[0][0], ["train", "val", "test"].index(kv[0][1]))):
        total = c["normal"] + c["abnormal"]
        out.append({
            "fold": fold, "role": role,
            "n_normal": c["normal"], "n_abnormal": c["abnormal"],
            "n_total": total,
            "pos_rate": round(c["abnormal"] / total, 4) if total else float("nan"),
        })
    return out


def group_structure(samples: Sequence[Sample]) -> Dict[str, object]:
    """Counts of sites / hive-proxies / sessions overall and per class."""
    per_class = {"normal": {"site": set(), "hive": set(), "sess": set(), "n": 0},
                 "abnormal": {"site": set(), "hive": set(), "sess": set(), "n": 0}}
    all_sites, all_hives, all_sess = set(), set(), set()
    hive_classes: Dict[str, set] = defaultdict(set)
    site_classes: Dict[str, set] = defaultdict(set)
    unparsed = 0
    for s in samples:
        d = per_class[s.class_dir]
        d["site"].add(s.site); d["hive"].add(s.hive_proxy); d["sess"].add(s.session)
        d["n"] += 1
        all_sites.add(s.site); all_hives.add(s.hive_proxy); all_sess.add(s.session)
        hive_classes[s.hive_proxy].add(s.class_dir)
        site_classes[s.site].add(s.class_dir)
        if s.disease_code is None:
            unparsed += 1
    return {
        "n_images": len(samples),
        "n_unparsed_filenames": unparsed,
        "n_sites": len(all_sites),
        "n_hive_proxies": len(all_hives),
        "n_sessions": len(all_sess),
        "n_images_normal": per_class["normal"]["n"],
        "n_images_abnormal": per_class["abnormal"]["n"],
        "n_sites_normal": len(per_class["normal"]["site"]),
        "n_sites_abnormal": len(per_class["abnormal"]["site"]),
        "n_hive_proxies_normal": len(per_class["normal"]["hive"]),
        "n_hive_proxies_abnormal": len(per_class["abnormal"]["hive"]),
        "n_sessions_normal": len(per_class["normal"]["sess"]),
        "n_sessions_abnormal": len(per_class["abnormal"]["sess"]),
        "n_hive_proxies_both_classes": sum(1 for v in hive_classes.values() if len(v) > 1),
        "n_sites_both_classes": sum(1 for v in site_classes.values() if len(v) > 1),
        "images_per_hive_proxy_mean": round(len(samples) / max(len(all_hives), 1), 2),
    }


def write_csv(path: str, rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns))
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})
    return path


# --------------------------------------------------------------------------- #
def fold_counts_table(per_task: Dict[str, Dict[str, List[Dict[str, object]]]]) -> str:
    header = ["Task", "Scheme", "Fold", "Role", "Normal", "Abnormal", "Total", "Positive rate"]
    rows: List[List[str]] = []
    mid: List[int] = []
    for task, schemes in per_task.items():
        for scheme, counts in schemes.items():
            for c in counts:
                rows.append([
                    latex.esc(task), latex.esc(scheme), str(c["fold"]), c["role"],
                    str(c["n_normal"]), str(c["n_abnormal"]), str(c["n_total"]),
                    latex.fmt(c["pos_rate"]),
                ])
            mid.append(len(rows) - 1)
    return latex.tabular(
        header, rows, align="llccrrrc", midrules_after=mid,
        notes=["Fold/role class counts. Positive rate = abnormal / total.",
               "Scheme 'stratified' = StratifiedKFold(3, shuffle, seed 42); "
               "'hive-grouped' = StratifiedGroupKFold on the f1_f2_f3 hive proxy."],
    )


def group_structure_table(per_task: Dict[str, Dict[str, object]]) -> str:
    fields = [
        ("Images (total)", "n_images"),
        ("Images (normal)", "n_images_normal"),
        ("Images (abnormal)", "n_images_abnormal"),
        ("Sites (f1\\_f2)", "n_sites"),
        ("Hive proxies (f1\\_f2\\_f3)", "n_hive_proxies"),
        ("Sessions (hive $\\times$ date)", "n_sessions"),
        ("Hive proxies, normal", "n_hive_proxies_normal"),
        ("Hive proxies, abnormal", "n_hive_proxies_abnormal"),
        ("Hive proxies in \\emph{both} classes", "n_hive_proxies_both_classes"),
        ("Sites in \\emph{both} classes", "n_sites_both_classes"),
        ("Mean images per hive proxy", "images_per_hive_proxy_mean"),
        ("Unparsable file names", "n_unparsed_filenames"),
    ]
    tasks = list(per_task)
    header = ["Quantity"] + [task_label(t) + " task" for t in tasks]
    rows = []
    for label, key in fields:
        cells = []
        for t in tasks:
            v = per_task[t].get(key, "")
            cells.append(latex.fmt(v) if isinstance(v, float) else str(v))
        rows.append([label] + cells)
    return latex.tabular(
        header, rows, align="l" + "r" * len(tasks),
        notes=["Proxy acquisition-unit structure derived from the AI-Hub file names.",
               "The provider does not document f1/f2/f3 semantics; they are used "
               "only as nested proxy identifiers."],
    )


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.table_dir, exist_ok=True)

    counts_by_task: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    struct_by_task: Dict[str, Dict[str, object]] = {}

    for task in args.tasks:
        samples = list_task_images(args.data_root, task)
        if not samples:
            print(f"[warn] no images found for task '{task}' under {args.data_root}")
            continue
        paths, labels, groups = samples_to_arrays(samples)
        n_pos = int(labels.sum())
        print(f"[{task}] {len(samples)} images ({len(samples) - n_pos} normal / {n_pos} abnormal)")

        struct = group_structure(samples)
        struct_by_task[task] = struct
        print(f"[{task}] sites={struct['n_sites']} hive-proxies={struct['n_hive_proxies']} "
              f"sessions={struct['n_sessions']} "
              f"hive-proxies in both classes={struct['n_hive_proxies_both_classes']}")

        strat_folds = make_stratified_folds(paths, labels, n_splits=args.n_splits, seed=args.seed)
        grouped_folds = make_grouped_folds(paths, labels, groups,
                                           n_splits=args.n_splits, seed=args.seed)

        strat_rows = build_manifest_rows(samples, strat_folds,
                                         val_fraction=args.val_fraction, seed=args.seed)
        grouped_rows = build_manifest_rows(samples, grouped_folds,
                                           val_fraction=args.val_fraction, seed=args.seed)

        counts_by_task[task] = {
            "stratified": fold_class_counts(strat_rows),
            "hive-grouped": fold_class_counts(grouped_rows),
        }

        if args.dry_run:
            print(f"[dry-run] would write manifests for {task} "
                  f"({len(strat_rows)} + {len(grouped_rows)} rows)")
            continue

        m1 = write_manifest(strat_rows, os.path.join(args.out_dir, f"split_manifest_{task}.csv"))
        m2 = write_manifest(grouped_rows, os.path.join(args.out_dir, f"split_manifest_grouped_{task}.csv"))
        c1 = write_csv(os.path.join(args.out_dir, f"fold_class_counts_{task}.csv"),
                       [dict(scheme="stratified", **r) for r in counts_by_task[task]["stratified"]]
                       + [dict(scheme="hive-grouped", **r) for r in counts_by_task[task]["hive-grouped"]],
                       ["scheme", "fold", "role", "n_normal", "n_abnormal", "n_total", "pos_rate"])
        c2 = write_csv(os.path.join(args.out_dir, f"group_structure_{task}.csv"),
                       [struct], list(struct))
        for p in (m1, m2, c1, c2):
            print("  wrote", p)

    if not args.dry_run and counts_by_task:
        f1 = latex.write_fragment(os.path.join(args.table_dir, "fold_class_counts.tex"),
                                  fold_counts_table(counts_by_task))
        f2 = latex.write_fragment(os.path.join(args.table_dir, "group_structure.tex"),
                                  group_structure_table(struct_by_task))
        print("  wrote", f1)
        print("  wrote", f2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
