#!/usr/bin/env python3
"""Regenerate every LaTeX fragment from the results CSVs, idempotently.

Each fragment is produced by delegating to the script that owns it. When that
script's inputs are missing, a **compile-safe placeholder** is written instead::

    \\textcolor{red}{[pending: run <script>]}

so the manuscript always builds and every gap is visible in the PDF. Running
this script twice with the same inputs produces byte-identical output.

Fragments and their owners
--------------------------
============================== ==================================
fragment                       produced by
============================== ==================================
fold_class_counts.tex          make_splits.py
group_structure.tex            make_splits.py
image_statistics.tex           trivial_baseline.py
trivial_baseline.tex           trivial_baseline.py
near_duplicates.tex            near_duplicates.py
matched_subset_balance.tex     rebuild_matched_subset.py
lr_selected.tex                train_cv.py (results/lr_selected_<task>.csv)
ablation.tex                   evaluate_cv.py
cv_metrics_original.tex        evaluate_cv.py --run-tag original
seed_table_matched.tex         seed_stats.py --run-tag matched
seed_pairwise_matched.tex      seed_stats.py --run-tag matched
faithfulness.tex               faithfulness.py (or the legacy POS_NEG/DROP_INC CSVs)
cv_metrics.tex                 evaluate_cv.py
cv_foldwise.tex                evaluate_cv.py
cv_pairwise_foldwise.tex       evaluate_cv.py
seed_table.tex                 seed_stats.py
seed_pairwise.tex              seed_stats.py
xai_localization.tex           xai_localization_foldwise.py
============================== ==================================

Examples
--------
    python make_tables.py                       # regenerate everything possible
    python make_tables.py --only cv_metrics.tex seed_table.tex
    python make_tables.py --placeholders-only   # stub every missing fragment
    python make_tables.py --dry-run
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Callable, Dict, List, Optional, Sequence

from common import latex
from common.data import TASKS

#: fragment -> (owning script, list of glob patterns that must match at least once)
FRAGMENTS: Dict[str, Dict[str, object]] = {
    "fold_class_counts.tex": {
        "script": "make_splits.py",
        "requires": ["{r}/split_manifest_*.csv"],
    },
    "group_structure.tex": {
        "script": "make_splits.py",
        "requires": ["{r}/group_structure_*.csv"],
    },
    "image_statistics.tex": {
        "script": "trivial_baseline.py",
        "requires": ["{r}/features_*.csv", "{r}/split_manifest_*.csv"],
    },
    "trivial_baseline.tex": {
        "script": "trivial_baseline.py",
        "requires": ["{r}/features_*.csv", "{r}/split_manifest_*.csv"],
    },
    "near_duplicates.tex": {
        "script": "near_duplicates.py",
        "requires": ["{r}/features_*.csv"],
    },
    "matched_subset_balance.tex": {
        # Rebuilt from the balance CSV; the matching itself needs the large normal
        # pool, so this fragment is regenerated, never re-matched, from here.
        "script": "rebuild_matched_subset.py",
        "requires": ["{r}/matched_balance_*.csv"],
    },
    "lr_selected.tex": {
        "script": "train_cv.py",
        "requires": ["{r}/lr_selected_*.csv"],
    },
    "cv_metrics.tex": {"script": "evaluate_cv.py", "requires": ["{r}/oof_*.csv"]},
    "cv_foldwise.tex": {"script": "evaluate_cv.py", "requires": ["{r}/oof_*.csv"]},
    "cv_pairwise_foldwise.tex": {"script": "evaluate_cv.py", "requires": ["{r}/oof_*.csv"]},
    "seed_table.tex": {
        "script": "seed_stats.py",
        "requires_any": ["{r}/seeds_*.csv", "{r}/seed_table_*_95CI.csv"],
    },
    "seed_pairwise.tex": {
        "script": "seed_stats.py",
        "requires_any": ["{r}/seeds_*.csv", "{r}/seed_table_*_95CI.csv"],
    },
    "ablation.tex": {"script": "evaluate_cv.py", "requires": ["{r}/oof_*.csv"]},
    "cv_metrics_original.tex": {
        "script": "evaluate_cv.py --run-tag original",
        "requires": ["{r}/oof_*_original.csv"],
    },
    "seed_table_matched.tex": {
        "script": "seed_stats.py --run-tag matched",
        "requires_any": ["{r}/seeds_*_matched.csv", "{r}/matched/seeds_*.csv"],
    },
    "seed_pairwise_matched.tex": {
        "script": "seed_stats.py --run-tag matched",
        "requires_any": ["{r}/seeds_*_matched.csv", "{r}/matched/seeds_*.csv"],
    },
    "xai_localization.tex": {
        "script": "xai_localization_foldwise.py",
        "requires": ["{r}/xai_localization_*.csv"],
    },
    "faithfulness.tex": {
        "script": "faithfulness.py",
        "requires_any": ["{r}/faithfulness_*.csv",
                         "{p}/*_pos_neg_perturb_auc.csv",
                         "{d}/*_drop_inc.csv"],
    },
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="./data",
                   help="Forwarded to make_splits.py when the split tables are rebuilt.")
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--table-dir", default="./tables")
    p.add_argument("--figure-dir", default="./figures")
    p.add_argument("--only", nargs="+", default=None,
                   help="Regenerate only these fragment file names.")
    p.add_argument("--summary-csv", nargs="*", default=[],
                   help="Legacy seed summary CSVs as task=path, forwarded to seed_stats.py.")
    p.add_argument("--placeholders-only", action="store_true",
                   help="Do not run any generator; just stub every missing fragment.")
    p.add_argument("--force-placeholders", action="store_true",
                   help="Overwrite an existing fragment with a placeholder when its "
                        "inputs are missing (default: keep the existing content).")
    p.add_argument("--pos-neg-dir", default="./results_POS_NEG",
                   help="Legacy per-fold perturbation-AUC CSVs "
                        "(<task>_pos_neg_perturb_auc.csv).")
    p.add_argument("--drop-inc-dir", default="./results_DROP_INC",
                   help="Legacy per-fold Average-Drop / Increase-in-Confidence CSVs "
                        "(<task>_grad_saliency_drop_inc.csv, <task>_lime_shap_drop_inc.csv).")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--dry-run", action="store_true",
                   help="Report the plan; write nothing.")
    return p


def satisfied(spec: Dict[str, object], results_dir: str,
              extra_inputs: Sequence[str] = (),
              pos_neg_dir: str = "./results_POS_NEG",
              drop_inc_dir: str = "./results_DROP_INC") -> bool:
    """True when the fragment's owning script has the inputs it needs.

    ``extra_inputs`` lets a caller declare inputs found outside ``results_dir``
    (e.g. legacy seed summary CSVs discovered next to the package).
    """
    fmt = dict(r=results_dir, p=pos_neg_dir, d=drop_inc_dir)
    req = [p.format(**fmt) for p in spec.get("requires", [])]  # type: ignore[arg-type]
    any_req = [p.format(**fmt) for p in spec.get("requires_any", [])]  # type: ignore[arg-type]
    if req and not all(glob.glob(p) for p in req):
        return False
    if any_req and not (any(glob.glob(p) for p in any_req) or extra_inputs):
        return False
    return bool(req or any_req)


def find_legacy_summaries(results_dir: str, extra: Sequence[str]) -> List[str]:
    """Build the ``task=path`` list for seed_stats.py from CLI args plus discovery."""
    out = list(extra)
    have = {item.split("=", 1)[0] for item in out if "=" in item}
    here = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [results_dir, os.path.join(results_dir, ".."), here,
                   os.path.join(here, ".."), "."]
    for task in TASKS:
        if task in have:
            continue
        for d in search_dirs:
            cand = os.path.join(d, f"seed_table_{task}_95CI.csv")
            if os.path.exists(cand):
                out.append(f"{task}={os.path.normpath(cand)}")
                break
    return out


def is_placeholder(path: str) -> bool:
    """True when a fragment on disk is one of our own red 'pending' stubs."""
    try:
        with open(path, encoding="utf-8") as fh:
            return "[pending: run " in fh.read()
    except OSError:
        return False


def generators(args) -> Dict[str, Callable[[], int]]:
    """fragment -> zero-argument callable that regenerates it (and its siblings)."""
    import evaluate_cv
    import make_splits
    import near_duplicates
    import seed_stats
    import trivial_baseline

    from common.data import task_label
    from common.models import display_name

    common = ["--table-dir", args.table_dir]

    def run_splits():
        return make_splits.main(["--data-root", args.data_root,
                                 "--out-dir", args.results_dir] + common)

    def run_trivial():
        return trivial_baseline.main(
            ["--features-dir", args.results_dir, "--manifest-dir", args.results_dir,
             "--out-dir", args.results_dir, "--n-boot", str(args.n_boot)] + common)

    def run_dupes():
        return near_duplicates.main(
            ["--features-dir", args.results_dir, "--manifest-dir", args.results_dir,
             "--out-dir", args.results_dir] + common)

    def run_eval():
        return evaluate_cv.main(
            ["--results-dir", args.results_dir, "--out-dir", args.results_dir,
             "--n-boot", str(args.n_boot)] + common)

    def run_seeds():
        argv = ["--results-dir", args.results_dir, "--out-dir", args.results_dir,
                "--figure-dir", args.figure_dir] + common
        summaries = find_legacy_summaries(args.results_dir, args.summary_csv)
        if summaries:
            argv += ["--summary-csv"] + summaries
        return seed_stats.main(argv)

    def run_matched():
        # The balance table is rebuilt from the per-task balance CSVs so that a
        # table refresh never re-runs (and never silently re-randomises) matching.
        import csv as _csv

        rows = []
        for path in sorted(glob.glob(os.path.join(args.results_dir, "matched_balance_*.csv"))):
            with open(path, newline="", encoding="utf-8-sig") as fh:
                rows.extend(list(_csv.DictReader(fh)))
        if not rows:
            return 1
        header = ["Task", "Quantity", "Positive (mean)", "Control (mean)",
                  "SMD before", "SMD after", "Balanced ($|$SMD$| < 0.1$)"]
        out = []
        for r in rows:
            out.append([
                latex.esc(r.get("task", "")), r.get("label", r.get("feature", "")),
                latex.fmt(r.get("mean_positive_after")),
                latex.fmt(r.get("mean_control_after")),
                latex.fmt(r.get("smd_before")), latex.fmt(r.get("smd_after")),
                "yes" if str(r.get("balanced_after")) == "1" else "no",
            ])
        latex.write_fragment(
            os.path.join(args.table_dir, "matched_subset_balance.tex"),
            latex.tabular(header, out, align="llccccc",
                          notes=["Acquisition-matched subset balance, rebuilt from "
                                 "results/matched_balance_<task>.csv.",
                                 "SMD = (mean$_{positive}$ $-$ mean$_{control}$) / pooled SD."]))
        print("wrote", os.path.join(args.table_dir, "matched_subset_balance.tex"))
        return 0

    def run_lr():
        """tables/lr_selected.tex from results/lr_selected_<task>.csv."""
        import csv as _csv

        by_task: Dict[str, Dict[str, Dict[float, Dict[str, float]]]] = {}
        for path in sorted(glob.glob(os.path.join(args.results_dir, "lr_selected_*.csv"))):
            with open(path, newline="", encoding="utf-8-sig") as fh:
                for r in _csv.DictReader(fh):
                    task = r.get("task", "")
                    try:
                        lr = float(r["lr"])
                        ba = float(r["val_ba"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    entry = by_task.setdefault(task, {}).setdefault(r["model"], {})
                    entry[lr] = {"val_ba": ba,
                                 "selected": 1.0 if str(r.get("selected", "0")) in
                                 ("1", "True", "true") else 0.0}
        if not by_task:
            return 1

        tasks = sorted(by_task)
        grid = sorted({lr for t in by_task.values() for m in t.values() for lr in m},
                      reverse=True)
        models = sorted({m for t in by_task.values() for m in t})

        header = ["Model"]
        group: List = [("", 1)]
        for task in tasks:
            header += ["Selected LR"] + [f"{lr:g}" for lr in grid]
            group.append((latex.esc(task_label(task)), 1 + len(grid)))
        rows = []
        for model in models:
            cells = [latex.esc(display_name(model))]
            for task in tasks:
                per_lr = by_task[task].get(model, {})
                sel = [lr for lr, d in per_lr.items() if d["selected"]]
                cells.append(f"{sel[0]:g}" if sel else "--")
                for lr in grid:
                    d = per_lr.get(lr)
                    if d is None:
                        cells.append("--")
                    else:
                        v = latex.fmt(d["val_ba"])
                        cells.append(latex.bold(v) if d["selected"] else v)
            rows.append(cells)
        out = latex.tabular(
            header, rows, align="l" + ("c" * (1 + len(grid))) * len(tasks),
            group_header=group,
            notes=["Learning-rate selection. Values are the validation balanced "
                   "accuracy at the best-validation-loss epoch on fold 0's "
                   "validation split; the bold entry is the selected rate, which is "
                   "then held fixed for every fold.",
                   "Selection never sees a test fold. From-scratch models use the "
                   "narrower grid, so blank cells are rates that were not searched."])
        latex.write_fragment(os.path.join(args.table_dir, "lr_selected.tex"), out)
        print("wrote", os.path.join(args.table_dir, "lr_selected.tex"))
        return 0

    def run_eval_tagged(tag: str):
        """evaluate_cv.py for a tagged protocol (e.g. the original-corpus run)."""
        def _run():
            return evaluate_cv.main(
                ["--results-dir", args.results_dir, "--out-dir", args.results_dir,
                 "--n-boot", str(args.n_boot), "--run-tag", tag] + common)
        _run.__name__ = f"run_eval_{tag}"
        return _run

    def run_seeds_tagged(tag: str):
        """seed_stats.py for a tagged protocol (e.g. the matched-subset run)."""
        def _run():
            rdir = args.results_dir
            # a tagged run may live either as seeds_<task>_<tag>.csv here or as
            # seeds_<task>.csv inside a per-protocol subdirectory
            if not glob.glob(os.path.join(rdir, f"seeds_*_{tag}.csv")) and \
                    glob.glob(os.path.join(rdir, tag, "seeds_*.csv")):
                rdir = os.path.join(rdir, tag)
            return seed_stats.main(
                ["--results-dir", rdir, "--out-dir", rdir,
                 "--figure-dir", args.figure_dir, "--run-tag", tag] + common)
        _run.__name__ = f"run_seeds_{tag}"
        return _run

    def run_faithfulness():
        """tables/faithfulness.tex from the new CSVs, or the author's legacy exports."""
        import csv as _csv

        from faithfulness import faithfulness_table, load_legacy_faithfulness

        rows: List[Dict[str, object]] = []
        for path in sorted(glob.glob(os.path.join(args.results_dir, "faithfulness_*.csv"))):
            if "perimage" in os.path.basename(path):
                continue
            with open(path, newline="", encoding="utf-8-sig") as fh:
                rows.extend(list(_csv.DictReader(fh)))
        if rows:
            print(f"  faithfulness: {len(rows)} rows from results/faithfulness_*.csv")
        else:
            rows = load_legacy_faithfulness(args.pos_neg_dir, args.drop_inc_dir)
            if rows:
                print(f"  faithfulness: {len(rows)} fold rows recovered from the legacy "
                      f"exports in {args.pos_neg_dir} / {args.drop_inc_dir}")
        if not rows:
            return 1
        latex.write_fragment(os.path.join(args.table_dir, "faithfulness.tex"),
                             faithfulness_table(rows))
        print("wrote", os.path.join(args.table_dir, "faithfulness.tex"))
        return 0

    def run_xai():
        # xai_localization_foldwise.py needs checkpoints and annotations, so its
        # fragment is only ever *rebuilt* from an existing per-image CSV.
        import csv as _csv

        from xai_localization_foldwise import summary_table

        rows = []
        for p in sorted(glob.glob(os.path.join(args.results_dir, "xai_localization_*.csv"))):
            with open(p, newline="", encoding="utf-8-sig") as fh:
                rows.extend(list(_csv.DictReader(fh)))
        if not rows:
            return 1
        latex.write_fragment(os.path.join(args.table_dir, "xai_localization.tex"),
                             summary_table(rows))
        print("wrote", os.path.join(args.table_dir, "xai_localization.tex"))
        return 0

    return {
        "fold_class_counts.tex": run_splits,
        "group_structure.tex": run_splits,
        "image_statistics.tex": run_trivial,
        "trivial_baseline.tex": run_trivial,
        "near_duplicates.tex": run_dupes,
        "matched_subset_balance.tex": run_matched,
        "lr_selected.tex": run_lr,
        "ablation.tex": run_eval,
        "cv_metrics_original.tex": run_eval_tagged("original"),
        "seed_table_matched.tex": run_seeds_tagged("matched"),
        "seed_pairwise_matched.tex": run_seeds_tagged("matched"),
        "faithfulness.tex": run_faithfulness,
        "cv_metrics.tex": run_eval,
        "cv_foldwise.tex": run_eval,
        "cv_pairwise_foldwise.tex": run_eval,
        "seed_table.tex": run_seeds,
        "seed_pairwise.tex": run_seeds,
        "xai_localization.tex": run_xai,
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    os.makedirs(args.table_dir, exist_ok=True)
    wanted = list(args.only) if args.only else list(FRAGMENTS)
    unknown = [f for f in wanted if f not in FRAGMENTS]
    if unknown:
        print(f"[error] unknown fragment(s): {', '.join(unknown)}")
        return 2

    seed_inputs = find_legacy_summaries(args.results_dir, args.summary_csv)
    gens = None if args.placeholders_only else generators(args)
    done: set = set()
    made, stubbed, kept = [], [], []

    for frag in wanted:
        spec = FRAGMENTS[frag]
        script = str(spec["script"])
        extra = seed_inputs if script == "seed_stats.py" else ()
        ok = satisfied(spec, args.results_dir, extra,
                       args.pos_neg_dir, args.drop_inc_dir)
        if args.dry_run:
            print(f"{frag:<28s} {'regenerate via ' + script if ok else 'PLACEHOLDER (' + script + ')'}")
            continue
        if ok and gens is not None:
            fn = gens[frag]
            key = getattr(fn, "__name__", frag)
            if key in done:
                made.append(frag)
                continue
            print(f"--- {frag}: running {script}")
            try:
                rc = fn()
            except Exception as exc:
                print(f"[warn] {script} failed for {frag}: {exc}")
                rc = 1
            done.add(key)
            if rc == 0 and os.path.exists(os.path.join(args.table_dir, frag)):
                made.append(frag)
                continue
            ok = False
        if not ok:
            dest = os.path.join(args.table_dir, frag)
            # Never overwrite a real fragment with a stub: a previous run (or a
            # generator invoked directly with arguments this script cannot
            # reconstruct, e.g. seed_stats.py with explicit --summary-csv) may
            # have produced valid content whose inputs are no longer on disk.
            if os.path.exists(dest) and not is_placeholder(dest) and not args.force_placeholders:
                print(f"[keep] {frag}: inputs missing but an existing fragment is "
                      f"preserved (use --force-placeholders to overwrite)")
                kept.append(frag)
                continue
            latex.write_fragment(dest, latex.placeholder(script, frag))
            stubbed.append(frag)

    if not args.dry_run:
        print(f"\nregenerated:  {len(made)} -> {', '.join(made) if made else '(none)'}")
        print(f"kept as-is:   {len(kept)} -> {', '.join(kept) if kept else '(none)'}")
        print(f"placeholders: {len(stubbed)} -> {', '.join(stubbed) if stubbed else '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
