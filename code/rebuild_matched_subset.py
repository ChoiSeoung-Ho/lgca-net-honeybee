#!/usr/bin/env python3
"""Build an acquisition-matched, class-balanced subset to break the site/brightness confound.

The problem
-----------
In the delivered task folders the normal and abnormal images are not drawn from
the same acquisition conditions: they come from different sites and differ
strongly in global brightness. A classifier can then reach high accuracy from
mean L* alone (see ``trivial_baseline.py``), and no amount of cross-validation
fixes that, because the confound is present in every fold.

The remedy
----------
Rebuild the negative class by **matching** it to the positives on acquisition
proxies, so the two classes become comparable on everything except disease:

(a) **Restrict** the candidate normals to the acquisition units present among the
    positives -- sites (``f1_f2``) with ``--match-level site`` (default), or
    hive proxies (``f1_f2_f3``) with ``--match-level hive``, which is stricter
    but usually leaves far fewer candidates.
(b) **1:1 nearest-neighbour match without replacement** on mean L* at 256x256
    (optionally also a*/b* via ``--match-features``), inside a caliper of
    ``--caliper`` pooled SDs (default 0.1). Positives are processed in a random
    order seeded with ``--seed`` (42), each taking its nearest unused candidate
    *within its own matching stratum*; unmatched positives are dropped so the
    result stays exactly 1:1 and balanced.
(c) **Report** standardised mean differences (SMD) for L*, a*, b* and file size
    before and after matching, plus the site overlap between classes.

With multiple matching features the distance is Euclidean in **standardised**
space (each feature divided by its pooled SD), and the caliper applies to that
same standardised distance, so ``--caliper`` means the same thing regardless of
how many features are matched on.

Inputs
------
``--positive-features``
    Features CSV covering the positive (diseased) images.
``--candidate-features``
    Features CSV covering the **large** candidate pool of normal images -- run
    ``extract_image_features.py`` over the full AI-Hub normal folder first, which
    is the whole point: matching needs many more candidates than positives.

Both may be the same file (it is split by the ``label`` column); supplying a
large separate normal pool is strongly preferred.

Outputs
-------
``results/matched_subset_<task>.csv``    ``filename, label, path`` (+ matching detail)
``results/matched_pairs_<task>.csv``     positive <-> matched control, with distances
``tables/matched_subset_balance.tex``    SMD before/after per feature, N matched

Exit status is non-zero when fewer than ``--n-per-class`` pairs are matched,
unless ``--allow-fewer`` is given -- so a pipeline cannot silently proceed on an
underpowered subset.

Examples
--------
    python rebuild_matched_subset.py --task you_chalk_brood \\
        --positive-features results/features_you_chalk_brood.csv \\
        --candidate-features results/features_normal_pool.csv
    python rebuild_matched_subset.py --task you_foulbrood --match-level hive \\
        --match-features lab_L lab_a lab_b --caliper 0.2 --allow-fewer
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from common import latex
from common.data import TASKS, task_label

#: features whose CSV column carries a ``_native`` / ``_256`` suffix
RESOLUTION_DEPENDENT = ("lab_L", "lab_a", "lab_b", "rgb_r", "rgb_g", "rgb_b")

#: features whose balance is reported before/after matching
BALANCE_FEATURES = ["lab_L", "lab_a", "lab_b", "file_size_bytes"]

FEATURE_LABEL = {
    "lab_L": "CIELAB $L^*$ (0--100)",
    "lab_a": "CIELAB $a^*$ (CIE units)",
    "lab_b": "CIELAB $b^*$ (CIE units)",
    "file_size_bytes": "JPEG file size (bytes)",
    "mean_intensity_256": "Mean grey level (0--255)",
    "native_w": "Image width (px)",
    "native_h": "Image height (px)",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="you_chalk_brood", choices=list(TASKS))
    p.add_argument("--positive-features", required=True,
                   help="Features CSV containing the positive (label=1) images.")
    p.add_argument("--candidate-features", required=True,
                   help="Features CSV for the large candidate pool of normal images.")
    p.add_argument("--out-dir", default="./results")
    p.add_argument("--table-dir", default="./tables")
    p.add_argument("--match-level", choices=["site", "hive"], default="site",
                   help="Acquisition unit the candidate pool is restricted to and "
                        "matched within (site = f1_f2, hive = f1_f2_f3).")
    p.add_argument("--match-features", nargs="+", default=["lab_L"],
                   help="Features matched on (default: lab_L only).")
    p.add_argument("--resolution", choices=["256", "native"], default="256",
                   help="Resolution of the colour features used for matching.")
    p.add_argument("--caliper", type=float, default=0.1,
                   help="Maximum standardised distance for a valid match, in pooled "
                        "SDs (default 0.1).")
    p.add_argument("--n-per-class", type=int, default=2000,
                   help="Required matched pairs per class (default 2000).")
    p.add_argument("--allow-fewer", action="store_true",
                   help="Exit 0 even when fewer than --n-per-class pairs are matched.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="Report the matching outcome and balance; write nothing.")
    return p


# --------------------------------------------------------------------------- #
def column_for(feature: str, resolution: str) -> str:
    """Map a logical feature name to its CSV column at the requested resolution."""
    return f"{feature}_{resolution}" if feature in RESOLUTION_DEPENDENT else feature


def load_features(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def unit_of(row: Dict[str, str], level: str) -> str:
    """The acquisition unit a row belongs to."""
    return row.get("site", "") if level == "site" else row.get("hive_proxy", "")


def values(rows: Sequence[Dict[str, str]], feature: str, resolution: str) -> np.ndarray:
    col = column_for(feature, resolution)
    out = []
    for r in rows:
        try:
            out.append(float(r[col]))
        except (KeyError, TypeError, ValueError):
            out.append(np.nan)
    return np.asarray(out, dtype=float)


def smd(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference using the pooled SD.

    Convention: ``(mean_positive - mean_control) / pooled_SD``. |SMD| < 0.1 is
    the usual "balanced" criterion in the matching literature.
    """
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    sp = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    if sp <= 0:
        return 0.0 if abs(a.mean() - b.mean()) < 1e-12 else float("nan")
    return float((a.mean() - b.mean()) / sp)


# --------------------------------------------------------------------------- #
def match(
    positives: Sequence[Dict[str, str]],
    candidates: Sequence[Dict[str, str]],
    match_features: Sequence[str],
    resolution: str,
    level: str,
    caliper: float,
    seed: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Greedy 1:1 nearest-neighbour matching without replacement, within stratum.

    Returns ``(pair_rows, diagnostics)``. Each pair row records the positive, its
    matched control, the standardised distance and the shared acquisition unit.
    """
    rng = np.random.default_rng(seed)

    pos_mat = np.column_stack([values(positives, f, resolution) for f in match_features])
    cand_mat = np.column_stack([values(candidates, f, resolution) for f in match_features])

    # Pooled SD per feature, computed once over both classes, so the caliper has
    # a fixed meaning independent of which subset is being examined.
    scale = np.empty(len(match_features), dtype=float)
    for j in range(len(match_features)):
        a = pos_mat[:, j][np.isfinite(pos_mat[:, j])]
        b = cand_mat[:, j][np.isfinite(cand_mat[:, j])]
        if a.size < 2 or b.size < 2:
            scale[j] = np.nan
        else:
            sp = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
            scale[j] = sp if sp > 0 else np.nan
    if not np.all(np.isfinite(scale)):
        bad = [f for f, s in zip(match_features, scale) if not np.isfinite(s)]
        raise ValueError(f"cannot standardise matching feature(s) {bad}: zero or "
                         "undefined pooled SD (constant or missing column)")

    pos_z = pos_mat / scale
    cand_z = cand_mat / scale

    # index candidates by acquisition unit
    by_unit: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(candidates):
        by_unit[unit_of(r, level)].append(i)

    order = rng.permutation(len(positives))
    used = np.zeros(len(candidates), dtype=bool)
    pairs: List[Dict[str, object]] = []
    n_no_unit = n_no_candidate = n_outside_caliper = 0

    for pi in order:
        prow = positives[pi]
        unit = unit_of(prow, level)
        pool = by_unit.get(unit)
        if not pool:
            n_no_unit += 1
            continue
        avail = [c for c in pool if not used[c]]
        if not avail:
            n_no_candidate += 1
            continue
        pz = pos_z[pi]
        if not np.all(np.isfinite(pz)):
            n_no_candidate += 1
            continue
        cz = cand_z[avail]
        d = np.linalg.norm(cz - pz[None, :], axis=1)
        d = np.where(np.isfinite(d), d, np.inf)
        best = int(np.argmin(d))
        if not np.isfinite(d[best]) or d[best] > caliper:
            n_outside_caliper += 1
            continue
        ci = avail[best]
        used[ci] = True
        crow = candidates[ci]
        pairs.append({
            "positive_filename": prow["filename"], "positive_path": prow.get("path", ""),
            "control_filename": crow["filename"], "control_path": crow.get("path", ""),
            "unit": unit, "unit_level": level,
            "distance_sd": float(d[best]),
            **{f"positive_{f}": float(pos_mat[pi, j]) for j, f in enumerate(match_features)},
            **{f"control_{f}": float(cand_mat[ci, j]) for j, f in enumerate(match_features)},
        })

    pairs.sort(key=lambda r: str(r["positive_filename"]))
    diagnostics = {
        "n_positives": len(positives),
        "n_candidates": len(candidates),
        "n_matched": len(pairs),
        "n_unmatched_no_unit": n_no_unit,
        "n_unmatched_pool_exhausted": n_no_candidate,
        "n_unmatched_outside_caliper": n_outside_caliper,
        "mean_distance_sd": float(np.mean([p["distance_sd"] for p in pairs])) if pairs else float("nan"),
        "max_distance_sd": float(np.max([p["distance_sd"] for p in pairs])) if pairs else float("nan"),
    }
    return pairs, diagnostics


# --------------------------------------------------------------------------- #
def balance_rows(
    positives: Sequence[Dict[str, str]],
    candidates: Sequence[Dict[str, str]],
    matched_pos: Sequence[Dict[str, str]],
    matched_ctl: Sequence[Dict[str, str]],
    resolution: str,
    task: str = "",
) -> List[Dict[str, object]]:
    """SMD of each balance feature, before matching and after."""
    rows: List[Dict[str, object]] = []
    for feat in BALANCE_FEATURES:
        col = column_for(feat, resolution)
        if not positives or col not in positives[0]:
            continue
        before = smd(values(positives, feat, resolution), values(candidates, feat, resolution))
        after = smd(values(matched_pos, feat, resolution), values(matched_ctl, feat, resolution))
        rows.append({
            "task": task,
            "feature": feat,
            "label": FEATURE_LABEL.get(feat, feat),
            "mean_positive_before": float(np.nanmean(values(positives, feat, resolution))),
            "mean_control_before": float(np.nanmean(values(candidates, feat, resolution))),
            "smd_before": before,
            "mean_positive_after": float(np.nanmean(values(matched_pos, feat, resolution))) if matched_pos else float("nan"),
            "mean_control_after": float(np.nanmean(values(matched_ctl, feat, resolution))) if matched_ctl else float("nan"),
            "smd_after": after,
            "balanced_after": int(abs(after) < 0.1) if np.isfinite(after) else 0,
        })
    return rows


def site_overlap(pos: Sequence[Dict[str, str]], ctl: Sequence[Dict[str, str]],
                 level: str) -> Dict[str, object]:
    """Jaccard overlap of the acquisition units used by the two classes."""
    a = {unit_of(r, level) for r in pos}
    b = {unit_of(r, level) for r in ctl}
    inter, union = a & b, a | b
    return {
        "n_units_positive": len(a), "n_units_control": len(b),
        "n_units_shared": len(inter),
        "jaccard": (len(inter) / len(union)) if union else float("nan"),
    }


def balance_table(task: str, brows: Sequence[Dict[str, object]],
                  diag: Dict[str, object], ov_before: Dict[str, object],
                  ov_after: Dict[str, object], level: str) -> str:
    header = ["Quantity", "Positive (mean)", "Control (mean)", "SMD before",
              "SMD after", "Balanced ($|$SMD$| < 0.1$)"]
    rows: List[List[str]] = []
    for r in brows:
        big = abs(float(r["mean_positive_after" if np.isfinite(float(r["mean_positive_after"]))
                                                 else "mean_positive_before"])) >= 1000
        dec = 0 if big else 3
        rows.append([
            str(r["label"]),
            latex.fmt(r["mean_positive_after"], dec),
            latex.fmt(r["mean_control_after"], dec),
            latex.fmt(r["smd_before"]),
            latex.fmt(r["smd_after"]),
            "yes" if r["balanced_after"] else "no",
        ])
    mid = [len(rows) - 1]
    unit_word = "sites" if level == "site" else "hive proxies"
    rows.append([f"Shared {unit_word}", str(ov_after["n_units_positive"]),
                 str(ov_after["n_units_control"]),
                 latex.fmt(ov_before["jaccard"]), latex.fmt(ov_after["jaccard"]), "--"])
    rows.append(["Matched pairs per class", str(diag["n_matched"]), str(diag["n_matched"]),
                 "--", "--", "--"])
    rows.append(["Mean $|$distance$|$ (pooled SD)", latex.fmt(diag["mean_distance_sd"]),
                 "--", "--", "--", "--"])
    return latex.tabular(
        header, rows, align="lccccc", midrules_after=mid,
        notes=[f"Acquisition-matched subset for {task_label(task)}: 1:1 nearest-neighbour "
               f"matching without replacement within {unit_word}.",
               "SMD = (mean$_{\\mathrm{positive}}$ $-$ mean$_{\\mathrm{control}}$) / pooled SD; "
               "'before' uses the full candidate pool, 'after' the matched subset.",
               "'Positive/Control (mean)' columns describe the matched subset.",
               "Overlap columns report the Jaccard index of the acquisition units "
               "used by the two classes."],
    )


def write_csv(path: str, rows: Sequence[Dict[str, object]],
              columns: Optional[Sequence[str]] = None) -> str:
    if not rows:
        return path
    cols = list(columns) if columns else list(rows[0])
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

    pos_rows = [r for r in load_features(args.positive_features) if int(r["label"]) == 1]
    cand_all = load_features(args.candidate_features)
    cand_rows = [r for r in cand_all if int(r["label"]) == 0]
    if not pos_rows:
        print(f"[error] no positive (label=1) rows in {args.positive_features}")
        return 2
    if not cand_rows:
        print(f"[error] no normal (label=0) rows in {args.candidate_features}")
        return 2

    # (a) restrict candidates to the acquisition units present among positives
    pos_units = {unit_of(r, args.match_level) for r in pos_rows}
    restricted = [r for r in cand_rows if unit_of(r, args.match_level) in pos_units]
    print(f"[{args.task}] positives={len(pos_rows)}  candidate normals={len(cand_rows)}")
    print(f"  {args.match_level}s among positives: {len(pos_units)}; "
          f"candidates in those {args.match_level}s: {len(restricted)}")
    if not restricted:
        print(f"[error] no candidate normals share a {args.match_level} with the positives; "
              "the classes are fully confounded at this level and no matched subset "
              "exists. Try --match-level site, or a larger normal pool.")
        return 2

    # (b) match
    try:
        pairs, diag = match(pos_rows, restricted, args.match_features, args.resolution,
                            args.match_level, args.caliper, args.seed)
    except ValueError as exc:
        print(f"[error] {exc}")
        return 2

    print(f"  matched {diag['n_matched']} pairs "
          f"(caliper {args.caliper} SD on {'+'.join(args.match_features)})")
    print(f"    unmatched: no {args.match_level} {diag['n_unmatched_no_unit']}, "
          f"pool exhausted {diag['n_unmatched_pool_exhausted']}, "
          f"outside caliper {diag['n_unmatched_outside_caliper']}")
    if pairs:
        print(f"    distance in pooled SD: mean {diag['mean_distance_sd']:.4f}, "
              f"max {diag['max_distance_sd']:.4f}")

    pos_by_name = {r["filename"]: r for r in pos_rows}
    ctl_by_name = {r["filename"]: r for r in restricted}
    matched_pos = [pos_by_name[str(p["positive_filename"])] for p in pairs]
    matched_ctl = [ctl_by_name[str(p["control_filename"])] for p in pairs]

    # (c) balance diagnostics
    brows = balance_rows(pos_rows, restricted, matched_pos, matched_ctl,
                         args.resolution, args.task)
    ov_before = site_overlap(pos_rows, cand_rows, args.match_level)
    ov_after = site_overlap(matched_pos, matched_ctl, args.match_level)
    print(f"  {args.match_level} overlap (Jaccard): before {ov_before['jaccard']:.3f}, "
          f"after {ov_after['jaccard']:.3f}")
    print(f"  {'feature':<22s} {'SMD before':>11s} {'SMD after':>10s}")
    for r in brows:
        print(f"  {r['feature']:<22s} {r['smd_before']:>11.4f} {r['smd_after']:>10.4f}")

    enough = diag["n_matched"] >= args.n_per_class  # type: ignore[operator]
    if not enough:
        print(f"\n[warn] matched {diag['n_matched']} pairs per class, "
              f"below the required --n-per-class {args.n_per_class}. "
              "Widen --caliper, use --match-level site, enlarge the normal pool, "
              "or pass --allow-fewer to accept this subset.")

    if args.dry_run:
        print("[dry-run] nothing written.")
        return 0 if (enough or args.allow_fewer) else 1

    subset: List[Dict[str, object]] = []
    for p in pairs:
        subset.append({"filename": p["positive_filename"], "label": 1,
                       "path": p["positive_path"], "unit": p["unit"],
                       "match_group": p["positive_filename"]})
        subset.append({"filename": p["control_filename"], "label": 0,
                       "path": p["control_path"], "unit": p["unit"],
                       "match_group": p["positive_filename"]})
    subset.sort(key=lambda r: (str(r["match_group"]), int(r["label"])))

    if subset:
        print("\nwrote", write_csv(os.path.join(args.out_dir, f"matched_subset_{args.task}.csv"),
                                   subset, ["filename", "label", "path", "unit", "match_group"]))
        print("wrote", write_csv(os.path.join(args.out_dir, f"matched_pairs_{args.task}.csv"), pairs))
    else:
        print("\n[warn] no pairs matched: no subset CSV written. The balance table "
              "still records the (unmatched) 'before' state.")
    print("wrote", write_csv(os.path.join(args.out_dir, f"matched_balance_{args.task}.csv"), brows))
    print("wrote", latex.write_fragment(
        os.path.join(args.table_dir, "matched_subset_balance.tex"),
        balance_table(args.task, brows, diag, ov_before, ov_after, args.match_level)))

    if not enough and not args.allow_fewer:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
