#!/usr/bin/env python3
"""Near-duplicate audit via perceptual hashing, with a threshold sweep and exact-duplicate check.

Two images are treated as near-duplicates when the Hamming distance between
their 64-bit pHashes (from ``extract_image_features.py``) is at most
``--threshold``.

Choosing the threshold
----------------------
The default is **4**, not 8. On real honeybee comb imagery a threshold of 8
links a large majority of the corpus across *different* hives: the frames are
dark, low-texture and visually near-uniform, so their DCT low-frequency blocks
collide and pHash stops discriminating. A threshold that flags ~90% of pairs
measures the hash's failure, not duplication.

Because no single cut-off is defensible a priori, this script **always** reports
a sweep over ``--sweep-thresholds`` (default ``0 2 4 6 8``) so the reader can see
how the conclusion moves with the cut-off. The key diagnostic is the split
between **same-hive** pairs (plausible duplicates: consecutive frames of one
comb) and **different-hive** pairs (implausible: distinct colonies should not
produce identical images). A threshold at which different-hive pairs dominate is
too loose by construction.

Exact duplicates
----------------
Independently of pHash, images whose **MD5 of the raw file bytes** matches are
byte-identical: the same file present twice. These are reported separately, are
threshold-independent, and are the only unambiguous duplicates in the corpus.

Outputs
-------
``results/near_duplicate_sweep_<task>.csv``
    One row per threshold: pair counts (total / same-hive / different-hive /
    same-class / cross-class), images removed by de-duplication, and per-fold
    train/test crossings.
``results/near_duplicates_<task>.csv``
    Pair-level detail at the operating threshold (``--threshold``).
``results/near_duplicate_components_<task>.csv``
    Connected-component id and representative for every image, at ``--threshold``.
``results/exact_duplicates_<task>.csv``
    MD5 groups containing more than one file.
``results/split_manifest_dedup_<task>.csv``
    The manifest filtered to one representative per connected component, for
    re-running CV without near-duplicates.
``tables/near_duplicates.tex``
    The sweep, thresholds as rows, both tasks side by side.

Examples
--------
    python near_duplicates.py --features-dir ./results --manifest-dir ./results
    python near_duplicates.py --threshold 2 --sweep-thresholds 0 1 2 4 8
    python near_duplicates.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from common import latex
from common.data import TASKS, MANIFEST_COLUMNS, read_manifest, task_label

DEFAULT_THRESHOLD = 4
DEFAULT_SWEEP = (0, 2, 4, 6, 8)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features-dir", default="./results")
    p.add_argument("--manifest-dir", default="./results")
    p.add_argument("--out-dir", default="./results")
    p.add_argument("--table-dir", default="./tables")
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                   help=f"Operating pHash Hamming threshold for the pair/component "
                        f"outputs and the dedup manifest (default {DEFAULT_THRESHOLD}; "
                        "8 is too loose on this imagery -- see the module docstring).")
    p.add_argument("--sweep-thresholds", nargs="+", type=int, default=list(DEFAULT_SWEEP),
                   help="Thresholds reported in the sweep (default: "
                        f"{' '.join(map(str, DEFAULT_SWEEP))}).")
    p.add_argument("--manifest-prefix", default="split_manifest",
                   help="Manifest basename prefix (use split_manifest_grouped for grouped CV).")
    p.add_argument("--max-pairs-csv", type=int, default=20000,
                   help="Cap the number of pair rows written (the summary counts are exact).")
    p.add_argument("--no-exact", action="store_true",
                   help="Skip the MD5 exact-duplicate check (e.g. if the column is absent).")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute and print the sweep; write nothing.")
    return p


# --------------------------------------------------------------------------- #
# pHash plumbing
# --------------------------------------------------------------------------- #
def hex_to_bits(hexstr: str) -> np.ndarray:
    """16 hex chars -> uint8 array of 64 bits (MSB first)."""
    v = int(hexstr, 16)
    return np.array([(v >> (63 - i)) & 1 for i in range(64)], dtype=np.uint8)


def all_pairs_by_distance(bits: np.ndarray, max_threshold: int,
                          block: int = 512) -> List[Tuple[int, int, int]]:
    """Every pair (i < j) with Hamming distance <= ``max_threshold``, computed once.

    The sweep then filters this single list per threshold instead of recomputing
    the O(n^2) distance matrix for each cut-off.
    """
    out: List[Tuple[int, int, int]] = []
    n = bits.shape[0]
    if n == 0:
        return out
    b16 = bits.astype(np.int16)
    for s in range(0, n, block):
        e = min(s + block, n)
        d = (b16[s:e, None, :] != b16[None, :, :]).sum(axis=2)  # (m, n)
        for local_i in range(e - s):
            i = s + local_i
            row = d[local_i, i + 1:]
            for k in np.where(row <= max_threshold)[0]:
                out.append((i, i + 1 + int(k), int(row[k])))
    return out


class _DSU:
    """Minimal disjoint-set union for connected components of duplicate pairs."""

    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def components(names: Sequence[str], pairs: Sequence[Tuple[int, int, int]]):
    """``(representative_by_name, component_rows, n_multi_member_components, n_components)``."""
    dsu = _DSU(len(names))
    for i, j, _ in pairs:
        dsu.union(i, j)
    comp: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(names)):
        comp[dsu.find(idx)].append(idx)
    representative: Dict[str, str] = {}
    rows: List[Dict[str, object]] = []
    for cid, members in comp.items():
        member_names = sorted(names[m] for m in members)
        rep = member_names[0]
        for m in member_names:
            representative[m] = rep
            rows.append({"filename": m, "component_id": cid,
                         "component_size": len(members), "representative": rep,
                         "is_representative": int(m == rep)})
    n_multi = sum(1 for members in comp.values() if len(members) > 1)
    return representative, rows, n_multi, len(comp)


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def load_features(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def roles_by_fold(manifest_rows: Sequence[Dict[str, str]]) -> Dict[int, Dict[str, str]]:
    """``fold -> {filename: role}``."""
    out: Dict[int, Dict[str, str]] = defaultdict(dict)
    for r in manifest_rows:
        out[int(r["fold"])][r["filename"]] = r["role"]
    return dict(out)


def write_csv(path: str, rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns))
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})
    return path


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def exact_duplicate_groups(features: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    """MD5 groups with more than one member (byte-identical files)."""
    by_md5: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in features:
        digest = (r.get("md5") or "").strip()
        if digest:
            by_md5[digest].append(r)
    rows: List[Dict[str, object]] = []
    for digest, members in sorted(by_md5.items()):
        if len(members) < 2:
            continue
        labels = {m["label"] for m in members}
        hives = {m.get("hive_proxy", "") for m in members}
        rows.append({
            "md5": digest,
            "n_files": len(members),
            "files": ";".join(sorted(m["filename"] for m in members)),
            "same_label": int(len(labels) == 1),
            "same_hive_proxy": int(len(hives) == 1),
        })
    return rows


def sweep_row(
    threshold: int,
    names: Sequence[str],
    labels: Dict[str, int],
    hives: Dict[str, str],
    all_pairs: Sequence[Tuple[int, int, int]],
    fold_roles: Dict[int, Dict[str, str]],
) -> Dict[str, object]:
    """One sweep row: pair counts split by hive/class, dedup impact, fold crossings."""
    pairs = [(i, j, d) for (i, j, d) in all_pairs if d <= threshold]
    same_hive = diff_hive = same_class = cross_class = 0
    crossings = {k: 0 for k in sorted(fold_roles)}
    for i, j, _ in pairs:
        ni, nj = names[i], names[j]
        if hives.get(ni, "") == hives.get(nj, ""):
            same_hive += 1
        else:
            diff_hive += 1
        if labels[ni] == labels[nj]:
            same_class += 1
        else:
            cross_class += 1
        for k, roles in fold_roles.items():
            ra, rb = roles.get(ni), roles.get(nj)
            if (ra in ("train", "val") and rb == "test") or \
               (rb in ("train", "val") and ra == "test"):
                crossings[k] += 1

    _, _, n_multi, n_components = components(names, pairs)
    n_removed = len(names) - n_components
    total = len(pairs)
    return {
        "threshold": threshold,
        "n_images": len(names),
        "n_pairs": total,
        "n_pairs_same_hive": same_hive,
        "n_pairs_diff_hive": diff_hive,
        "frac_pairs_diff_hive": (diff_hive / total) if total else float("nan"),
        "n_pairs_same_class": same_class,
        "n_pairs_cross_class": cross_class,
        "n_components": n_components,
        "n_multi_components": n_multi,
        "n_removed_by_dedup": n_removed,
        "frac_images_removed": (n_removed / len(names)) if names else float("nan"),
        **{f"crossings_fold{k}": v for k, v in crossings.items()},
        "crossings_total": sum(crossings.values()),
    }


def analyse_task(
    task: str,
    features: List[Dict[str, str]],
    manifest_rows: Optional[List[Dict[str, str]]],
    threshold: int,
    sweep_thresholds: Sequence[int],
    check_exact: bool,
) -> Dict[str, object]:
    """Sweep + operating-threshold detail + exact-duplicate groups for one task."""
    names = [r["filename"] for r in features]
    labels = {r["filename"]: int(r["label"]) for r in features}
    hives = {r["filename"]: r.get("hive_proxy", "") for r in features}
    bits = (np.stack([hex_to_bits(r["phash_hex"]) for r in features])
            if features else np.zeros((0, 64), np.uint8))

    fold_roles = roles_by_fold(manifest_rows) if manifest_rows else {}
    folds = sorted(fold_roles)

    thresholds = sorted(set(list(sweep_thresholds) + [threshold]))
    all_pairs = all_pairs_by_distance(bits, max(thresholds) if thresholds else 0)
    sweep = [sweep_row(t, names, labels, hives, all_pairs, fold_roles) for t in thresholds]

    # operating-threshold detail
    op_pairs = [(i, j, d) for (i, j, d) in all_pairs if d <= threshold]
    pair_rows: List[Dict[str, object]] = []
    for i, j, d in op_pairs:
        ni, nj = names[i], names[j]
        row: Dict[str, object] = {
            "file_a": ni, "file_b": nj, "hamming": d,
            "same_label": int(labels[ni] == labels[nj]),
            "same_hive_proxy": int(hives.get(ni, "") == hives.get(nj, "")),
        }
        for k in folds:
            ra, rb = fold_roles[k].get(ni), fold_roles[k].get(nj)
            row[f"crosses_fold{k}"] = int(
                (ra in ("train", "val") and rb == "test")
                or (rb in ("train", "val") and ra == "test")
            )
        pair_rows.append(row)

    representative, comp_rows, n_multi, n_components = components(names, op_pairs)
    keep = {n for n, rep in representative.items() if n == rep}
    exact = exact_duplicate_groups(features) if check_exact else []
    has_md5 = bool(features) and bool((features[0].get("md5") or "").strip())

    return {
        "task": task, "n_images": len(names), "threshold": threshold,
        "sweep": sweep, "pair_rows": pair_rows, "comp_rows": comp_rows,
        "keep": keep, "folds": folds,
        "n_components": n_components, "n_multi_components": n_multi,
        "n_removed_by_dedup": len(names) - len(keep),
        "exact_groups": exact,
        "n_exact_groups": len(exact),
        "n_exact_extra_files": sum(int(g["n_files"]) - 1 for g in exact),
        "has_md5": has_md5,
    }


# --------------------------------------------------------------------------- #
# Table
# --------------------------------------------------------------------------- #
def sweep_table(results: Sequence[Dict[str, object]]) -> str:
    """Thresholds as rows, tasks side by side."""
    tasks = [str(r["task"]) for r in results]
    thresholds = sorted({int(row["threshold"])
                         for r in results for row in r["sweep"]})  # type: ignore[index]
    per_metric = [
        ("Near-duplicate pairs", "n_pairs", 0),
        ("\\quad same hive proxy", "n_pairs_same_hive", 0),
        ("\\quad different hive proxy", "n_pairs_diff_hive", 0),
        ("\\quad \\% different hive", "frac_pairs_diff_hive", 1),
        ("Images removed by dedup.", "n_removed_by_dedup", 0),
        ("\\quad \\% of corpus", "frac_images_removed", 1),
        ("Train/test crossings (all folds)", "crossings_total", 0),
    ]
    header = ["$d_H \\le$", "Quantity"] + [latex.esc(task_label(t)) for t in tasks]
    rows: List[List[str]] = []
    mid: List[int] = []
    for ti, t in enumerate(thresholds):
        for label, key, kind in per_metric:
            cells = [str(t) if label == per_metric[0][0] else "", label]
            for r in results:
                row = next((x for x in r["sweep"] if int(x["threshold"]) == t), None)  # type: ignore[index]
                if row is None:
                    cells.append("--")
                elif kind == 1:
                    v = float(row[key])
                    cells.append("--" if not np.isfinite(v) else latex.fmt(100.0 * v, 1))
                else:
                    cells.append(str(row[key]))
            rows.append(cells)
        mid.append(len(rows) - 1)

    # exact duplicates: threshold-independent, so one trailing block
    exact_cells = ["--", "Exact (MD5) duplicate groups"]
    extra_cells = ["", "\\quad redundant files"]
    for r in results:
        if r.get("has_md5"):
            exact_cells.append(str(r["n_exact_groups"]))
            extra_cells.append(str(r["n_exact_extra_files"]))
        else:
            exact_cells.append("n/a")
            extra_cells.append("n/a")
    rows.append(exact_cells)
    rows.append(extra_cells)

    return latex.tabular(
        header, rows, align="ll" + "r" * len(tasks), midrules_after=mid,
        notes=["Sensitivity of the near-duplicate audit to the pHash Hamming "
               "threshold $d_H$ (64-bit DCT perceptual hashes).",
               "A 'train/test crossing' is a duplicate pair with one member in a "
               "fold's train or val portion and the other in that fold's held-out "
               "test portion, summed over folds.",
               "Different-hive pairs are implausible as genuine duplicates: a high "
               "share of them indicates the threshold is too loose for this "
               "low-texture imagery rather than that the corpus is redundant.",
               "Exact duplicates are byte-identical files (MD5 of the raw bytes) and "
               "are independent of $d_H$."],
    )


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    results: List[Dict[str, object]] = []

    for task in args.tasks:
        fpath = os.path.join(args.features_dir, f"features_{task}.csv")
        if not os.path.exists(fpath):
            print(f"[warn] missing {fpath}; run extract_image_features.py first")
            continue
        features = load_features(fpath)
        mpath = os.path.join(args.manifest_dir, f"{args.manifest_prefix}_{task}.csv")
        manifest_rows = read_manifest(mpath) if os.path.exists(mpath) else None
        if manifest_rows is None:
            print(f"[warn] no manifest at {mpath}; skipping the train/test crossing analysis")

        res = analyse_task(task, features, manifest_rows, args.threshold,
                           args.sweep_thresholds, not args.no_exact)
        results.append(res)

        print(f"\n[{task}] N={res['n_images']}  (operating threshold d_H <= {args.threshold})")
        print(f"  {'d_H':>4} {'pairs':>8} {'same-hive':>10} {'diff-hive':>10} "
              f"{'%diff':>7} {'removed':>8} {'crossings':>10}")
        for row in res["sweep"]:  # type: ignore[index]
            frac = float(row["frac_pairs_diff_hive"])
            fs = "n/a" if not np.isfinite(frac) else f"{100.0 * frac:.1f}"
            print(f"  {row['threshold']:>4} {row['n_pairs']:>8} {row['n_pairs_same_hive']:>10} "
                  f"{row['n_pairs_diff_hive']:>10} {fs:>7} {row['n_removed_by_dedup']:>8} "
                  f"{row['crossings_total']:>10}")
        if res["has_md5"]:
            print(f"  exact (MD5) duplicate groups: {res['n_exact_groups']} "
                  f"({res['n_exact_extra_files']} redundant files)")
        else:
            print("  exact (MD5) check skipped: no 'md5' column in the features CSV "
                  "(re-run extract_image_features.py)")

        if args.dry_run:
            continue

        sweep_cols = list(res["sweep"][0]) if res["sweep"] else []  # type: ignore[index]
        write_csv(os.path.join(args.out_dir, f"near_duplicate_sweep_{task}.csv"),
                  res["sweep"], sweep_cols)  # type: ignore[arg-type]
        pair_cols = ["file_a", "file_b", "hamming", "same_label", "same_hive_proxy"] + \
                    [f"crosses_fold{k}" for k in res["folds"]]  # type: ignore[index]
        write_csv(os.path.join(args.out_dir, f"near_duplicates_{task}.csv"),
                  res["pair_rows"][: args.max_pairs_csv], pair_cols)  # type: ignore[index]
        write_csv(os.path.join(args.out_dir, f"near_duplicate_components_{task}.csv"),
                  res["comp_rows"],  # type: ignore[arg-type]
                  ["filename", "component_id", "component_size", "representative",
                   "is_representative"])
        if res["exact_groups"]:
            write_csv(os.path.join(args.out_dir, f"exact_duplicates_{task}.csv"),
                      res["exact_groups"],  # type: ignore[arg-type]
                      ["md5", "n_files", "files", "same_label", "same_hive_proxy"])
        if manifest_rows is not None:
            dedup = [r for r in manifest_rows if r["filename"] in res["keep"]]  # type: ignore[operator]
            write_csv(os.path.join(args.out_dir, f"split_manifest_dedup_{task}.csv"),
                      dedup, MANIFEST_COLUMNS)
            print(f"  dedup manifest (d_H <= {args.threshold}) kept "
                  f"{len(dedup)}/{len(manifest_rows)} manifest rows")

    if results and not args.dry_run:
        print("\nwrote", latex.write_fragment(
            os.path.join(args.table_dir, "near_duplicates.tex"), sweep_table(results)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
