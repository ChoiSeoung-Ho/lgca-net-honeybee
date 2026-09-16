#!/usr/bin/env python3
"""Seed-variance experiment on a single, explicitly defined held-out split.

The held-out split (stated here so it is unambiguous in the manuscript)
-----------------------------------------------------------------------
* **Test**  = the held-out *test* portion of **fold 0** of the stratified 3-fold
  split (``StratifiedKFold(3, shuffle=True, random_state=42)``). It is fixed
  across every seed and every model.
* **Validation** = a **stratified 25%** carve-out of fold 0's training portion
  (``train_test_split(stratify=y, random_state=42)``). Also fixed across seeds.
* **Train** = the remaining 75% of fold 0's training portion.

Seeds ``0..4`` (``--seeds``) vary **only** weight initialisation, optimiser
stochasticity and batch shuffling — never the data partition. That isolates
training variance from split variance, which is the point of the table.

Training hyper-parameters match ``train_cv.py``: Adam, batch 32, <=50 epochs,
early stopping on validation loss with patience 10, unweighted cross-entropy,
256x256 inputs in [0, 1], no augmentation. The learning rate comes from
``results/lr_selected_<task>.csv`` when present, else the per-model default.

Outputs
-------
``results/seeds_<task>.csv``
    ``model, seed, BA, sensitivity, specificity, precision, F1, AUROC, loss``
    (consumed directly by ``seed_stats.py``).
``results/seed_preds_<task>_<model>_seed<s>.csv``
    Per-image predictions for every seed.
``model_save/<task>/seeds/<model>_seed<s>.pt``
    Per-seed checkpoints (unless ``--no-save``).

Examples
--------
    python train_seeds.py --task you_chalk_brood --models lgca_net resnet50
    python train_seeds.py --task you_foulbrood --seeds 0 1 2 3 4
    python train_seeds.py --task you_chalk_brood --models lgca_net --dry-run
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

from common.data import TASKS, read_manifest
from common.metrics import compute_metrics
from common.latex import tagged
from common.models import MODEL_REGISTRY, default_lr


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--task", default="you_chalk_brood", choices=list(TASKS))
    p.add_argument("--models", nargs="+", default=["lgca_net"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--fold", type=int, default=0,
                   help="Which CV fold defines the fixed held-out split (default 0).")
    p.add_argument("--manifest-dir", default="./results")
    p.add_argument("--manifest", default=None)
    p.add_argument("--out-dir", default="./results")
    p.add_argument("--ckpt-dir", default="./model_save")
    p.add_argument("--lr", type=float, default=None,
                   help="Override the learning rate for every model.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--run-tag", default="",
                   help="Suffix appended to every output file stem for this run "
                        "(e.g. 'matched'); seed_stats.py --run-tag reads it back.")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-save", action="store_true", help="Do not write checkpoints.")
    p.add_argument("--dry-run", action="store_true",
                   help="2 epochs, 2 seeds, no checkpoints.")
    return p


def load_selected_lr(path: str) -> Dict[str, float]:
    """``model -> lr`` from ``results/lr_selected_<task>.csv`` (rows flagged selected)."""
    out: Dict[str, float] = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if str(r.get("selected", "0")) in ("1", "True", "true"):
                out[r["model"]] = float(r["lr"])
    return out


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        import torch
        import torch.nn as nn
    except Exception:
        print("PyTorch is not installed; train_seeds.py needs torch>=2.9.0.", file=sys.stderr)
        return 2

    # train_cv provides the shared training loop; importing it keeps the two
    # scripts byte-for-byte consistent in optimiser / early-stopping behaviour.
    from train_cv import run_epoch, train_one
    from common.data import make_dataloader
    from common.data import list_task_images

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"device: {device}")

    mpath = args.manifest or os.path.join(args.manifest_dir, f"split_manifest_{args.task}.csv")
    if not os.path.exists(mpath):
        print(f"[error] manifest not found: {mpath}. Run make_splits.py first.", file=sys.stderr)
        return 2
    manifest = read_manifest(mpath)

    samples = list_task_images(args.data_root, args.task)
    path_by_name = {s.filename: s.path for s in samples}
    for r in manifest:
        path_by_name.setdefault(r["filename"], r.get("path", ""))

    fold_rows = [r for r in manifest if int(r["fold"]) == args.fold]
    parts = {role: [r for r in fold_rows if r["role"] == role] for role in ("train", "val", "test")}
    if not parts["test"] or not parts["train"]:
        print(f"[error] fold {args.fold} has an empty train or test portion.", file=sys.stderr)
        return 2
    print(f"fixed split from fold {args.fold}: "
          f"train={len(parts['train'])} val={len(parts['val'])} test={len(parts['test'])}")
    for role in ("train", "val", "test"):
        pos = sum(1 for r in parts[role] if int(r["label"]) == 1)
        print(f"  {role:<5s} n={len(parts[role]):5d}  positives={pos}")

    def pl(role):
        return ([path_by_name[r["filename"]] for r in parts[role]],
                [int(r["label"]) for r in parts[role]])

    tr_paths, tr_labels = pl("train")
    va_paths, va_labels = pl("val")
    te_paths, te_labels = pl("test")

    selected = load_selected_lr(os.path.join(
        args.out_dir, tagged(f"lr_selected_{args.task}", args.run_tag) + ".csv"))
    seeds = args.seeds[:2] if args.dry_run else args.seeds
    os.makedirs(args.out_dir, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for model_name in args.models:
        if model_name not in MODEL_REGISTRY:
            print(f"[error] unknown model '{model_name}'", file=sys.stderr)
            return 2
        lr = args.lr if args.lr is not None else selected.get(model_name, default_lr(model_name))
        print(f"\n=== {args.task} / {model_name} (lr={lr:g}) ===")
        for seed in seeds:
            print(f"  -- seed {seed}")
            try:
                model, best, _ = train_one(model_name, lr, tr_paths, tr_labels,
                                           va_paths, va_labels, args, device, seed=seed)
            except ImportError as exc:
                print(f"[skip] {model_name}: {exc}", file=sys.stderr)
                break

            te_loader = make_dataloader(te_paths, te_labels, size=args.image_size,
                                        batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, seed=seed)
            _, y, p, li, names = run_epoch(model, te_loader, nn.CrossEntropyLoss(), device)
            m = compute_metrics(y, p, (p >= 0.5).astype(int), loss=li)
            rows.append({
                "task": args.task, "model": model_name, "seed": seed, "lr": lr,
                "BA": m["BA"], "sensitivity": m["Sensitivity"],
                "specificity": m["Specificity"], "precision": m["Precision"],
                "F1": m["F1"], "AUROC": m["AUROC"], "loss": m["Loss"],
                "best_epoch": best["epoch"], "n_test": int(len(y)),
            })
            print(f"     BA={m['BA']:.4f} F1={m['F1']:.4f} AUROC={m['AUROC']:.4f} "
                  f"loss={m['Loss']:.4f}")

            if not args.dry_run:
                pred_csv = os.path.join(args.out_dir, tagged(
                    f"seed_preds_{args.task}_{model_name}_seed{seed}", args.run_tag) + ".csv")
                with open(pred_csv, "w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=["filename", "y_true", "prob_pos",
                                                       "pred", "loss"])
                    w.writeheader()
                    for nm, yt, pp, ll in zip(names, y, p, li):
                        w.writerow({"filename": nm, "y_true": int(yt), "prob_pos": float(pp),
                                    "pred": int(pp >= 0.5), "loss": float(ll)})
                if not args.no_save:
                    d = os.path.join(args.ckpt_dir, args.task, tagged("seeds", args.run_tag))
                    os.makedirs(d, exist_ok=True)
                    torch.save({"model": model_name, "task": args.task, "seed": seed,
                                "lr": lr, "state_dict": model.state_dict()},
                               os.path.join(d, f"{model_name}_seed{seed}.pt"))

    if rows and not args.dry_run:
        out = os.path.join(args.out_dir, tagged(f"seeds_{args.task}", args.run_tag) + ".csv")
        cols = ["task", "model", "seed", "lr", "BA", "sensitivity", "specificity",
                "precision", "F1", "AUROC", "loss", "best_epoch", "n_test"]
        # append when the file already holds other models
        existing: List[Dict[str, str]] = []
        if os.path.exists(out):
            with open(out, newline="", encoding="utf-8-sig") as fh:
                existing = [r for r in csv.DictReader(fh)
                            if (r.get("model"), r.get("seed")) not in
                            {(str(x["model"]), str(x["seed"])) for x in rows}]
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in existing:
                w.writerow({c: r.get(c, "") for c in cols})
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})
        print(f"\nwrote {out} ({len(rows)} new rows). Run seed_stats.py next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
