#!/usr/bin/env python3
"""3-fold cross-validation training with per-model learning-rate selection.

Protocol (identical to the original pipeline except where the revision changed it)
---------------------------------------------------------------------------------
* Optimiser Adam, batch size 32, at most 50 epochs, **unweighted** cross-entropy.
* Images resized to 256x256 and scaled to [0, 1]; **no augmentation**.
* Early stopping on the **validation loss**, patience 10; the best-validation-loss
  checkpoint is restored before scoring the held-out fold.
* Folds come from the manifest written by ``make_splits.py`` (or, with
  ``--file-list``, are built on the fly from an explicit file list such as the
  acquisition-matched subset -- same splitter, same seed):
  ``StratifiedKFold(3, shuffle=True, random_state=42)`` (was un-stratified
  ``KFold``), with a **stratified** 25% validation carve-out of each fold's
  training portion (was "the last 25% of the shuffled training indices").
* Learning rate is selected **once per model** on fold 0's validation split
  (highest validation balanced accuracy at the best-val-loss epoch) and then held
  fixed for every fold, so the choice never sees a test fold.
  Grid: ``1e-3,3e-4,1e-4,3e-5`` for ImageNet-pretrained baselines,
  ``1e-3,3e-4`` for the from-scratch proposed model and its ablations.

Outputs
-------
``model_save/<task>/fold<k>/<model>.pt``      per-fold best checkpoint
``results/oof_<task>_<model>.csv``            per-image out-of-fold predictions
                                              (filename, fold, y_true, prob_pos, pred, loss)
``results/lr_selected_<task>.csv``            selected LR and its fold-0 validation BA
``results/train_log_<task>_<model>.csv``      per-epoch train/val loss and val BA

Examples
--------
    python train_cv.py --data-root ./data --task you_chalk_brood --models lgca_net resnet50
    python train_cv.py --task you_foulbrood --grouped                 # hive-grouped CV
    python train_cv.py --task you_foulbrood --manifest results/split_manifest_dedup_you_foulbrood.csv
    python train_cv.py --task you_chalk_brood --file-list results/matched_subset_you_chalk_brood.csv
    python train_cv.py --task you_chalk_brood --models lgca_net --dry-run
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.data import TASKS, list_task_images, read_manifest
from common.metrics import balanced_accuracy
from common.latex import tagged
from common.models import MODEL_REGISTRY, FROM_SCRATCH, lr_grid_for
from common.seed import set_seed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--task", default="you_chalk_brood", choices=list(TASKS))
    p.add_argument("--models", nargs="+", default=["lgca_net"],
                   help=f"Any of: {', '.join(sorted(MODEL_REGISTRY))}")
    p.add_argument("--manifest", default=None,
                   help="Explicit manifest CSV (e.g. a de-duplicated one). "
                        "Overrides --grouped.")
    p.add_argument("--file-list", default=None,
                   help="CSV with columns filename,label[,path] (e.g. "
                        "results/matched_subset_<task>.csv from "
                        "rebuild_matched_subset.py). Folds are built fresh from this "
                        "list with StratifiedKFold(3, shuffle, seed 42) and a "
                        "stratified 25%% validation carve-out -- the same protocol as "
                        "the full-corpus run. Overrides --manifest; combine with --grouped for hive-grouped folds.")
    p.add_argument("--grouped", action="store_true",
                   help="Use split_manifest_grouped_<task>.csv (hive-proxy-grouped CV).")
    p.add_argument("--manifest-dir", default="./results")
    p.add_argument("--out-dir", default="./results")
    p.add_argument("--ckpt-dir", default="./model_save")
    p.add_argument("--lr-grid", default=None,
                   help="Comma-separated LR grid overriding the per-model default "
                        "(pretrained: 1e-3,3e-4,1e-4,3e-5; from-scratch: 1e-3,3e-4).")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-splits", type=int, default=3,
                   help="Folds built when --file-list is used (default 3).")
    p.add_argument("--val-fraction", type=float, default=0.25,
                   help="Stratified validation fraction when --file-list is used.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--run-tag", default="",
                   help="Suffix appended to every output file stem for this run "
                        "(e.g. 'matched', 'original'), so several protocols can "
                        "coexist in one results directory. Empty = current names.")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Build the timm baselines without ImageNet weights.")
    p.add_argument("--dry-run", action="store_true",
                   help="2 epochs, 1 LR, fold 0 only, no checkpoint written.")
    return p


# --------------------------------------------------------------------------- #
def _manifest_path(args) -> str:
    if args.manifest:
        return args.manifest
    prefix = "split_manifest_grouped" if args.grouped else "split_manifest"
    return os.path.join(args.manifest_dir, f"{prefix}_{args.task}.csv")


def _manifest_from_file_list(path: str, seed: int, n_splits: int,
                             val_fraction: float, grouped: bool = False) -> List[Dict[str, str]]:
    """Build an in-memory manifest from a filename,label[,path] CSV.

    Uses exactly the same splitter as ``make_splits.py`` -- StratifiedKFold with
    ``shuffle=True`` and the given seed, plus a stratified validation carve-out --
    so a matched-subset run is protocol-identical to the full-corpus run and only
    the underlying set of images differs.
    """
    from common.data import (Sample, build_manifest_rows, make_stratified_folds,
                             make_grouped_folds, parse_filename)

    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} is empty")
    missing = {"filename", "label"} - set(rows[0])
    if missing:
        raise ValueError(f"{path} lacks required column(s): {', '.join(sorted(missing))}")

    samples = [
        Sample(path=r.get("path", ""), filename=r["filename"], label=int(r["label"]),
               class_dir="abnormal" if int(r["label"]) == 1 else "normal",
               site=r.get("site", r.get("unit", "")),
               hive_proxy=r.get("hive_proxy", r.get("unit", "")),
               session=r.get("session", ""), disease_code=r.get("disease_code"))
        for r in rows
    ]
    samples.sort(key=lambda s: (s.class_dir, s.filename))
    labels = [s.label for s in samples]
    if grouped:
        # Hive-proxy-grouped folds (StratifiedGroupKFold), as in the manuscript.
        groups = []
        for smp in samples:
            g = smp.hive_proxy
            if not g:
                parsed = parse_filename(smp.filename)
                g = parsed.get("hive_proxy", smp.filename) if parsed else smp.filename
            groups.append(g)
        folds = make_grouped_folds([s.path for s in samples], labels, groups,
                                   n_splits=n_splits, seed=seed)
    else:
        folds = make_stratified_folds([s.path for s in samples], labels,
                                      n_splits=n_splits, seed=seed)
    built = build_manifest_rows(samples, folds, val_fraction=val_fraction, seed=seed)
    return [{k: str(v) for k, v in r.items()} for r in built]


def _paths_labels(rows, path_by_name) -> Tuple[List[str], List[int]]:
    return ([path_by_name[r["filename"]] for r in rows],
            [int(r["label"]) for r in rows])


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass. Returns ``(mean_loss, y_true, prob_pos, per_image_loss, names)``."""
    import torch
    import torch.nn.functional as F

    train = optimizer is not None
    model.train(train)
    losses, ys, ps, per_img, names = [], [], [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y, n in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                pi = F.cross_entropy(logits, y, reduction="none")
                prob = torch.softmax(logits, dim=1)[:, 1]
            losses.append(float(loss.detach()) * x.size(0))
            per_img.extend(pi.detach().cpu().tolist())
            ys.extend(y.detach().cpu().tolist())
            ps.extend(prob.detach().cpu().tolist())
            names.extend(list(n))
    n_tot = max(len(ys), 1)
    return sum(losses) / n_tot, np.array(ys), np.array(ps), np.array(per_img), names


def train_one(
    model_name: str, lr: float, train_paths, train_labels, val_paths, val_labels,
    args, device, seed: int, log_csv: Optional[str] = None,
):
    """Train with early stopping on validation loss; return (model, best_state, history)."""
    import torch
    import torch.nn as nn

    from common.data import make_dataloader
    from common.models import build_model

    set_seed(seed)
    model = build_model(model_name, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()  # unweighted, as in the original code
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    tr_loader = make_dataloader(train_paths, train_labels, size=args.image_size,
                                batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, seed=seed)
    va_loader = make_dataloader(val_paths, val_labels, size=args.image_size,
                                batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, seed=seed)

    best = {"val_loss": float("inf"), "epoch": -1, "state": None, "val_ba": float("nan")}
    history: List[Dict[str, float]] = []
    bad = 0
    epochs = 2 if args.dry_run else args.epochs

    for epoch in range(epochs):
        t0 = time.time()
        tr_loss, *_ = run_epoch(model, tr_loader, criterion, device, optimizer)
        va_loss, y, p, _, _ = run_epoch(model, va_loader, criterion, device)
        va_ba = balanced_accuracy(y, (p >= 0.5).astype(int))
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
                        "val_ba": va_ba, "seconds": time.time() - t0})
        print(f"    epoch {epoch:3d}  train {tr_loss:.4f}  val {va_loss:.4f}  "
              f"val BA {va_ba:.4f}  ({time.time() - t0:.1f}s)")

        if va_loss < best["val_loss"] - 1e-6:
            best = {"val_loss": va_loss, "epoch": epoch, "val_ba": va_ba,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"    early stop at epoch {epoch} "
                      f"(best epoch {best['epoch']}, val loss {best['val_loss']:.4f})")
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    if log_csv and history:
        os.makedirs(os.path.dirname(os.path.abspath(log_csv)) or ".", exist_ok=True)
        with open(log_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(history[0]))
            w.writeheader()
            w.writerows(history)
    return model, best, history


def select_lr(model_name: str, fold0, path_by_name, args, device) -> Tuple[float, float, List[Dict]]:
    """Pick the LR maximising fold-0 validation BA at the best-val-loss epoch."""
    if args.lr_grid:
        grid = tuple(float(v) for v in args.lr_grid.split(","))
    else:
        grid = lr_grid_for(model_name)
    if args.dry_run:
        grid = grid[:1]

    tr_paths, tr_labels = _paths_labels(fold0["train"], path_by_name)
    va_paths, va_labels = _paths_labels(fold0["val"], path_by_name)

    trials = []
    best_lr, best_ba = grid[0], -1.0
    for lr in grid:
        print(f"  [lr search] {model_name} lr={lr:g}")
        _, best, _ = train_one(model_name, lr, tr_paths, tr_labels, va_paths, va_labels,
                               args, device, seed=args.seed)
        ba = float(best["val_ba"])
        trials.append({"model": model_name, "lr": lr, "val_ba": ba,
                       "val_loss": best["val_loss"], "best_epoch": best["epoch"]})
        print(f"  [lr search] {model_name} lr={lr:g} -> fold-0 val BA {ba:.4f}")
        if np.isfinite(ba) and ba > best_ba:
            best_lr, best_ba = lr, ba
    return best_lr, best_ba, trials


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        import torch
    except Exception:
        print("PyTorch is not installed. train_cv.py needs torch>=2.9.0 (+ timm for the "
              "baselines). The CPU-only analysis scripts run without it.", file=sys.stderr)
        return 2

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"device: {device}")

    if args.file_list:
        if not os.path.exists(args.file_list):
            print(f"[error] file list not found: {args.file_list}", file=sys.stderr)
            return 2
        try:
            manifest = _manifest_from_file_list(args.file_list, args.seed,
                                                args.n_splits, args.val_fraction,
                                                grouped=args.grouped)
        except ValueError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 2
        n_img = len({r["filename"] for r in manifest})
        n_pos = len({r["filename"] for r in manifest if int(r["label"]) == 1})
        print(f"file list: {args.file_list} -> {n_img} images "
              f"({n_img - n_pos} normal / {n_pos} abnormal), folds built with "
              f"StratifiedKFold({args.n_splits}, shuffle, seed {args.seed})")
        if not args.dry_run:
            os.makedirs(args.out_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(args.file_list))[0]
            out = os.path.join(args.out_dir, tagged(f"split_manifest_{stem}", args.run_tag) + ".csv")
            from common.data import write_manifest
            write_manifest(manifest, out)
            print(f"  split manifest for this run -> {out}")
    else:
        mpath = _manifest_path(args)
        if not os.path.exists(mpath):
            print(f"[error] manifest not found: {mpath}. Run make_splits.py first.",
                  file=sys.stderr)
            return 2
        manifest = read_manifest(mpath)
        print(f"manifest: {mpath} ({len(manifest)} rows)")

    samples = list_task_images(args.data_root, args.task)
    path_by_name = {s.filename: s.path for s in samples}
    # A manifest/file list may point outside the task folder (the matched subset
    # draws its controls from the full normal pool), so an explicit non-empty
    # path in the manifest always wins over the task-folder scan.
    for r in manifest:
        p_ = (r.get("path") or "").strip()
        if p_:
            path_by_name[r["filename"]] = p_
        else:
            path_by_name.setdefault(r["filename"], "")
    missing = sorted({r["filename"] for r in manifest if not path_by_name.get(r["filename"])})
    if missing:
        print(f"[error] {len(missing)} manifest entries have no resolvable image path "
              f"(first: {missing[0]}). Check --data-root or the 'path' column.",
              file=sys.stderr)
        return 2

    folds = sorted({int(r["fold"]) for r in manifest})
    if args.dry_run:
        folds = folds[:1]
    by_fold = {
        k: {role: [r for r in manifest if int(r["fold"]) == k and r["role"] == role]
            for role in ("train", "val", "test")}
        for k in folds
    }

    os.makedirs(args.out_dir, exist_ok=True)
    lr_rows: List[Dict[str, object]] = []

    for model_name in args.models:
        if model_name not in MODEL_REGISTRY:
            print(f"[error] unknown model '{model_name}'", file=sys.stderr)
            return 2
        print(f"\n=== {args.task} / {model_name} "
              f"({'from scratch' if model_name in FROM_SCRATCH else 'ImageNet-pretrained'}) ===")
        try:
            lr, lr_ba, trials = select_lr(model_name, by_fold[folds[0]], path_by_name, args, device)
        except ImportError as exc:
            print(f"[skip] {model_name}: {exc}", file=sys.stderr)
            continue
        print(f"  selected lr={lr:g} (fold-0 val BA {lr_ba:.4f}); fixed for all folds")
        for t in trials:
            lr_rows.append({"task": args.task, "selected": int(t["lr"] == lr), **t})

        oof: List[Dict[str, object]] = []
        for k in folds:
            print(f"  -- fold {k}")
            tr_paths, tr_labels = _paths_labels(by_fold[k]["train"], path_by_name)
            va_paths, va_labels = _paths_labels(by_fold[k]["val"], path_by_name)
            te_paths, te_labels = _paths_labels(by_fold[k]["test"], path_by_name)

            log = os.path.join(args.out_dir,
                               tagged(f"train_log_{args.task}_{model_name}_fold{k}", args.run_tag) + ".csv")
            model, best, _ = train_one(model_name, lr, tr_paths, tr_labels,
                                       va_paths, va_labels, args, device,
                                       seed=args.seed, log_csv=log)

            if not args.dry_run:
                ck_dir = os.path.join(args.ckpt_dir, args.task, tagged(f"fold{k}", args.run_tag))
                os.makedirs(ck_dir, exist_ok=True)
                ck = os.path.join(ck_dir, f"{model_name}.pt")
                torch.save({"model": model_name, "task": args.task, "fold": k, "lr": lr,
                            "best_epoch": best["epoch"], "best_val_loss": best["val_loss"],
                            "state_dict": model.state_dict()}, ck)
                print(f"     checkpoint -> {ck}")

            from common.data import make_dataloader
            import torch.nn as nn

            te_loader = make_dataloader(te_paths, te_labels, size=args.image_size,
                                        batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, seed=args.seed)
            _, y, p, li, names = run_epoch(model, te_loader, nn.CrossEntropyLoss(), device)
            for nm, yt, pp, ll in zip(names, y, p, li):
                oof.append({"filename": nm, "fold": k, "y_true": int(yt),
                            "prob_pos": float(pp), "pred": int(pp >= 0.5), "loss": float(ll)})
            print(f"     fold {k} test BA = {balanced_accuracy(y, (p >= 0.5).astype(int)):.4f}")

        if not args.dry_run and oof:
            out = os.path.join(args.out_dir, tagged(f"oof_{args.task}_{model_name}", args.run_tag) + ".csv")
            with open(out, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=["filename", "fold", "y_true",
                                                   "prob_pos", "pred", "loss"])
                w.writeheader()
                w.writerows(oof)
            print(f"  wrote {out} ({len(oof)} rows)")

    if lr_rows and not args.dry_run:
        out = os.path.join(args.out_dir, tagged(f"lr_selected_{args.task}", args.run_tag) + ".csv")
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["task", "model", "lr", "val_ba",
                                               "val_loss", "best_epoch", "selected"])
            w.writeheader()
            w.writerows(lr_rows)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
