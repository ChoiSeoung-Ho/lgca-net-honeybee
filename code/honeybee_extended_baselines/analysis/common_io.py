# -*- coding: utf-8 -*-
"""Shared paths, CLI and small helpers for the analysis layer."""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd

TASKS = {"you_chalk_brood": "Chalkbrood", "you_foulbrood": "Foulbrood"}

#: The six global image statistics used by the model-free classifiers.
#: Identical on the original and the matched subsets so the two are comparable.
GLOBAL_FEATS = ["lab_L_256", "lab_a_256", "lab_b_256",
                "native_w", "native_h", "file_size_bytes"]

#: Statistics reported in the covariate-balance table (descriptive only).
BALANCE_FEATS = ["lab_a_256", "lab_b_256", "rgb_r_256", "rgb_g_256", "rgb_b_256",
                 "file_size_bytes", "otsu_largest_area_frac"]


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--out-dir", default="./results")
    p.add_argument("--tables-dir", default="./tables")
    p.add_argument("--figures-dir", default="./manuscript")
    p.add_argument("--run-tag", default="original",
                   help="suffix of the oof_/cv_ files to analyse (default: original)")
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument("--seed", type=int, default=20260906)
    return p


def oof_path(results_dir: str, task: str, model: str, run_tag: str) -> str:
    tag = f"_{run_tag}" if run_tag else ""
    return os.path.join(results_dir, f"oof_{task}_{model}{tag}.csv")


def read_oof(results_dir: str, task: str, model: str, run_tag: str) -> pd.DataFrame:
    return pd.read_csv(oof_path(results_dir, task, model, run_tag)).set_index("filename")


def read_features(results_dir: str, task: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(results_dir, f"features_{task}.csv")).set_index(
        "filename", drop=False)


def smd(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    denom = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / denom) if denom > 0 else float("nan")


def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * p[idx])
        adj[idx] = min(1.0, run)
    return adj


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
