#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""How far is each classifier's OWN DECISION reproducible from acquisition statistics?

For every classifier, task and subset, a logistic regression on six global image
statistics is trained -- on the folds of the audited run, read from the saved
out-of-fold prediction files -- to predict the classifier's own decision.  The
reference is the same regression trained to predict the disease label.  A
classifier whose decisions are more predictable from acquisition statistics than
the disease label itself is tracking the acquisition channel more closely than
the acquisition channel tracks the disease.

This is the annotation-free substitute for box-level attribution localisation:
it needs no lesion annotation, it is per model, and it separates architectures
that the attribution maps do not.

    python confound_reliance.py --results-dir ./results --out-dir ./results
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_io import GLOBAL_FEATS, base_parser, ensure_dir, read_features, read_oof  # noqa: E402
from matched_subsets import threshold_by_folds  # noqa: E402
from model_registry import REFERENCE_MODEL, common_models  # noqa: E402

REFERENCE_ROW = "__reference_label__"


def surrogate_oof(X, target, fold):
    """Leave-one-fold-out surrogate, using the audited fold assignment."""
    oof = np.full(len(target), np.nan)
    if len(np.unique(target)) < 2:
        return oof
    for f in np.unique(fold):
        te, tr = fold == f, fold != f
        if len(np.unique(target[tr])) < 2 or te.sum() == 0:
            continue
        m = Pipeline([("s", StandardScaler()),
                      ("c", LogisticRegression(max_iter=5000))]).fit(X[tr], target[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def score_surrogate(target, p):
    ok = ~np.isnan(p)
    if ok.sum() < 10 or len(np.unique(target[ok])) < 2:
        return np.nan, np.nan
    return (roc_auc_score(target[ok], p[ok]),
            balanced_accuracy_score(target[ok], (p[ok] > .5).astype(int)))


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--scopes", nargs="+", default=["original", "matched"])
    a = ap.parse_args()
    ensure_dir(a.out_dir)

    models = common_models(a.results_dir, a.tasks, a.run_tag)
    print(f"[info] {len(models)} models: {', '.join(models)}")

    rows = []
    for task in a.tasks:
        feats = read_features(a.results_dir, task)
        matched_path = os.path.join(a.out_dir, f"matched_{task}.csv")
        matched = set(pd.read_csv(matched_path).filename) if os.path.isfile(matched_path) else set()
        if "matched" in a.scopes and not matched:
            print(f"[warn] {matched_path} missing; run matched_subsets.py first")

        for scope in a.scopes:
            base = read_oof(a.results_dir, task, REFERENCE_MODEL, a.run_tag)
            if scope == "matched":
                if not matched:
                    continue
                base = base[base.index.isin(matched)]
            idx = base.index
            ff = feats.set_index("filename").loc[idx]
            X = ff[GLOBAL_FEATS].to_numpy(float)
            L = ff["lab_L_256"].to_numpy(float)
            fold = base.fold.to_numpy(int)
            y = base.y_true.to_numpy(int)

            auc_l, ba_l = score_surrogate(y, surrogate_oof(X, y, fold))
            thr = threshold_by_folds(L, y, fold)
            thr_pred = np.where(np.isnan(thr), -1, (thr > 0).astype(int))
            rows.append(dict(task=task, scope=scope, model=REFERENCE_ROW, n=len(idx),
                             model_ba=np.nan, surrogate_auc=auc_l, surrogate_ba=ba_l,
                             kappa_threshold=np.nan, spearman_L=np.nan))

            for mod in models:
                d = read_oof(a.results_dir, task, mod, a.run_tag).loc[idx]
                if not (d.fold.to_numpy(int) == fold).all():
                    print(f"[error] fold mismatch {task}/{mod}", file=sys.stderr)
                    return 3
                ymod = d.pred.to_numpy(int)
                auc, ba = score_surrogate(ymod, surrogate_oof(X, ymod, fold))
                ok = thr_pred >= 0
                k = (cohen_kappa_score(ymod[ok], thr_pred[ok])
                     if ok.sum() > 10 and len(np.unique(thr_pred[ok])) > 1 else np.nan)
                rho = spearmanr(d.prob_pos.to_numpy(), L).statistic
                rows.append(dict(task=task, scope=scope, model=mod, n=len(d),
                                 model_ba=balanced_accuracy_score(y, ymod),
                                 surrogate_auc=auc, surrogate_ba=ba,
                                 kappa_threshold=k, spearman_L=rho))

    df = pd.DataFrame(rows)
    out = os.path.join(a.out_dir, "confound_reliance.csv")
    df.to_csv(out, index=False)
    for (t, s), g in df.groupby(["task", "scope"]):
        ref = g[g.model == REFERENCE_ROW].surrogate_ba.iloc[0]
        gg = g[g.model != REFERENCE_ROW]
        print(f"[{t} / {s}] label reference BA={ref:.3f}; "
              f"model surrogate BA {gg.surrogate_ba.min():.3f}-{gg.surrogate_ba.max():.3f}; "
              f"largest gap {(gg.surrogate_ba - ref).max():+.3f} "
              f"({gg.loc[(gg.surrogate_ba - ref).idxmax(), 'model']})")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
