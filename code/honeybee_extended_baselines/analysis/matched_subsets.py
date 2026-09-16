#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Acquisition-matched subsets, and every classifier evaluated on them.

Steps, per task:
  1. restrict normal candidates to the proxy sites that contribute disease
     images, then pair each positive with one normal from the same site by
     greedy nearest-neighbour matching on mean L* without replacement, caliper
     0.2 pooled SD;
  2. refit the three model-free classifiers on the matched images, under folds
     inherited from the audited run;
  3. restrict every deep model's saved out-of-fold predictions to the matched
     images (so no image is scored by a model that trained on it);
  4. hive-clustered bootstrap CIs and fold-wise DeLong / exact McNemar tests
     with Holm correction against the reference model.

Model list is discovered from the results directory: adding a baseline needs no
edit here.

    python matched_subsets.py --results-dir ./results --out-dir ./results \
        --tables-dir ./tables --run-tag original
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_io import (BALANCE_FEATS, GLOBAL_FEATS, TASKS, base_parser,  # noqa: E402
                       ensure_dir, holm, read_features, read_oof, smd)
from model_registry import REFERENCE_MODEL, common_models, order_models  # noqa: E402


# ---------------------------------------------------------------- matching --
def match_within_site(f: pd.DataFrame, rng, caliper_frac: float = 0.2):
    sd = f["lab_L_256"].std(ddof=1)
    caliper = caliper_frac * sd
    pairs = []
    for _, g in f.groupby("site"):
        pos, neg = g[g.label == 1], g[g.label == 0]
        if pos.empty or neg.empty:
            continue
        pos = pos.sample(frac=1, random_state=rng)
        negL = neg["lab_L_256"].to_numpy(float)
        negidx = neg.index.to_numpy()
        used = np.zeros(len(neg), bool)
        for _, p in pos.iterrows():
            d = np.abs(negL - p["lab_L_256"])
            d[used] = np.inf
            j = int(np.argmin(d))
            if d[j] <= caliper:
                used[j] = True
                pairs.append((p.name, negidx[j]))
    return pairs


# --------------------------------------------------------------- statistics --
def delong(y, p1, p2):
    """DeLong test for two correlated AUROCs (positives ordered first)."""
    y = np.asarray(y)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    idx = np.concatenate([pos, neg])
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return np.nan, np.nan, np.nan, 1.0
    preds = np.vstack([np.asarray(p1)[idx], np.asarray(p2)[idx]])

    def midrank(x):
        J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
        while i < N:
            j = i
            while j < N - 1 and Z[j + 1] == Z[i]:
                j += 1
            T[i:j + 1] = 0.5 * (i + j) + 1
            i = j + 1
        out = np.empty(N); out[J] = T
        return out

    tx = np.array([midrank(p[:m]) for p in preds])
    ty = np.array([midrank(p[m:]) for p in preds])
    tz = np.array([midrank(p) for p in preds])
    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    S = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    l = np.array([1.0, -1.0])
    var = float(l @ S @ l)
    if var <= 0:
        return aucs[0], aucs[1], np.nan, 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs[0], aucs[1], z, 2 * stats.norm.sf(abs(z))


def mcnemar_exact(y, a, b):
    b_ = int(((a == y) & (b != y)).sum())
    c_ = int(((a != y) & (b == y)).sum())
    n = b_ + c_
    if n == 0:
        return b_, c_, 1.0
    return b_, c_, float(min(1.0, 2 * stats.binom.cdf(min(b_, c_), n, 0.5)))


def boot_ci(y, score, pred, groups, B=2000, seed=0):
    rng = np.random.RandomState(seed)
    ug = np.unique(groups)
    gidx = {g: np.where(groups == g)[0] for g in ug}
    bas, aucs = [], []
    for _ in range(B):
        idx = np.concatenate([gidx[g] for g in rng.choice(ug, len(ug), replace=True)])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        bas.append(balanced_accuracy_score(yy, pred[idx]))
        aucs.append(roc_auc_score(yy, score[idx]))
    if not bas:
        return (np.nan, np.nan), (np.nan, np.nan)
    return np.percentile(bas, [2.5, 97.5]), np.percentile(aucs, [2.5, 97.5])


def threshold_by_folds(L, y, fold):
    """Single fitted L* threshold, trained per fold on the other folds."""
    score = np.full(len(y), np.nan)
    for f in np.unique(fold):
        te, tr = fold == f, fold != f
        if len(np.unique(y[tr])) < 2:
            continue
        best = (-1.0, None, None)
        for c in np.unique(np.quantile(L[tr], np.linspace(.01, .99, 99))):
            for pol in (1, -1):
                b = balanced_accuracy_score(y[tr], (pol * (L[tr] - c) > 0).astype(int))
                if b > best[0]:
                    best = (b, c, pol)
        _, c, pol = best
        score[te] = pol * (L[te] - c)
    return score


# -------------------------------------------------------------------- main --
def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--caliper", type=float, default=0.2)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=0,
                    help="RNG seed for the hive-clustered bootstrap (default 0, "
                         "the value used for the published intervals)")
    a = ap.parse_args()
    ensure_dir(a.out_dir); ensure_dir(a.tables_dir)
    rng = np.random.RandomState(a.seed)

    models = common_models(a.results_dir, a.tasks, a.run_tag)
    if REFERENCE_MODEL not in models:
        print(f"[error] reference model '{REFERENCE_MODEL}' has no OOF file", file=sys.stderr)
        return 2
    print(f"[info] {len(models)} models: {', '.join(models)}")

    report, pairwise_rows = {}, []
    for task in a.tasks:
        f = read_features(a.results_dir, task)
        before = dict(
            n_pos=int((f.label == 1).sum()), n_neg=int((f.label == 0).sum()),
            n_sites_pos=int(f.loc[f.label == 1, "site"].nunique()),
            n_sites_neg=int(f.loc[f.label == 0, "site"].nunique()),
            L_smd=smd(f.loc[f.label == 1, "lab_L_256"], f.loc[f.label == 0, "lab_L_256"]),
            L_mean_pos=float(f.loc[f.label == 1, "lab_L_256"].mean()),
            L_mean_neg=float(f.loc[f.label == 0, "lab_L_256"].mean()))

        pairs = match_within_site(f, rng, a.caliper)
        keep = [x for p in pairs for x in p]
        m = f.loc[keep].copy()
        after = dict(
            n_pairs=len(pairs), n=len(m),
            L_smd=smd(m.loc[m.label == 1, "lab_L_256"], m.loc[m.label == 0, "lab_L_256"]),
            L_mean_pos=float(m.loc[m.label == 1, "lab_L_256"].mean()),
            L_mean_neg=float(m.loc[m.label == 0, "lab_L_256"].mean()),
            n_sites=int(m.site.nunique()), sites=sorted(m.site.unique().tolist()),
            n_hives=int(m.hive_proxy.nunique()))
        balance = {c: dict(smd_before=smd(f.loc[f.label == 1, c], f.loc[f.label == 0, c]),
                           smd_after=smd(m.loc[m.label == 1, c], m.loc[m.label == 0, c]))
                   for c in BALANCE_FEATS}
        m[["filename", "site", "hive_proxy", "label", "lab_L_256"]].to_csv(
            os.path.join(a.out_dir, f"matched_{task}.csv"), index=False)

        # fold assignment inherited from the audited run
        ref = read_oof(a.results_dir, task, REFERENCE_MODEL, a.run_tag)
        ref_m = ref[ref.index.isin(set(m.filename))]
        idx = ref_m.index
        mm = m.set_index("filename").loc[idx]
        fold = ref_m.fold.to_numpy(int)
        y = ref_m.y_true.to_numpy(int)
        grp = mm.hive_proxy.to_numpy()
        X = mm[GLOBAL_FEATS].to_numpy(float)
        L = mm["lab_L_256"].to_numpy(float)

        # ---- model-free classifiers refitted on the matched images ----
        triv = {}
        s_thr = threshold_by_folds(L, y, fold)
        for name, score in [("threshold_L", s_thr)] + _fit_multivariate(X, y, fold):
            cut = 0.0 if name == "threshold_L" else 0.5
            pred = (score > cut).astype(int)
            fb = [balanced_accuracy_score(y[fold == k], pred[fold == k]) for k in np.unique(fold)]
            fa = [roc_auc_score(y[fold == k], score[fold == k]) for k in np.unique(fold)
                  if len(np.unique(y[fold == k])) > 1]
            triv[name] = dict(BA=balanced_accuracy_score(y, pred), AUROC=roc_auc_score(y, score),
                              BA_foldmean=float(np.mean(fb)), BA_foldsd=float(np.std(fb, ddof=1)),
                              AUROC_foldmean=float(np.mean(fa)), AUROC_foldsd=float(np.std(fa, ddof=1)))

        # ---- the same model-free classifiers on the FULL subset, audited folds ----
        # The published pipeline fitted these under an ungrouped stratified split
        # while every deep model used hive-grouped folds, so the two were not
        # comparable. Recomputing them here on the audited folds makes the
        # "same folds as the deep models" claim true.
        full = read_oof(a.results_dir, task, REFERENCE_MODEL, a.run_tag)
        ff = f.set_index("filename").loc[full.index]
        fold_o = full.fold.to_numpy(int)
        y_o = full.y_true.to_numpy(int)
        X_o = ff[GLOBAL_FEATS].to_numpy(float)
        L_o = ff["lab_L_256"].to_numpy(float)
        triv_orig = {}
        for name, score in ([("threshold_L", threshold_by_folds(L_o, y_o, fold_o))]
                            + _fit_multivariate(X_o, y_o, fold_o)):
            cut = 0.0 if name == "threshold_L" else 0.5
            pred = (score > cut).astype(int)
            fb = [balanced_accuracy_score(y_o[fold_o == k], pred[fold_o == k])
                  for k in np.unique(fold_o)]
            fa = [roc_auc_score(y_o[fold_o == k], score[fold_o == k])
                  for k in np.unique(fold_o) if len(np.unique(y_o[fold_o == k])) > 1]
            triv_orig[name] = dict(
                BA=balanced_accuracy_score(y_o, pred), AUROC=roc_auc_score(y_o, score),
                BA_foldmean=float(np.mean(fb)), BA_foldsd=float(np.std(fb, ddof=1)),
                AUROC_foldmean=float(np.mean(fa)), AUROC_foldsd=float(np.std(fa, ddof=1)))

        # ---- deep models restricted to the matched images ----
        deep, ci = {}, {}
        preds = {}
        for mod in models:
            d = read_oof(a.results_dir, task, mod, a.run_tag).loc[idx]
            if not (d.fold.to_numpy(int) == fold).all():
                print(f"[error] fold mismatch for {task}/{mod}", file=sys.stderr)
                return 3
            preds[mod] = d
            fb, fa = [], []
            for k in np.unique(fold):
                g = d[fold == k]
                if g.y_true.nunique() < 2:
                    continue
                fb.append(balanced_accuracy_score(g.y_true, g.pred))
                fa.append(roc_auc_score(g.y_true, g.prob_pos))
            (bl, bh), (al, ah) = boot_ci(y, d.prob_pos.to_numpy(), d.pred.to_numpy(int),
                                         grp, B=a.bootstrap, seed=a.bootstrap_seed)
            deep[mod] = dict(n=int(len(d)), BA=balanced_accuracy_score(y, d.pred),
                             AUROC=roc_auc_score(y, d.prob_pos), F1=f1_score(y, d.pred),
                             BA_foldmean=float(np.mean(fb)), BA_foldsd=float(np.std(fb, ddof=1)),
                             AUROC_foldmean=float(np.mean(fa)), AUROC_foldsd=float(np.std(fa, ddof=1)),
                             n_folds=len(fb))
            ci[mod] = dict(BA=deep[mod]["BA"], BA_lo=bl, BA_hi=bh,
                           AUROC=deep[mod]["AUROC"], AUROC_lo=al, AUROC_hi=ah)

        # ---- fold-wise paired tests against the reference model ----
        r0 = preds[REFERENCE_MODEL]
        rows = []
        for mod in models:
            if mod == REFERENCE_MODEL:
                continue
            for k in np.unique(fold):
                s = fold == k
                if len(np.unique(y[s])) < 2:
                    continue
                a1, a2, z, pd_ = delong(y[s], r0.prob_pos.to_numpy()[s],
                                        preds[mod].prob_pos.to_numpy()[s])
                b_, c_, pm = mcnemar_exact(y[s], r0.pred.to_numpy(int)[s],
                                           preds[mod].pred.to_numpy(int)[s])
                rows.append(dict(task=task, fold=int(k), model=mod, n=int(s.sum()),
                                 auc_ref=a1, auc_model=a2, delong_z=z, delong_p=pd_,
                                 mcnemar_b=b_, mcnemar_c=c_, mcnemar_p=pm))
        r = pd.DataFrame(rows)
        if len(r):
            r["delong_p_holm"] = holm(r.delong_p.fillna(1).to_numpy())
            r["mcnemar_p_holm"] = holm(r.mcnemar_p.to_numpy())
            r.to_csv(os.path.join(a.out_dir, f"matched_pairwise_{task}.csv"), index=False)
            pairwise_rows.append(r)

        report[task] = dict(before=before, after=after, balance=balance,
                            trivial=triv, trivial_original=triv_orig, deep=deep, ci=ci,
                            n_comparisons=int(len(r)),
                            n_sig_delong=int((r.delong_p_holm < .05).sum()) if len(r) else 0,
                            n_sig_mcnemar=int((r.mcnemar_p_holm < .05).sum()) if len(r) else 0,
                            models=models)
        print(f"[{task}] {after['n_pairs']} pairs / {after['n']} images, "
              f"L* SMD {before['L_smd']:+.3f} -> {after['L_smd']:+.3f}; "
              f"Holm-significant DeLong {report[task]['n_sig_delong']}/{len(r)}")

    json.dump(report, open(os.path.join(a.out_dir, "matched_report.json"), "w"),
              indent=1, default=float)
    print(f"wrote {os.path.join(a.out_dir, 'matched_report.json')}")
    return 0


def _fit_multivariate(X, y, fold):
    out = []
    for name, mk in [("logreg", lambda: Pipeline([("s", StandardScaler()),
                                                  ("c", LogisticRegression(max_iter=3000))])),
                     ("gbm", lambda: HistGradientBoostingClassifier(random_state=0))]:
        score = np.full(len(y), np.nan)
        for f in np.unique(fold):
            te, tr = fold == f, fold != f
            if len(np.unique(y[tr])) < 2:
                continue
            score[te] = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        out.append((name, score))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
