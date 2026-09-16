"""Metrics, bootstrap confidence intervals and statistical tests.

Pure NumPy/SciPy — no torch, no statsmodels. Everything here is importable on a
CPU-only machine.

Contents
--------
* Point metrics: balanced accuracy, sensitivity, specificity, precision,
  macro-F1, AUROC, mean cross-entropy loss (:func:`compute_metrics`).
* :func:`bootstrap_ci` — seeded percentile bootstrap (n = 2000 by default).
* :func:`delong_roc_test` — fast DeLong test for two *correlated* AUCs
  (DeLong et al. 1988; fast midrank implementation of Sun & Xu 2014).
* :func:`mcnemar_test` — exact binomial and mid-p variants.
* :func:`holm_bonferroni` — step-down Holm adjustment.
* :func:`welch_t_from_summary` / :func:`welch_t_from_raw` — Welch's unequal
  variance t-test, from summary statistics or raw arrays.
* :func:`hedges_g` — bias-corrected standardised mean difference.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Point metrics
# --------------------------------------------------------------------------- #
def _binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def sensitivity(y_true, y_pred) -> float:
    """Recall of the positive (diseased) class = TP / (TP + FN)."""
    tp, _, _, fn = _binary_counts(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else float("nan")


def specificity(y_true, y_pred) -> float:
    """Recall of the negative (normal) class = TN / (TN + FP)."""
    _, tn, fp, _ = _binary_counts(y_true, y_pred)
    return tn / (tn + fp) if (tn + fp) > 0 else float("nan")


def balanced_accuracy(y_true, y_pred) -> float:
    """Mean of sensitivity and specificity."""
    return 0.5 * (sensitivity(y_true, y_pred) + specificity(y_true, y_pred))


def precision_positive(y_true, y_pred) -> float:
    """Positive predictive value = TP / (TP + FP)."""
    tp, _, fp, _ = _binary_counts(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else float("nan")


def f1_macro(y_true, y_pred) -> float:
    """Unweighted mean of the per-class F1 scores (macro-F1)."""
    tp, tn, fp, fn = _binary_counts(y_true, y_pred)
    # positive class
    p1 = tp / (tp + fp) if (tp + fp) else 0.0
    r1 = tp / (tp + fn) if (tp + fn) else 0.0
    f1_pos = 2 * p1 * r1 / (p1 + r1) if (p1 + r1) else 0.0
    # negative class
    p0 = tn / (tn + fn) if (tn + fn) else 0.0
    r0 = tn / (tn + fp) if (tn + fp) else 0.0
    f1_neg = 2 * p0 * r0 / (p0 + r0) if (p0 + r0) else 0.0
    return 0.5 * (f1_pos + f1_neg)


def auroc(y_true, scores) -> float:
    """AUROC via the Mann-Whitney U statistic with midranks for ties."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def mean_cross_entropy(y_true, prob_pos) -> float:
    """Mean binary cross-entropy (natural log), unweighted, as in training."""
    y_true = np.asarray(y_true).astype(float)
    p = np.clip(np.asarray(prob_pos, dtype=float), EPS, 1.0 - EPS)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


METRIC_ORDER = ["BA", "Sensitivity", "Specificity", "Precision", "F1", "AUROC", "Loss"]


def compute_metrics(
    y_true: Sequence[int],
    prob_pos: Sequence[float],
    y_pred: Optional[Sequence[int]] = None,
    threshold: float = 0.5,
    loss: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """All seven reported metrics in one dict.

    Note there is deliberately **no** separate ``Recall`` entry: for binary
    classification it is identical to ``Sensitivity`` and the manuscript's
    original table carried both.

    ``loss`` may be supplied as per-image losses (e.g. from the training
    criterion); otherwise it is recomputed from ``prob_pos``.
    """
    y_true = np.asarray(y_true).astype(int)
    prob_pos = np.asarray(prob_pos, dtype=float)
    if y_pred is None:
        y_pred = (prob_pos >= threshold).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    ce = float(np.mean(np.asarray(loss, dtype=float))) if loss is not None else mean_cross_entropy(y_true, prob_pos)
    return {
        "BA": balanced_accuracy(y_true, y_pred),
        "Sensitivity": sensitivity(y_true, y_pred),
        "Specificity": specificity(y_true, y_pred),
        "Precision": precision_positive(y_true, y_pred),
        "F1": f1_macro(y_true, y_pred),
        "AUROC": auroc(y_true, prob_pos),
        "Loss": ce,
    }


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_ci(
    y_true: Sequence[int],
    prob_pos: Sequence[float],
    y_pred: Optional[Sequence[int]] = None,
    loss: Optional[Sequence[float]] = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    stratified: bool = True,
) -> Dict[str, Tuple[float, float, float]]:
    """Seeded percentile bootstrap CIs for every metric in :func:`compute_metrics`.

    Parameters
    ----------
    n_boot:
        Number of resamples (2000 by default, as reported).
    stratified:
        Resample within each class so every replicate keeps both classes
        present (otherwise sensitivity/AUROC can be undefined in small folds).

    Returns
    -------
    dict
        ``metric -> (point_estimate, ci_low, ci_high)``.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    prob_pos = np.asarray(prob_pos, dtype=float)
    if y_pred is None:
        y_pred = (prob_pos >= 0.5).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    loss_arr = np.asarray(loss, dtype=float) if loss is not None else None

    point = compute_metrics(y_true, prob_pos, y_pred, loss=loss_arr)
    n = len(y_true)
    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]

    draws: Dict[str, List[float]] = {k: [] for k in point}
    for _ in range(int(n_boot)):
        if stratified and idx_pos.size and idx_neg.size:
            bi = np.concatenate(
                [rng.choice(idx_pos, idx_pos.size, replace=True),
                 rng.choice(idx_neg, idx_neg.size, replace=True)]
            )
        else:
            bi = rng.choice(n, n, replace=True)
        m = compute_metrics(
            y_true[bi], prob_pos[bi], y_pred[bi],
            loss=loss_arr[bi] if loss_arr is not None else None,
        )
        for k, v in m.items():
            draws[k].append(v)

    out: Dict[str, Tuple[float, float, float]] = {}
    for k, v in point.items():
        arr = np.asarray(draws[k], dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            out[k] = (v, float("nan"), float("nan"))
        else:
            lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            out[k] = (v, float(lo), float(hi))
    return out


# --------------------------------------------------------------------------- #
# Fast DeLong (Sun & Xu, IEEE SPL 2014)
# --------------------------------------------------------------------------- #
def _midrank(x: np.ndarray) -> np.ndarray:
    """Midranks of ``x`` (ties get the average rank)."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    """Core fast-DeLong computation.

    Parameters
    ----------
    predictions_sorted_transposed:
        ``(k, n)`` array of scores for ``k`` classifiers, with all positive
        samples first.
    label_1_count:
        Number of positive samples ``m``.

    Returns
    -------
    (aucs, cov) : ndarray of shape (k,), ndarray of shape (k, k)
    """
    m = int(label_1_count)
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _midrank(positive[r, :])
        ty[r, :] = _midrank(negative[r, :])
        tz[r, :] = _midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    sx = np.atleast_2d(sx)
    sy = np.atleast_2d(sy)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(
    y_true: Sequence[int],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
) -> Dict[str, float]:
    """DeLong test for two AUCs measured on the *same* samples (correlated).

    Returns a dict with ``auc_a``, ``auc_b``, ``diff`` (a - b), ``z``, ``p``
    (two-sided), and ``se`` of the difference.

    ``p`` is NaN when the estimated variance of the difference is zero (which
    happens when the two score vectors are identical).
    """
    y_true = np.asarray(y_true).astype(int)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    order = (-y_true).argsort(kind="mergesort")  # positives first, stable
    label_1_count = int(y_true.sum())
    if label_1_count == 0 or label_1_count == len(y_true):
        return {"auc_a": float("nan"), "auc_b": float("nan"), "diff": float("nan"),
                "se": float("nan"), "z": float("nan"), "p": float("nan")}
    preds = np.vstack([a[order], b[order]])
    aucs, cov = _fast_delong(preds, label_1_count)
    l = np.array([[1, -1]], dtype=float)
    var = float(np.atleast_2d(l @ cov @ l.T).reshape(-1)[0])
    diff = float(aucs[0] - aucs[1])
    if not np.isfinite(var) or var <= 0:
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff,
                "se": 0.0, "z": float("nan"),
                "p": 1.0 if abs(diff) < 1e-12 else float("nan")}
    se = math.sqrt(var)
    z = diff / se
    p = float(2 * stats.norm.sf(abs(z)))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff,
            "se": se, "z": float(z), "p": p}


# --------------------------------------------------------------------------- #
# McNemar
# --------------------------------------------------------------------------- #
def mcnemar_test(
    y_true: Sequence[int],
    pred_a: Sequence[int],
    pred_b: Sequence[int],
    method: str = "exact",
) -> Dict[str, float]:
    """McNemar's test on paired correctness of two classifiers.

    ``b`` = A correct, B wrong; ``c`` = A wrong, B correct.

    Parameters
    ----------
    method:
        ``"exact"`` — two-sided exact binomial test on ``min(b, c)``
        (implemented directly, no statsmodels).
        ``"midp"`` — mid-p variant, which is less conservative for small
        discordant counts.

    Returns dict with ``b``, ``c``, ``n_discordant``, ``p``, and ``method``.
    """
    y_true = np.asarray(y_true).astype(int)
    ca = (np.asarray(pred_a).astype(int) == y_true)
    cb = (np.asarray(pred_b).astype(int) == y_true)
    b = int(np.sum(ca & ~cb))
    c = int(np.sum(~ca & cb))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p": 1.0, "method": method}

    k = min(b, c)
    # two-sided exact binomial p under p=0.5
    p_exact = float(min(1.0, 2.0 * stats.binom.cdf(k, n, 0.5)))
    if method == "exact":
        p = p_exact
    elif method == "midp":
        p = float(min(1.0, 2.0 * stats.binom.cdf(k, n, 0.5) - stats.binom.pmf(k, n, 0.5)))
        p = max(p, 0.0)
    else:
        raise ValueError("method must be 'exact' or 'midp'")
    return {"b": b, "c": c, "n_discordant": n, "p": p, "method": method}


# --------------------------------------------------------------------------- #
# Multiple comparison correction
# --------------------------------------------------------------------------- #
def holm_bonferroni(pvals: Sequence[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Holm-Bonferroni step-down adjustment (no statsmodels).

    NaN p-values are passed through as NaN and excluded from the family size.

    Returns
    -------
    (p_adj, reject)
        ``p_adj`` are monotone-enforced adjusted p-values clipped at 1.0;
        ``reject`` is ``p_adj <= alpha``.
    """
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    finite = np.where(np.isfinite(p))[0]
    if finite.size == 0:
        return out, np.zeros(p.shape, dtype=bool)

    sub = p[finite]
    m = sub.size
    order = np.argsort(sub, kind="mergesort")
    sorted_p = sub[order]
    adj = np.empty(m, dtype=float)
    running = 0.0
    for i in range(m):
        val = (m - i) * sorted_p[i]
        running = max(running, val)          # enforce monotonicity
        adj[i] = min(1.0, running)
    unsorted = np.empty(m, dtype=float)
    unsorted[order] = adj
    out[finite] = unsorted
    reject = np.zeros(p.shape, dtype=bool)
    reject[finite] = unsorted <= alpha
    return out, reject


# --------------------------------------------------------------------------- #
# Effect sizes / Welch t
# --------------------------------------------------------------------------- #
def welch_t_from_summary(
    mean1: float, sd1: float, n1: int, mean2: float, sd2: float, n2: int
) -> Dict[str, float]:
    """Welch's unequal-variance t-test from summary statistics.

    Used by ``seed_stats.py`` when only the legacy "mean (lo-hi)" summary CSV is
    available and the SD has been recovered from the reported CI half-width.
    """
    n1, n2 = int(n1), int(n2)
    if n1 < 2 or n2 < 2:
        return {"t": float("nan"), "df": float("nan"), "p": float("nan"),
                "diff": mean1 - mean2}
    v1 = (sd1 ** 2) / n1
    v2 = (sd2 ** 2) / n2
    denom = v1 + v2
    diff = float(mean1 - mean2)
    if denom <= 0:
        return {"t": float("nan"), "df": float("nan"),
                "p": 1.0 if abs(diff) < 1e-12 else float("nan"), "diff": diff}
    t = diff / math.sqrt(denom)
    df = (denom ** 2) / ((v1 ** 2) / (n1 - 1) + (v2 ** 2) / (n2 - 1))
    p = float(2 * stats.t.sf(abs(t), df))
    return {"t": float(t), "df": float(df), "p": p, "diff": diff}


def welch_t_from_raw(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Welch's t-test from raw per-seed values."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return {"t": float("nan"), "df": float("nan"), "p": float("nan"),
                "diff": float(np.mean(a) - np.mean(b)) if a.size and b.size else float("nan")}
    return welch_t_from_summary(
        float(a.mean()), float(a.std(ddof=1)), a.size,
        float(b.mean()), float(b.std(ddof=1)), b.size,
    )


def hedges_g(
    mean1: float, sd1: float, n1: int, mean2: float, sd2: float, n2: int
) -> float:
    """Hedges' g: Cohen's d on the pooled SD, times the small-sample correction J."""
    n1, n2 = int(n1), int(n2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    df = n1 + n2 - 2
    sp2 = ((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / df
    if sp2 <= 0:
        return 0.0 if abs(mean1 - mean2) < 1e-12 else float("nan")
    d = (mean1 - mean2) / math.sqrt(sp2)
    J = 1.0 - 3.0 / (4.0 * df - 1.0)
    return float(J * d)


def ci_overlap(lo1: float, hi1: float, lo2: float, hi2: float) -> bool:
    """True when two confidence intervals overlap."""
    if not all(np.isfinite([lo1, hi1, lo2, hi2])):
        return True
    return not (hi1 < lo2 or hi2 < lo1)


def sd_from_ci(mean: float, lo: float, hi: float, n: int = 5, conf: float = 0.95) -> float:
    """Recover the SD from a mean +/- t_{crit} * SD / sqrt(n) confidence interval.

    The legacy seed tables report ``mean (lo-hi)`` where the half-width is
    ``t_{0.975, n-1} * SD / sqrt(n)``. Inverting gives
    ``SD = half_width * sqrt(n) / t_crit``.
    """
    if n < 2:
        return float("nan")
    half = (float(hi) - float(lo)) / 2.0
    tcrit = float(stats.t.ppf(0.5 + conf / 2.0, n - 1))
    return float(half * math.sqrt(n) / tcrit)
