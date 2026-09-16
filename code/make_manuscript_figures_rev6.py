#!/usr/bin/env python3
"""Regenerate manuscript Figures 3-7 and 9 at publication quality.

Run from a directory containing results/ (see reproduce_analysis.sh). Reads
matched_report.json, cv_metrics_foldwise_original.csv, confound_reliance.csv,
features_*.csv, matched_*.csv and the XAI faithfulness CSVs; writes figures_out/
with vector PDF, 300-dpi PNG and 300-dpi TIFF (LZW) for each figure.
"""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

R = "results/"
OUT = "figures_out/"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "axes.labelsize": 9,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42})

TASKS = [("you_chalk_brood", "Chalkbrood"), ("you_foulbrood", "Foulbrood")]
C_NORM, C_DIS = "#3B6DB3", "#C8453F"
C_ORIG, C_MATCH = "#8C8C8C", "#3F9E5F"
C_HL = "#FFE58A"

MODELS = ["lgca_net", "lgca_net_no_cross_attn", "lgca_net_cnn_only",
          "resnet50", "densenet121", "efficientnetv2",
          "coatnet", "crossvit", "nextvit_small", "conformer", "lsnet_t", "mambavision"]
NAME = {"lgca_net": "LGCA-Net (proposed)", "lgca_net_no_cross_attn": "LGCA-Net w/o cross-attn.",
        "lgca_net_cnn_only": "LGCA-Net CNN only", "resnet50": "ResNet-50",
        "densenet121": "DenseNet-121", "efficientnetv2": "EfficientNetV2-S",
        "coatnet": "CoAtNet-0", "crossvit": "CrossViT-S", "nextvit_small": "Next-ViT-S",
        "conformer": "Conformer-S", "lsnet_t": "LSNet-T", "mambavision": "MambaVision-T"}
TRIV = [("threshold_L", "$L^*$ threshold"), ("logreg", "Logistic reg. (6 stats)"),
        ("gbm", "Gradient boosting (6 stats)")]


def save(fig, name):
    fig.savefig(OUT + name + ".pdf")
    fig.savefig(OUT + name + ".png", dpi=300)
    fig.savefig(OUT + name + ".tif", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("wrote", name)


rep = json.load(open(R + "matched_report.json"))
fold = pd.read_csv(R + "cv_metrics_foldwise_original.csv")

# ------------------------------------------------------------------ Fig: audit
fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.8))
for i, (t, lab) in enumerate(TASKS):
    f = pd.read_csv(R + f"features_{t}.csv")
    m = pd.read_csv(R + f"matched_{t}.csv")
    bins = np.linspace(5, 80, 46)
    for j, (d, ttl) in enumerate([(f, "before matching"), (m, "after matching")]):
        ax = axes[i, j]
        ax.hist(d.loc[d.label == 0, "lab_L_256"], bins=bins, color=C_NORM, alpha=.8, lw=0, label="Normal")
        ax.hist(d.loc[d.label == 1, "lab_L_256"], bins=bins, color=C_DIS, alpha=.8, lw=0, label="Disease-positive")
        ax.set_title(f"({chr(97 + i * 3 + j)}) {lab}, {ttl}", loc="left", fontsize=8.5)
        ax.set_xlabel("Mean CIELAB $L^*$"); ax.set_ylabel("Images")
        n0, n1 = (d.label == 0).sum(), (d.label == 1).sum()
        ax.text(0.98, 0.95, f"n = {n0} + {n1}", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="0.3")
        if i == 0 and j == 0:
            ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.9))
    ax = axes[i, 2]
    ct = pd.crosstab(f.site, f.label).reindex(sorted(f.site.unique())).fillna(0)
    x = np.arange(len(ct))
    ax.bar(x - .2, ct[0], .4, color=C_NORM)
    ax.bar(x + .2, ct[1] if 1 in ct else 0, .4, color=C_DIS)
    ax.set_xticks(x); ax.set_xticklabels(ct.index, rotation=90, fontsize=6.5)
    ax.set_title(f"({chr(99 + i * 3)}) {lab}, per site", loc="left", fontsize=8.5)
    ax.set_ylabel("Images"); ax.set_xlabel("Proxy acquisition site")
fig.tight_layout(pad=.7, w_pad=1.2)
save(fig, "fig_audit")

# ---------------------------------------------------------- Fig: performance
# model-free classifier values come from matched_report.json (hive-grouped folds, both settings)
keys = MODELS + [k for k, _ in TRIV]
labels = [NAME[k] for k in MODELS] + [n for _, n in TRIV]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 5.4), sharey=True)
for i, (t, lab) in enumerate(TASKS):
    ax = axes[i]; ys = np.arange(len(keys))[::-1]
    om, os_, km, ks = [], [], [], []
    for k in keys:
        if k in MODELS:
            fr = fold[(fold.task == t) & (fold.model == k)].iloc[0]
            om.append(fr.BA_mean); os_.append(fr.BA_sd)
            d = rep[t]["deep"][k]; km.append(d["BA_foldmean"]); ks.append(d["BA_foldsd"])
        else:
            o = rep[t]["trivial_original"][k]; om.append(o["BA_foldmean"]); os_.append(o["BA_foldsd"])
            d = rep[t]["trivial"][k]; km.append(d["BA_foldmean"]); ks.append(d["BA_foldsd"])
    ax.barh(ys + .2, om, .4, xerr=os_, color=C_ORIG, ecolor="0.2", capsize=2, error_kw=dict(lw=.8), label="Original subset")
    ax.barh(ys - .2, km, .4, xerr=ks, color=C_MATCH, ecolor="0.2", capsize=2, error_kw=dict(lw=.8), label="Acquisition-matched subset")
    ax.axvline(.5, color="k", ls=":", lw=.8)
    ax.axhline(len(keys) - len(MODELS) - .5, color="0.5", lw=.8, ls="--")
    ax.axhspan(ys[0] - .5, ys[0] + .5, color=C_HL, alpha=.45, zorder=0)
    ax.set_xlim(.3, 1.0); ax.set_xlabel("Balanced accuracy (fold mean ± SD)")
    ax.set_title(f"({chr(97 + i)}) {lab}", loc="left")
    ax.grid(axis="x", color="0.9", lw=.6); ax.set_axisbelow(True)
    if i == 0:
        ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8)
        hh, ll = ax.get_legend_handles_labels()
fig.legend(hh, ll, frameon=False, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(pad=.7, w_pad=1.0, rect=(0, 0.04, 1, 1))
save(fig, "fig_performance")

# ------------------------------------------------------------- Fig: ablation
abl = ["lgca_net_cnn_only", "lgca_net_no_cross_attn", "lgca_net"]
ABN = ["CNN branch only\n(0.29 M)", "CNN + Transformer,\nposition-wise additive\nfusion (1.20 M)", "CNN + Transformer,\ncross-attention fusion\n= LGCA-Net (1.46 M)"]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
for i, (t, lab) in enumerate(TASKS):
    ax = axes[i]; xs = np.arange(3)
    om = [fold[(fold.task == t) & (fold.model == k)].iloc[0].BA_mean for k in abl]
    os_ = [fold[(fold.task == t) & (fold.model == k)].iloc[0].BA_sd for k in abl]
    mm = [rep[t]["deep"][k]["BA_foldmean"] for k in abl]
    ms = [rep[t]["deep"][k]["BA_foldsd"] for k in abl]
    b1 = ax.bar(xs - .2, om, .4, yerr=os_, color=C_ORIG, ecolor="0.2", capsize=2, error_kw=dict(lw=.8), label="Original subset")
    b2 = ax.bar(xs + .2, mm, .4, yerr=ms, color=C_MATCH, ecolor="0.2", capsize=2, error_kw=dict(lw=.8), label="Matched subset")
    for bars, vals in [(b1, om), (b2, mm)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, 0.42, f"{v:.3f}", ha="center", va="bottom", fontsize=6.8, color="white", fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(ABN, fontsize=6.6)
    ax.set_title(f"({chr(97 + i)}) {lab}", loc="left"); ax.set_ylim(.4, 1.02)
    ax.grid(axis="y", color="0.9", lw=.6); ax.set_axisbelow(True)
    if i == 0:
        ax.set_ylabel("Balanced accuracy (fold mean ± SD)")
        hh, ll = ax.get_legend_handles_labels()
fig.legend(hh, ll, frameon=False, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(pad=.7, rect=(0, 0.06, 1, 1))
save(fig, "fig_ablation")

# ------------------------------------------------------------- Fig: confound
c = pd.read_csv(R + "confound_reliance.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.9), sharey=True)
for i, (t, lab) in enumerate(TASKS):
    ax = axes[i]; ys = np.arange(len(MODELS))[::-1]
    for j, (scope, col, lg) in enumerate([("original", C_ORIG, "Original subset"), ("matched", C_MATCH, "Acquisition-matched subset")]):
        g = c[(c.task == t) & (c.scope == scope)].set_index("model")
        vals = [g.loc[m].surrogate_ba for m in MODELS]
        ax.barh(ys + (.2 if j == 0 else -.2), vals, .4, color=col, label=lg)
        ref = g.loc["__reference_label__"].surrogate_ba
        ax.axvline(ref, color=col, ls=(0, (4, 2)), lw=1.3, zorder=5)
        ax.text(ref, len(MODELS) - .3 + (0.75 if j == 0 else 0.0), f"label reference, {scope}: {ref:.3f}", color=col, fontsize=6.5,
                ha="center", va="bottom", bbox=dict(fc="white", ec="none", pad=0.6))
    ax.axhspan(ys[0] - .5, ys[0] + .5, color=C_HL, alpha=.45, zorder=0)
    ax.set_xlim(.5, 1.0); ax.set_ylim(-.6, len(MODELS) + 1.5)
    ax.set_xlabel("Balanced accuracy of the surrogate\n(six global statistics → model decision)")
    ax.set_title(f"({chr(97 + i)}) {lab}", loc="left")
    ax.grid(axis="x", color="0.9", lw=.6); ax.set_axisbelow(True)
    if i == 0:
        ax.set_yticks(ys); ax.set_yticklabels([NAME[m] for m in MODELS], fontsize=8)
        hh, ll = ax.get_legend_handles_labels()
fig.legend(hh, ll, frameon=False, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(pad=.7, w_pad=1.0, rect=(0, 0.05, 1, 1))
save(fig, "fig_confound")

# ------------------------------------------------------------- Fig: xai faithfulness
di, pn = [], []
for t, lab in TASKS:
    a = pd.read_csv(R + f"{t}_grad_saliency_drop_inc.csv"); b = pd.read_csv(R + f"{t}_lime_shap_drop_inc.csv")
    d = pd.concat([a, b]); d = d[d.Fold.astype(str).isin(["0", "1", "2"])]; d["task"] = lab; di.append(d)
    p = pd.read_csv(R + f"{t}_pos_neg_perturb_auc.csv"); q = pd.read_csv(R + f"{t}_lime_shap_pos_neg_perturb_auc.csv")
    pq = pd.concat([p, q]); pq = pq[pq.Fold.astype(str).isin(["0", "1", "2"])]; pq["task"] = lab; pn.append(pq)
di = pd.concat(di); pn = pd.concat(pn)
methods = ["GradCAM", "Saliency", "LIME", "SHAP"]; mlab = ["Grad-CAM", "Gradient\nsaliency", "LIME", "SHAP"]
fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2))
panels = [("AverageDropPercent", "Average drop (%)  ↓ better", "(a) Average drop", di),
          ("IncreaseInConfidencePercent", "Increase in confidence (%)  ↑ better", "(b) Increase in confidence", di),
          ("PosAUC", "Positive-perturbation AUC  ↓ better", "(c) Deletion curve, most-attributed first", pn),
          ("NegAUC", "Negative-perturbation AUC  ↑ better", "(d) Deletion curve, least-attributed first", pn)]
for ax, (col, ylab, title, src) in zip(axes.ravel(), panels):
    xs = np.arange(len(methods))
    for j, (t, lab) in enumerate(TASKS):
        s = src[src.task == lab]
        mv = [s[s.Method == m][col].astype(float).mean() for m in methods]
        sv = [s[s.Method == m][col].astype(float).std(ddof=1) for m in methods]
        ax.bar(xs + (j - .5) * .4, mv, .4, yerr=sv, capsize=2, error_kw=dict(lw=.8), color=[C_NORM, C_DIS][j], label=lab)
    ax.set_xticks(xs); ax.set_xticklabels(mlab, fontsize=7.5); ax.set_ylabel(ylab, fontsize=8); ax.set_title(title, loc="left", fontsize=9)
    ax.grid(axis="y", color="0.9", lw=.6); ax.set_axisbelow(True)
    if col in ("PosAUC", "NegAUC"):
        # No reference line: the deletion-curve area expected from an uninformative
        # pixel ranking is classifier-dependent and is not 0.5 in general (rev. 7).
        ax.set_ylim(0, 0.9)
    if col == "AverageDropPercent":
        hh, ll = ax.get_legend_handles_labels()
fig.legend(hh, ll, frameon=False, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(pad=.7, rect=(0, 0.04, 1, 1))
save(fig, "fig_xai")

# ------------------------------------------------------------- Fig: embedding
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
F = ["lab_L_256", "lab_a_256", "lab_b_256", "rgb_r_256", "rgb_g_256", "rgb_b_256", "file_size_bytes", "otsu_largest_area_frac"]
fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.0))
for i, (t, lab) in enumerate(TASKS):
    f = pd.read_csv(R + f"features_{t}.csv")
    Z = StandardScaler().fit_transform(f[F].values)
    E = TSNE(n_components=2, random_state=0, perplexity=30, init="pca").fit_transform(Z)
    ax = axes[i, 0]
    for l, cc, n in [(0, C_NORM, "Normal"), (1, C_DIS, "Disease-positive")]:
        s = f.label.values == l; ax.scatter(E[s, 0], E[s, 1], s=6, c=cc, alpha=.65, lw=0, label=n)
    ax.set_title(f"({chr(97 + 2 * i)}) {lab}: coloured by class", loc="left"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    if i == 0: ax.legend(frameon=False, markerscale=2.5, loc="best")
    ax = axes[i, 1]
    sites = sorted(f.site.unique()); cmap = plt.get_cmap("tab20")
    for k, s_ in enumerate(sites):
        s = f.site.values == s_; ax.scatter(E[s, 0], E[s, 1], s=6, color=cmap(k % 20), alpha=.75, lw=0, label=s_)
    ax.set_title(f"({chr(98 + 2 * i)}) {lab}: coloured by proxy acquisition site", loc="left"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("t-SNE 1")
    ax.legend(frameon=False, markerscale=2.2, fontsize=6.2, ncol=2, loc="center left", bbox_to_anchor=(1.0, 0.5), title="Site", title_fontsize=6.5)
fig.tight_layout(pad=.7)
fig.savefig(OUT + "fig_embedding.pdf", bbox_inches="tight")
fig.savefig(OUT + "fig_embedding.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT + "fig_embedding.tif", dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig); print("wrote fig_embedding")
