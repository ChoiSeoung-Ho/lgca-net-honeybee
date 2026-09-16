> **Note (revision 6).** This is the original per-script documentation of the pipeline. The protocol actually used for the manuscript is summarised in the top-level `README.md`: all networks were trained with `train_cv.py --grouped --run-tag original`, and the acquisition-matched comparison is an *evaluation-time* analysis of the saved out-of-fold predictions performed by `honeybee_extended_baselines/analysis/matched_subsets.py` (Section 3b below describes a matched *training* run that was not executed for the manuscript). The `--run-tag matched` and seed-repetition stages are provided but their outputs are not reported.

# LGCA-Net — honeybee larval disease classification: reproducibility package

Code accompanying the EAAI revision of *a hybrid CNN–Transformer with
cross-attention (**LGCA-Net**, Local-query Global-key Cross-Attention Network)
for honeybee larval disease classification*, on the
AI-Hub honeybee dataset (No. 71667), tasks `you_chalk_brood` (chalkbrood,
disease code `002`) and `you_foulbrood` (foulbrood, code `003`).

Everything the reviewers asked for — stratified and hive-grouped splits with
written manifests, model-free trivial baselines, a near-duplicate audit,
bootstrap CIs, fold-wise DeLong/McNemar with Holm correction, seed-variance
statistics, and leakage-free quantitative XAI localisation — is produced by the
scripts below and lands in `tables/` as `\input`-ready LaTeX fragments.

---

## 1. Environment

The paper's environment is **Python 3.10, PyTorch 2.9.0, CUDA 12.8**, with
`timm` for the ImageNet-pretrained baselines.

```bash
pip install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Only stages 5–8 need a GPU. **Everything else runs on CPU with just NumPy,
SciPy, pandas, scikit-learn, Pillow, OpenCV and matplotlib**, and the modules
that touch torch guard the import so the analysis scripts stay importable
without it.

Two dependencies are deliberately avoided so the package installs cleanly:
`imagehash` (pHash is implemented directly with a DCT in
`extract_image_features.py`) and `statsmodels` (Holm–Bonferroni, DeLong and
McNemar are implemented in `common/metrics.py`).

## 2. Data layout

```
data/
  you_chalk_brood/{normal,abnormal}/*.jpg
  you_foulbrood/{normal,abnormal}/*.jpg
normal_pool/            # the FULL AI-Hub normal folder (see below)
  <task>/normal/*.jpg
labels/                 # provider JSON lesion boxes, for the XAI stage only
```

`normal_pool/` is the large candidate pool the acquisition matching draws from.
The delivered task folders confound class with acquisition site and brightness,
so the negative class is rebuilt by matching against this pool — matching needs
many more candidates than there are positives, which the task folder alone does
not provide.

Native images are typically 1920×1080 JPEGs. File names follow the AI-Hub
convention, e.g. `B_001_018_20230819135434_001_004_000_002.jpg`, with eight
underscore-delimited fields `f1..f8`:

| Field | Use in this package |
| ----- | ------------------- |
| `f1_f2` | **site** — coarse proxy acquisition-unit identifier |
| `f1_f2_f3` | **hive-proxy** — the grouping unit for grouped CV |
| `f4` | `YYYYMMDDhhmmss` timestamp; its first 8 chars give the **session** date |
| `f8` | disease code: `000` normal, `002` chalkbrood, `003` foulbrood |

The provider does not document the semantics of `f1`, `f2`, `f3`, so they are
used **only as proxy acquisition-unit identifiers** and are described as such in
the manuscript. Session id = `f1_f2_f3` + date.

## 3. Run order

`bash run_all.sh` runs the whole pipeline. Stage by stage:

| # | Command | Produces |
| - | ------- | -------- |
| 0 | `python verify_model.py --check-params --ablations --forward` | asserts the proposed model has **1,459,074** trainable parameters |
| 1 | `python make_splits.py --data-root ./data` | `results/split_manifest_<task>.csv`, `results/split_manifest_grouped_<task>.csv`, fold class counts, group structure |
| 2 | `python extract_image_features.py --data-root ./data` | `results/features_<task>.csv` |
| 2b | `python extract_image_features.py --data-root ./normal_pool --out-dir ./results/pool` | `results/pool/features_<task>.csv` — the **full normal pool** |
| 2c | `python rebuild_matched_subset.py --task <task> --positive-features ... --candidate-features ...` | `results/matched_subset_<task>.csv`, `tables/matched_subset_balance.tex` |
| 3 | `python near_duplicates.py` | threshold sweep, exact (MD5) duplicates, `results/split_manifest_dedup_<task>.csv` |
| 4 | `python trivial_baseline.py` | trivial-baseline metrics and the univariate image statistics |
| 5 | `python train_cv.py --task <task> --models lgca_net resnet50 ...` | checkpoints, `results/oof_<task>_<model>.csv`, `results/lr_selected_<task>.csv` |
| 5b | `python train_cv.py --task <task> --file-list results/matched_subset_<task>.csv --grouped --run-tag matched` | the **headline** run: same protocol on the confound-controlled subset |
| 6 | `python evaluate_cv.py` | pooled + fold-wise metrics, fold-wise DeLong/McNemar |
| 7 | `python train_seeds.py --task <task> --seeds 0 1 2 3 4` | `results/seeds_<task>.csv` |
| 8 | `python xai_localization_foldwise.py --task <task> --annotation-dir ./labels` | fold-wise localisation metrics |
| 8b | `python faithfulness.py --task <task>` | `results/faithfulness_<task>.csv`, perturbation curves |
| 9 | `python seed_stats.py` | seed table, pairwise tests, forest plot |
| 10 | `python make_tables.py` | regenerates every fragment; stubs the missing ones |

The acquisition-matched subset is the scientifically meaningful comparison; the
raw-corpus run (stage 5) should be reported alongside it as the "uncontrolled"
number, not instead of it:

```bash
# 1. features for the full normal pool (large candidate set)
python extract_image_features.py --data-root ./normal_pool --out-dir ./results/pool

# 2. match controls to positives on site + brightness
python rebuild_matched_subset.py --task you_chalk_brood \
    --positive-features results/features_you_chalk_brood.csv \
    --candidate-features results/pool/features_you_chalk_brood.csv \
    --match-level site --match-features lab_L --caliper 0.1 --n-per-class 2000

# 3. train with the identical protocol on the matched list
python train_cv.py --task you_chalk_brood --file-list results/matched_subset_you_chalk_brood.csv
```

`--file-list` rebuilds folds from the given `filename,label,path` CSV using the
same `StratifiedKFold(3, shuffle=True, random_state=42)` and stratified 25%
validation carve-out, so only the underlying image set differs.

Other sensitivity analyses reuse the same trainer with a different manifest:

```bash
python train_cv.py --task you_foulbrood --grouped                                  # hive-grouped CV
python train_cv.py --task you_foulbrood --manifest results/split_manifest_dedup_you_foulbrood.csv
```

## 3b. Run tags: which run feeds which manuscript table

`train_cv.py`, `evaluate_cv.py`, `train_seeds.py` and `seed_stats.py` accept
`--run-tag`, which suffixes every output file stem so several protocols coexist
in one results directory. No tag → the current, untagged names.

| Run | Command | Fragments | Manuscript |
| --- | ------- | --------- | ---------- |
| **matched** (headline) | `train_cv.py --file-list results/matched_subset_<task>.csv --run-tag matched`, then `evaluate_cv.py --run-tag matched`, `train_seeds.py --run-tag matched`, `seed_stats.py --run-tag matched` | `cv_metrics.tex`, `cv_foldwise.tex`, `cv_pairwise_foldwise.tex`, `ablation.tex`, `lr_selected.tex`, `xai_localization.tex`, `faithfulness.tex`, `seed_table_matched.tex`, `seed_pairwise_matched.tex` | **main text, Tables 9–13** |
| **original** (uncontrolled) | `train_cv.py --run-tag original`, then `evaluate_cv.py --run-tag original` | `cv_metrics_original.tex` | **supplement, Table S6** |
| **original seeds** (legacy 5-seed run) | `seed_stats.py` (no tag; reads the legacy `seed_table_<task>_95CI.csv`) | `seed_table.tex`, `seed_pairwise.tex` | **supplement, Tables S3/S4** |

The main-text cross-validation tables must come from the **matched** run: the
raw corpus confounds class with acquisition site and brightness, so the
`_original` numbers belong in the supplement as the uncontrolled comparison,
not in the headline.

`evaluate_cv.py --run-tag original` reads `oof_<task>_<model>_original.csv` and
writes `tables/cv_metrics_original.tex`; `seed_stats.py --run-tag matched` reads
`seeds_<task>_matched.csv` (or `results/matched/seeds_<task>.csv`) and writes
`tables/seed_table_matched.tex` and `tables/seed_pairwise_matched.tex`.

## 4. Script → manuscript mapping

| Script | Fragment / figure | Manuscript element |
| ------ | ----------------- | ------------------ |
| `make_splits.py` | `tables/fold_class_counts.tex` | fold/role class counts for both split schemes |
| `make_splits.py` | `tables/group_structure.tex` | sites, hive-proxies, sessions; how many hive-proxies span both classes |
| `trivial_baseline.py` | `tables/image_statistics.tex` | **Table 4** (univariate class comparison, native *and* 256×256, units stated) |
| `trivial_baseline.py` | `tables/trivial_baseline.tex` | model-free baselines on the same folds |
| `near_duplicates.py` | `tables/near_duplicates.tex` | near-duplicate audit: pHash threshold sweep, same- vs different-hive pairs, exact (MD5) duplicates, train/test crossings |
| `rebuild_matched_subset.py` | `tables/matched_subset_balance.tex` | SMD before/after acquisition matching; evidence the site/brightness confound is removed |
| `train_cv.py` | `tables/lr_selected.tex` | learning-rate search: validation BA at every grid value, selected rate in bold |
| `evaluate_cv.py` | `tables/cv_metrics.tex` | main CV comparison table (pooled, bootstrap CIs) |
| `evaluate_cv.py` | `tables/ablation.tex` | ablation: full vs. w/o cross-attention vs. CNN-only, with fold-wise DeLong/McNemar |
| `evaluate_cv.py --run-tag original` | `tables/cv_metrics_original.tex` | the same table on the unmatched corpus (supplement) |
| `evaluate_cv.py` | `tables/cv_foldwise.tex` | fold-wise mean ± SD |
| `evaluate_cv.py` | `tables/cv_pairwise_foldwise.tex` | fold-wise DeLong + McNemar, Holm-corrected |
| `seed_stats.py` | `tables/seed_table.tex` | across-seed results on the fixed held-out split |
| `seed_stats.py` | `tables/seed_pairwise.tex` | Welch t, Hedges' g, Holm-adjusted p, CI overlap |
| `seed_stats.py` | `figures/seed_forest.png` | forest plot, (a) chalkbrood / (b) foulbrood × BA/F1/AUROC |
| `xai_localization_foldwise.py` | `tables/xai_localization.tex`, `figures/xai_localization.png` | quantitative XAI localisation (does the map hit the lesion?) |
| `faithfulness.py` | `tables/faithfulness.tex` | XAI faithfulness (does the map identify the pixels the model uses?): perturbation AUCs, Average Drop, Increase in Confidence |
| `verify_model.py` | — | the parameter count quoted in the architecture section |

## 5. What changed relative to the original code

| Original | Revision |
| -------- | -------- |
| `KFold(3, shuffle=True, random_state=42)` — **not stratified** | `StratifiedKFold(3, shuffle=True, random_state=42)` |
| validation = the last 25% of the shuffled training indices | validation = stratified 25% (`train_test_split(stratify=y, random_state=42)`) |
| splits existed only in memory | split manifests written to CSV (`filename,label,group,fold,role`) |
| single split, single run | + hive-grouped CV, + de-duplicated CV, + 5 seeds on a fixed split |
| point metrics only | bootstrap CIs, fold-wise SD, DeLong, McNemar, Holm, Hedges' g |
| separate `Recall` and `Sensitivity` columns | one **Sensitivity** column (they are identical for a binary task) |
| qualitative Grad-CAM figures | fold-wise quantitative localisation on held-out positives only |

Unchanged, to keep the comparison honest: Adam, batch 32, ≤50 epochs, early
stopping patience 10, unweighted cross-entropy, 256×256 inputs scaled to
`[0, 1]`, no augmentation.

## 6. Model

`common/models.py` defines `LGCANet` — **L**ocal-query **G**lobal-key
**C**ross-**A**ttention Network, named for the fusion step, where the queries
come from the local CNN token grid and the keys/values from the globally
transformer-encoded tokens. `HCFNet` (the name used in earlier drafts) and
`Hybrid_CNN_Transformer_AblationModel` remain as aliases:

```
Conv7x7(3->64, s2, p3) + BN + GELU
Conv3x3(64->256, s2, p1) + BN
AdaptiveAvgPool2d(14, 14)              -> 196 tokens x 256
Linear(256->256) + pos_embed(1,196,256)
nn.TransformerEncoderLayer(d_model=256, nhead=8, ff=1024, dropout=0.1, relu, batch_first)
CrossAttention(dim=256, heads=8, qkv_bias=False)   # Q = CNN tokens, K/V = transformer tokens
LayerNorm(cross_attn_out + cnn_tokens)
mean-pool over tokens
Linear(256->512) + GELU + Dropout(0.3) + Linear(512->2)
```

Total **1,459,074** trainable parameters. Ablations: `use_cross_attn=False`
fuses additively (`LayerNorm(trans + cnn)`); `use_transformer=False` is the
CNN-only variant (`LayerNorm(cnn)`).

Registry keys are `lgca_net`, `lgca_net_no_cross_attn`, `lgca_net_cnn_only`;
`proposed`, `proposed_no_cross_attn`, `proposed_cnn_only` and `hcfnet` are
accepted as aliases. The key appears in output filenames, so use one spelling
per results directory.

Baselines are built through `MODEL_REGISTRY`. `resnet50`, `densenet121`,
`efficientnetv2` (`tf_efficientnetv2_s`), `nextvit_small`, `coatnet`
(`coatnet_0_rw_224`) and `crossvit` (`crossvit_small_240`, resized to 240 inside
a wrapper) come from `timm` with ImageNet-1K weights. `mambavision`, `lsnet_t`,
`conformer` and `mobileformer` need external repositories and raise a
descriptive `ImportError` until installed — see
[`models/external/README.md`](models/external/README.md).

## 7. Statistical conventions

* **Near-duplicate threshold**: pHash Hamming `≤ 4` by default. At `≤ 8` the hash
  links most of the corpus across *different* hives — these frames are dark and
  low-texture, so their DCT low-frequency blocks collide — which measures the
  hash failing, not duplication. The reported sweep over `{0, 2, 4, 6, 8}` makes
  that visible; byte-identical (MD5) duplicates are reported separately.
* **XAI faithfulness**: perturbation AUCs are the trapezoidal area of accuracy
  (w.r.t. the *originally predicted* class) against the deleted fraction
  `x ∈ {0.1,…,0.9}`, normalised by the range and ×100 — positive perturbation
  deletes the highest-attribution pixels (lower is better), negative the lowest
  (higher is better). Average Drop and Increase in Confidence follow Chattopadhay
  et al. (Grad-CAM++) on the image multiplied by the min-max normalised map.
  `make_tables.py` can also rebuild this fragment from the author's older
  per-fold exports in `results_POS_NEG/` and `results_DROP_INC/`, ignoring their
  trailing `mean`/`std` summary rows.
* **Bootstrap CIs**: seeded stratified percentile bootstrap, 2000 replicates.
* **DeLong**: fast midrank implementation for two *correlated* AUCs; applied
  per fold, so the two score vectors are paired image-by-image.
* **McNemar**: two-sided exact binomial by default, `--mcnemar midp` available.
* **Holm–Bonferroni**: applied across every comparison in a given table;
  `seed_stats.py --holm-scope {all,task,metric}` narrows the family if needed.
* **Welch's t** and **Hedges' g** for the across-seed comparisons. When only the
  legacy summary CSV is available, the SD is recovered from the reported CI as
  `SD = half_width · √n / t_{0.975,n−1}` with `n = 5`.

Numbers are formatted to 3 decimals throughout, except IoU (4 decimals).

## 8. LaTeX integration

Fragments are `tabular` **bodies only** — no `table`/`caption` wrapper — so the
manuscript supplies its own float:

```latex
\begin{table}[t]
  \centering
  \caption{Cross-validated performance ...}
  \label{tab:cv}
  \input{tables/cv_metrics.tex}
\end{table}
```

They use `booktabs` (`\toprule`, `\midrule`, `\bottomrule`). Placeholders emitted
by `make_tables.py` use `\textcolor{red}{...}`, so the preamble needs
`\usepackage{booktabs}` and `\usepackage{xcolor}`.

## 9. Smoke testing without the real data

```bash
python make_fake_data.py --out ./data_fake --per-class 40
python make_splits.py             --data-root ./data_fake
python extract_image_features.py  --data-root ./data_fake
python near_duplicates.py
python trivial_baseline.py --n-boot 300
python make_tables.py --data-root ./data_fake
```

`make_fake_data.py` writes 64×36 noise images with realistic file names and a
few deliberate near-duplicates. It exists only to exercise the CPU-only path;
its numbers are meaningless.

## 10. Layout

```
common/{data,metrics,models,seed,latex}.py   shared library
make_fake_data.py                            synthetic smoke-test data
verify_model.py                              parameter-count check
make_splits.py                               (1) manifests + group structure
extract_image_features.py                    (2) model-free descriptors
rebuild_matched_subset.py                    (2c) acquisition-matched subset
near_duplicates.py                           (3) pHash sweep + MD5 duplicate audit
trivial_baseline.py                          (4) trivial baselines + Table 4
train_cv.py                                  (5) CV training + LR selection
evaluate_cv.py                               (6) CV metrics + paired tests
train_seeds.py                               (7) seed variance
xai_localization_foldwise.py                 (8) XAI localisation
faithfulness.py                              (8b) XAI faithfulness
seed_stats.py                                (9) seed statistics + forest plot
make_tables.py                               (10) fragment regeneration
run_all.sh                                   ordered pipeline
models/external/README.md                    how to plug in external backbones
results/  tables/  figures/  model_save/     outputs
```

Every script has `--help`, and most support `--dry-run` for a fast path.

## Feeding the manuscript

The manuscript (`../manuscript/main.tex`, `supplementary.tex`) `\input`s every
results table from `tables/*.tex`. After the final runs, regenerate the fragments
directly into the manuscript directory and rebuild the PDF:

```bash
python make_tables.py --table-dir ../manuscript/tables
cd ../manuscript && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Fragments with no result file are written as red `[pending …]` markers so that an
incomplete build is visible in the PDF (`pdftotext main.pdf - | grep -c pending`
must be 0 before submission).
