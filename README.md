# lgca-net-honeybee — code and data for "A Hybrid Convolutional Neural Network–Transformer Architecture with Cross-Attention Fusion for Honeybee Disease Classification"

Author: Seoung-Ho Choi (Hansung University / Korea University College of Medicine)
Manuscript: submitted to *Ecological Informatics* (revision 7, 2026-09-16)
Repository: https://github.com/ChoiSeoung-Ho/lgca-net-honeybee — archived version: https://doi.org/10.5281/zenodo.XXXXXXX
Licence: MIT for the code and derived data in this repository; the source images belong to AI-Hub and are **not** redistributed (see `data/README.md` and `docs/AIHUB_ACCESS_GUIDE.md`, an English step-by-step guide to obtaining them).

This repository follows the journal's reproducibility policy (Huettmann & Arhonditsis, 2023,
*Ecological Informatics* 76, 102132): every number, table and data figure in the article can be
regenerated from files in this repository with one command and without the images
(`make reproduce-tables`, ~2 min on a laptop CPU), and the full pipeline can be re-run from the
images once they have been obtained from the public portal.

**Quick start**

```bash
git clone https://github.com/ChoiSeoung-Ho/lgca-net-honeybee && cd lgca-net-honeybee
pip install numpy scipy pandas scikit-learn matplotlib Pillow
make reproduce-tables        # → ./reproduced/{results,tables,figures}
```

**What changed in revision 7** (relative to the 2026-09-08 package): `code/make_extra_tables.py`
(training-cost table and confusion-matrix table, Table 16 and Table S5 of the article), `Makefile`,
`docs/AIHUB_ACCESS_GUIDE.md`; the ablation variant without cross-attention is now labelled as what
the code implements, *position-wise additive fusion* (`LayerNorm(trans_tokens + cnn_tokens)`), not
concatenation; the 0.5 reference line was removed from the deletion-curve panels of Figure 9; the
F1 column of the performance tables is documented as the positive-class F1. No numerical result changed.

This package contains (i) the complete source code of the study, (ii) every
derived data file needed to reproduce **all tables and data figures of the
manuscript on a CPU without the images**, and (iii) the split manifests,
image lists and matched-subset lists needed to re-run the full GPU pipeline
after downloading the public images.

```
lgca-net-honeybee/
├── README.md                     ← this file
├── Makefile                      ← `make reproduce-tables`
├── docs/AIHUB_ACCESS_GUIDE.md    ← English guide to downloading the images from AI-Hub
├── LICENSE                       ← MIT (code) — see data/README.md for the image licence
├── CITATION.cff
├── reproduce_analysis.sh         ← CPU-only: regenerates every table and data figure (~2 min); called by the Makefile
├── code/                         ← full pipeline (audit, splits, matching, training, evaluation, XAI)
│   ├── run_all.sh                ← stage-by-stage driver for the GPU pipeline
│   ├── requirements.txt
│   ├── README_pipeline.md        ← per-script documentation (original code README)
│   ├── common/                   ← LGCA-Net and baseline registry (models.py), data loading, metrics, LaTeX helpers
│   ├── models/external/          ← adapters for Conformer-S, LSNet-T, MambaVision-T (+ Mobile-Former stub)
│   ├── honeybee_extended_baselines/  ← pre-flight verifier, driver scripts, and the model-agnostic
│   │                                  analysis layer (matched subsets, confound reliance, tables, figures)
│   └── make_manuscript_figures_rev6.py  ← Figures 3–7 and 9
├── data/                         ← derived data (no images; see data/README.md)
│   ├── image_lists/              ← the 1,000 file names per task (500 normal + 500 disease-positive)
│   ├── splits/                   ← split manifests (hive-grouped, stratified-ungrouped, de-duplicated) and fold class counts
│   ├── matched_subsets/          ← acquisition-matched image lists, matched pairwise tests, matched_report.json
│   ├── features/                 ← global image statistics per image, acquisition-unit structure, near-duplicate audit
│   ├── predictions/              ← per-image out-of-fold predictions of all 12 networks, training logs, LR search
│   ├── metrics/                  ← pooled/fold-wise metrics, fold-wise DeLong/McNemar, confound reliance, model provenance
│   └── xai/                      ← faithfulness (drop/increase, deletion-curve AUC), sensitivity and stability CSVs
└── logs/                         ← server logs of the extended-baseline runs (Conformer-S, LSNet-T, MambaVision-T)
```

## 1. Two levels of reproduction

### Level A — every table and figure, CPU only, no images (verified)

```bash
pip install numpy scipy pandas scikit-learn matplotlib Pillow
bash reproduce_analysis.sh            # → ./reproduced/{results,tables,figures}
```

This re-runs the acquisition matching, the hive-clustered bootstrap, the
fold-wise DeLong and exact McNemar tests with Holm correction, the
confound-reliance surrogates and the LaTeX table generator from the released
per-image predictions and image statistics. The regenerated fragments
(`cv_metrics_original.tex`, `cv_metrics_matched.tex`, `trivial_baseline.tex`,
`ablation.tex`, `matched_subset_balance.tex`, `pairwise_summary.tex`,
`cv_pairwise_original.tex`, `cv_pairwise_matched.tex`, `confound_reliance.tex`,
`training_cost.tex`, `confusion_matrices.tex`)
are byte-identical to the fragments compiled into the revision-7 manuscript
(`manuscript/tables/` of the submission), and `confound_reliance.csv` is reproduced exactly. Figures 3–7 and 9
are written as vector PDF and 300-dpi PNG/TIFF.

### Level B — full pipeline from the images (GPU)

1. Obtain the images from AI-Hub (Dataset No. 71667, free registration and
   data-use agreement; see `data/README.md`) and place the files listed in
   `data/image_lists/` under `code/data/<task>/{normal,abnormal}/`.
2. Install the environment (Python 3.11, PyTorch 2.9.0, CUDA 12.8, `timm`):
   ```bash
   pip install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu128
   pip install -r code/requirements.txt
   ```
3. Run the pipeline from `code/`:
   ```bash
   cd code
   python verify_model.py --check-params --ablations --forward     # asserts 1,459,074 parameters
   python make_splits.py --data-root ./data                        # hive-grouped stratified 3-fold manifests
   python extract_image_features.py --data-root ./data             # global image statistics
   python near_duplicates.py                                       # perceptual-hash audit
   python train_cv.py --task you_chalk_brood --grouped --run-tag original \
          --models lgca_net lgca_net_no_cross_attn lgca_net_cnn_only resnet50 densenet121 \
                   efficientnetv2 coatnet crossvit nextvit_small
   python train_cv.py --task you_foulbrood   --grouped --run-tag original --models ...   # same list
   python evaluate_cv.py --run-tag original                         # pooled + fold-wise metrics, DeLong/McNemar
   ```
   `run_all.sh` chains these stages.
4. Extended baselines (Conformer-S, LSNet-T, MambaVision-T). Install the
   external repositories as described in `code/models/external/README.md`,
   then:
   ```bash
   python honeybee_extended_baselines/patch_models_registry.py --code-root .   # idempotent
   python honeybee_extended_baselines/verify_external_models.py --code-root . \
          --models conformer lsnet_t mambavision                                 # builds, counts params, forward pass
   CODE_ROOT=. DATA_ROOT=./data GPU=0 bash honeybee_extended_baselines/run_extended_baselines.sh
   ```
   The verifier writes `results/model_provenance.json` (weights source,
   checkpoint hash, measured parameter count); the copy produced for the
   manuscript is in `data/metrics/`. Mobile-Former has no official code or
   ImageNet checkpoint; the adapter refuses to build it as a pretrained
   baseline, and it is therefore not reported.
5. Analysis layer (identical to Level A, now on your own predictions):
   ```bash
   A=honeybee_extended_baselines/analysis
   python $A/matched_subsets.py     --results-dir results --out-dir results --tables-dir tables --run-tag original --tasks you_chalk_brood you_foulbrood
   python $A/confound_reliance.py   --results-dir results --out-dir results --run-tag original --tasks you_chalk_brood you_foulbrood
   python $A/make_manuscript_tables.py --results-dir results --out-dir results --tables-dir tables --run-tag original --tasks you_chalk_brood you_foulbrood
   ```

## 2. Protocol implemented by the code (as reported in the manuscript)

* Two balanced binary tasks from AI-Hub dataset 71667: `you_chalk_brood`
  (disease code 002) and `you_foulbrood` (003); 500 normal + 500 disease-positive
  images each, the normal images shared between tasks.
* Audit: CIELAB/RGB statistics, JPEG size, Otsu-region statistics
  (`extract_image_features.py`), Welch tests with Hedges' g and Holm correction,
  model-free classifiers on six global statistics (`trivial_baseline.py`,
  re-fitted on the hive-grouped folds inside `analysis/matched_subsets.py`),
  64-bit DCT perceptual-hash near-duplicate sweep (`near_duplicates.py`).
* Splits: `StratifiedGroupKFold(3)` grouped by the hive proxy `f1_f2_f3`,
  seed 42; stratified 25 % validation carve-out per fold (`make_splits.py`).
* Acquisition-matched subsets: same-site restriction and greedy 1:1 nearest
  neighbour matching on mean L\* without replacement, caliper 0.2 pooled SD
  (`analysis/matched_subsets.py`; 109 pairs Chalkbrood, 174 pairs Foulbrood).
  Deep networks are evaluated on their out-of-fold predictions restricted to
  the matched images; model-free classifiers are re-fitted on the matched images.
* Training: Adam (0.9, 0.999), batch 32, ≤ 50 epochs, early stopping on
  validation loss (patience 10), unweighted cross-entropy, 256×256 input scaled
  to [0, 1], no augmentation, per-model learning-rate search on the fold-0
  validation split (`train_cv.py`), seed 42, deterministic cuDNN.
* Statistics: 2,000-resample stratified percentile bootstrap (image-level for
  the original subsets, hive-clustered for the matched subsets), fold-wise
  DeLong and exact McNemar tests with Holm–Bonferroni correction over all
  model–fold pairs (33 per task and setting), implemented in `common/metrics.py`.
* Confound reliance (Section 5.6): logistic regression on six global
  statistics trained, on the same folds, to predict each network's own
  decision, referenced against the same regression trained on the disease
  label (`analysis/confound_reliance.py`).
* Post-hoc attribution (Section 5.7): Grad-CAM, gradient saliency, LIME and
  kernel SHAP with perturbation-based faithfulness metrics
  (`faithfulness.py`, `dropInc*.py`, `auc*Perturb.py`, `DropInc_auc_util/`,
  `PosNeg_auc_util/`). As stated in the manuscript, the reported faithfulness
  values were computed with the earlier EfficientNetV2-based configuration on
  the same folds; the per-fold CSVs are in `data/xai/`.
  `xai_localization_foldwise.py` implements the lesion-box localisation
  metrics; it was not run because the matching annotation export was not
  available.

## 3. Which file produces which manuscript element

| Manuscript element | Produced by | Data in this package |
|---|---|---|
| Table 4–5 (dataset, acquisition structure) | `make_splits.py`, `extract_image_features.py` | `data/features/group_structure_*.csv` |
| Table 7 (image statistics) | `trivial_baseline.py` | `data/features/image_statistics_*.csv` |
| Table 8 (model-free classifiers) | `analysis/matched_subsets.py` | `data/matched_subsets/matched_report.json` (`trivial_original`) |
| Table 9 (near-duplicates) | `near_duplicates.py` | `data/features/near_duplicate*_*.csv` |
| Table 10 (matched balance) | `analysis/matched_subsets.py` | `data/matched_subsets/matched_*.csv` |
| Table 11 (learning rates) | `train_cv.py` | `data/predictions/lr_selected_*.csv` |
| Table 12 (original subsets) | `evaluate_cv.py` | `data/metrics/cv_metrics_{pooled,foldwise}_original.csv` |
| Table 13 (matched subsets) | `analysis/matched_subsets.py` | `data/matched_subsets/matched_report.json` |
| Table 14, S2, S3 (paired tests) | `evaluate_cv.py`, `analysis/matched_subsets.py` | `data/metrics/cv_pairwise_foldwise_original.csv`, `data/matched_subsets/matched_pairwise_*.csv` |
| Table 15 (ablation) | `analysis/make_manuscript_tables.py` | as Tables 12–13 |
| Table 16 (training cost), Table S5 (confusion matrices) | `make_extra_tables.py` | `data/predictions/train_log_*.csv`, `data/predictions/oof_*.csv` |
| Table 17 (confound reliance) | `analysis/confound_reliance.py` | `data/metrics/confound_reliance.csv` |
| Table 18, Figure 9 (faithfulness) | `faithfulness.py` and the `DropInc`/`PosNeg` utilities | `data/xai/*.csv` |
| Table S1 (fold composition) | `make_splits.py` | `data/splits/fold_class_counts_*.csv` |
| Table S4 (provenance) | `verify_external_models.py` | `data/metrics/model_provenance.json` |
| Figures 3–7 | `make_manuscript_figures_rev6.py` | `data/features`, `data/matched_subsets`, `data/metrics` |
| Figures 1–2 | TikZ sources in the manuscript package | — |

## 4. Notes on the released data

* `data/predictions/oof_<task>_<model>_original.csv`: one row per image
  (`filename, fold, y_true, prob_pos, pred, loss`) from the fold whose test
  partition contained the image; these are the inputs to every statistic.
* `data/predictions/train_log_<task>_<model>_fold<k>.csv`: per-epoch training/validation loss and
  validation balanced accuracy for every model and fold. `train_cv.py` writes these files with an
  additional `_original` run-tag suffix; the suffix was removed here only so that every file name
  stays within the 64-character limit of the submission system (all are from the `original` run).
* `data/predictions/lr_selected_*.csv`: validation balanced accuracy at every
  candidate learning rate; `selected = 1` marks the rate used for all folds.
  The Conformer-S / Foulbrood rows were transcribed from
  `logs/run_extended_20260907_085736.log` because the server overwrote that
  CSV in a later partial run; all other rows are the original files.
* `data/splits/split_manifest_grouped_*.csv` is the split used for every
  reported result; `split_manifest_*.csv` (ungrouped stratified) and
  `split_manifest_dedup_*.csv` are the comparison splits discussed in the
  near-duplicate audit.
* Model checkpoints (6.3 GB) are not included; they can be regenerated with
  `train_cv.py` and are available from the author on request.
* The images themselves are not redistributed (AI-Hub licence); see
  `data/README.md`.
