# Data

## Images (not redistributed)

All images come from the public **AI-Hub Honeybee Disease Diagnosis Image Dataset
(Dataset No. 71667)**, published by the National Information Society Agency,
Republic of Korea: https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71667

Access requires free registration and acceptance of the provider's data-use
agreement, which does not permit redistribution of the images. They are therefore
not included here. `image_lists/<task>_{normal,abnormal}.txt` gives the exact file
names (500 + 500 per task) used in the study; the file names are the provider's
own and can be matched directly against the downloaded archive. Place them under
`code/data/<task>/normal/` and `code/data/<task>/abnormal/` to re-run the pipeline.

File names have the form `f1_f2_f3_YYYYMMDDhhmmss_f5_f6_f7_f8.jpg`; the last field
is the disease code (000 normal, 002 Chalkbrood, 003 Foulbrood). `f1_f2` is used as
a proxy acquisition-site identifier and `f1_f2_f3` as a proxy hive identifier; the
provider does not document their semantics.

## Derived data (included)

| Folder | Contents |
|---|---|
| `image_lists/` | file names of the 1,000 images per task |
| `splits/` | `split_manifest_grouped_<task>.csv` (hive-grouped stratified 3-fold split used for all results: `filename, path, label, class_dir, group, site, session, fold, role`), the ungrouped stratified split, the de-duplicated split, and per-fold class counts |
| `matched_subsets/` | `matched_<task>.csv` (matched images with site, hive proxy and mean L\*), `matched_pairwise_<task>.csv` (fold-wise DeLong/McNemar on the matched images), `matched_report.json` (balance, CIs, model-free classifiers on both settings) |
| `features/` | `features_<task>.csv` (global image statistics per image at native and 256×256 resolution), `image_statistics_<task>.csv` (class comparison), `group_structure_<task>.csv`, near-duplicate sweep, components and pairs |
| `predictions/` | out-of-fold predictions of the 12 networks (`oof_*`), per-epoch training logs (`train_log_*`), learning-rate search (`lr_selected_*`); `trivial_oof_*` are the model-free classifiers under the ungrouped split kept for the near-duplicate/leakage comparison only — the hive-grouped values reported in the paper are in `matched_subsets/matched_report.json` |
| `metrics/` | pooled and fold-wise metrics (`cv_metrics_*_original.csv`), fold-wise paired tests (`cv_pairwise_foldwise_original.csv`), confound-reliance surrogates (`confound_reliance.csv`), model provenance (`model_provenance.json`) |
| `xai/` | per-fold faithfulness metrics (average drop / increase in confidence; positive and negative perturbation AUC) for Grad-CAM, gradient saliency, LIME and SHAP, plus the sensitivity and stability CSVs |

All CSVs are UTF-8 with a header row. Tasks are named `you_chalk_brood` and
`you_foulbrood` throughout the code and data.
