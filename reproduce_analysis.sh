#!/usr/bin/env bash
# CPU-only reproduction of every table and data figure in the manuscript from the
# released per-image predictions and image statistics (no images, no GPU needed).
#
#   bash reproduce_analysis.sh            # writes ./reproduced/{results,tables,figures}
#
# Expected runtime: ~2 min on a laptop CPU (hive-clustered bootstrap with 2,000 resamples dominates).
# The regenerated tables/*.tex are byte-identical to manuscript/tables/*.tex except
# for two cosmetic edits made by hand in the manuscript (1.2 -> 1.20 M, 31.15 -> 31.16 M).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HERE/reproduced}"
R="$OUT/results"; T="$OUT/tables"; F="$OUT/figures"
mkdir -p "$R" "$T" "$F"

# 1) flatten the released data into the results/ layout the analysis scripts expect
cp "$HERE"/data/splits/*.csv "$HERE"/data/features/*.csv \
   "$HERE"/data/predictions/oof_*.csv "$HERE"/data/predictions/lr_selected_*.csv \
   "$HERE"/data/predictions/train_log_*.csv \
   "$HERE"/data/metrics/*.csv "$HERE"/data/metrics/model_provenance.json \
   "$HERE"/data/xai/*_drop_inc.csv "$HERE"/data/xai/*_perturb_auc.csv "$R"/
rm -f "$R"/confound_reliance.csv     # recomputed below

A="$HERE/code/honeybee_extended_baselines/analysis"
TASKS="you_chalk_brood you_foulbrood"

# 2) acquisition-matched subsets, hive-clustered bootstrap CIs, fold-wise DeLong/McNemar
python "$A/matched_subsets.py"   --results-dir "$R" --out-dir "$R" --tables-dir "$T" --run-tag original --tasks $TASKS
# 3) confound-reliance surrogates (Section 5.5)
python "$A/confound_reliance.py" --results-dir "$R" --out-dir "$R" --run-tag original --tasks $TASKS
# 4) LaTeX table fragments (Tables 7, 9, 11-15, S2, S3)
python "$A/make_manuscript_tables.py" --results-dir "$R" --out-dir "$R" --tables-dir "$T" --run-tag original --tasks $TASKS
# 4b) revision-7 tables: training cost (Table 15) and confusion matrices (Table S5)
python "$HERE/code/make_extra_tables.py" --results-dir "$R" --tables-dir "$T"
# 5) figures 3-7 and 9 (vector PDF + 300-dpi PNG/TIFF)
( cd "$OUT" && python "$HERE/code/make_manuscript_figures_rev6.py" )   # reads ./results, writes ./figures_out
mv "$OUT"/figures_out/* "$F"/ && rmdir "$OUT"/figures_out

echo "done: tables in $T, figures in $F"
