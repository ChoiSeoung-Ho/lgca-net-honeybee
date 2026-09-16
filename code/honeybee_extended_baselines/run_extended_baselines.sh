#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Add Conformer, Mobile-Former, LSNet-T and MambaVision to the audited protocol
# and regenerate every table and figure that depends on the model set.
#
#   CODE_ROOT=/path/to/code DATA_ROOT=/path/to/data GPU=0 bash run_extended_baselines.sh
#
# Stages (select with STAGES="1 2 3", default all):
#   1  verify the four backbones build, load ImageNet weights and forward
#   2  learning-rate search + hive-grouped 3-fold training, both tasks
#   3  pooled/fold-wise metrics + fold-wise DeLong/McNemar with Holm
#   4  matched subsets, confound-reliance analysis, tables and figures
#
# Nothing here overwrites the nine existing models: the new runs write
# oof_<task>_<model>_<RUN_TAG>.csv alongside them and every downstream script
# discovers the union.
# ---------------------------------------------------------------------------
set -uo pipefail

CODE_ROOT="${CODE_ROOT:-.}"
DATA_ROOT="${DATA_ROOT:-./data}"
RESULTS="${RESULTS:-$CODE_ROOT/results}"
TABLES="${TABLES:-$CODE_ROOT/tables}"
FIGURES="${FIGURES:-$CODE_ROOT/../manuscript}"
RUN_TAG="${RUN_TAG:-original}"
TASKS="${TASKS:-you_chalk_brood you_foulbrood}"
NEW_MODELS="${NEW_MODELS:-conformer mobileformer lsnet_t mambavision}"
GPU="${GPU:-0}"
STAGES="${STAGES:-1 2 3 4}"
PY="${PY:-python}"
ANALYSIS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/analysis"
LOG="$RESULTS/run_extended_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$RESULTS" "$TABLES" "$FIGURES"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$CODE_ROOT:$(dirname "$ANALYSIS"):${PYTHONPATH:-}"

say() { echo -e "\n=== $* ===" | tee -a "$LOG"; }
run() { echo "+ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }
has() { [[ " $STAGES " == *" $1 "* ]]; }

say "extended baselines: $NEW_MODELS   (log: $LOG)"

# --------------------------------------------------------------- 1. verify --
if has 1; then
  say "stage 1 - verifying the external backbones (no training)"
  run "$PY" "$(dirname "$ANALYSIS")/verify_external_models.py" \
      --code-root "$CODE_ROOT" --models $NEW_MODELS --out "$RESULTS/model_provenance.json"
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "
[stop] At least one backbone is not installed. Nothing was trained.
       Follow models/external/README.md, then re-run.
       To proceed with only the backbones that do build, set
         NEW_MODELS=\"<the ones that passed>\"
" | tee -a "$LOG"
    exit 1
  fi
  # measure the nine existing models too, so the parameter column is complete
  run "$PY" "$(dirname "$ANALYSIS")/verify_external_models.py" \
      --code-root "$CODE_ROOT" --models all --out "$RESULTS/model_provenance.json" || true
fi

# ---------------------------------------------------------------- 2. train --
if has 2; then
  say "stage 2 - learning-rate search and hive-grouped 3-fold training"
  for task in $TASKS; do
    run "$PY" "$CODE_ROOT/train_cv.py" --data-root "$DATA_ROOT" --task "$task" \
        --models $NEW_MODELS --grouped --run-tag "$RUN_TAG" \
        --manifest-dir "$RESULTS" --out-dir "$RESULTS" \
        --ckpt-dir "$CODE_ROOT/model_save" --device cuda \
      || { echo "[stop] training failed for $task" | tee -a "$LOG"; exit 2; }
  done
fi

# ------------------------------------------------------------- 3. evaluate --
if has 3; then
  say "stage 3 - metrics and fold-wise paired tests over ALL models"
  for task in $TASKS; do
    run "$PY" "$CODE_ROOT/evaluate_cv.py" --task "$task" --run-tag "$RUN_TAG" \
        --results-dir "$RESULTS" --out-dir "$RESULTS" || true
  done
fi

# -------------------------------------------------------------- 4. analyse --
if has 4; then
  say "stage 4 - matched subsets, confound reliance, tables and figures"
  run "$PY" "$ANALYSIS/matched_subsets.py" --results-dir "$RESULTS" --out-dir "$RESULTS" \
      --tables-dir "$TABLES" --run-tag "$RUN_TAG" --tasks $TASKS || exit 3
  run "$PY" "$ANALYSIS/confound_reliance.py" --results-dir "$RESULTS" --out-dir "$RESULTS" \
      --run-tag "$RUN_TAG" --tasks $TASKS || exit 3
  run "$PY" "$ANALYSIS/make_manuscript_tables.py" --results-dir "$RESULTS" --out-dir "$RESULTS" \
      --tables-dir "$TABLES" --run-tag "$RUN_TAG" --tasks $TASKS || exit 3
  run "$PY" "$ANALYSIS/make_manuscript_figures.py" --results-dir "$RESULTS" --out-dir "$RESULTS" \
      --figures-dir "$FIGURES" --run-tag "$RUN_TAG" --tasks $TASKS || exit 3
fi

say "done"
echo "
Next:
  1. check results/model_provenance.json - any model with pretrained_loaded=false
     must be reported as randomly initialised, not as an ImageNet baseline;
  2. rebuild the manuscript (the table fragments in $TABLES and the figures in
     $FIGURES are now regenerated with the full model set);
  3. update the prose that quotes counts ('six pretrained baselines', 'nine
     networks', the number of Holm-significant comparisons) - grep for those
     phrases, they are the only places a model count is written by hand.
" | tee -a "$LOG"
