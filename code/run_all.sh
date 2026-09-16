#!/usr/bin/env bash
# =============================================================================
#  run_all.sh — ONE-SHOT reproduction pipeline for the LGCA-Net honeybee paper
#  (revision 3).  Runs every stage in order and finally rebuilds the manuscript.
#
#  Usage
#    bash run_all.sh                          # everything, defaults below
#    DATA_ROOT=/data/bee NORMAL_POOL=/data/aihub_normal ANNOTATIONS=/data/aihub_json bash run_all.sh
#    SKIP_TRAINING=1 bash run_all.sh          # CPU-only audit stages + tables
#    STAGES="5 6 7" bash run_all.sh           # only the listed stages
#    GPU=1 bash run_all.sh                    # choose CUDA device
#
#  Inputs (edit the block below or export the variables)
#    DATA_ROOT     ./data/<task>/{normal,abnormal}/*.jpg      (task = you_chalk_brood, you_foulbrood)
#    NORMAL_POOL   folder with the FULL AI-Hub normal images (any layout; scanned recursively by
#                  extract_image_features.py --pool-dir) -> needed for the
#                  acquisition-matched ("headline") subsets. If absent, the matched stages are
#                  skipped and the manuscript keeps its red [pending] markers.
#    ANNOTATIONS   folder with the provider JSON files (lesion boxes) for the XAI localisation stage.
#    MANUSCRIPT    ../manuscript (tables are written into $MANUSCRIPT/tables and the PDF is rebuilt)
#
#  Stage map (manuscript table/figure fed by each stage)
#    0  verify_model            parameter count 1,459,074
#    1  make_splits             Table 3 (group structure), Table S2 (fold class counts)
#    2  extract_image_features  features for stages 3, 4 and matching
#    2b normal-pool features + rebuild_matched_subset   Table 7
#    3  near_duplicates         Table 6
#    4  trivial_baseline        Tables 4, 5
#    5  train_cv (matched, hive-grouped, LR search) -> 6 evaluate_cv: Tables 8, 9, 10, 11, 14
#    5o train_cv (original subsets)                 -> 6o evaluate_cv: Table S6
#    7  train_seeds + seed_stats (matched)          Tables 12, 13, Fig 5
#    7o seed_stats on original subsets              Tables S3, S4
#    8  xai_localization_foldwise (held-out only)   Table 15, Fig 8
#    8b faithfulness                                Table 16
#    9  make_tables -> $MANUSCRIPT/tables ; pdflatex main + supplementary ; pending-marker check
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

# ----------------------------------------------------------------- settings --
DATA_ROOT="${DATA_ROOT:-./data}"
NORMAL_POOL="${NORMAL_POOL:-./normal_pool}"
ANNOTATIONS="${ANNOTATIONS:-./labels}"
MANUSCRIPT="${MANUSCRIPT:-../manuscript}"
RESULTS="${RESULTS:-./results}"
TABLES="${TABLES:-./tables}"
FIGURES="${FIGURES:-./figures}"
CKPT="${CKPT:-./model_save}"
TASKS="${TASKS:-you_chalk_brood you_foulbrood}"
# Baselines available through timm out of the box; append  mambavision lsnet_t conformer mobileformer
# once the external adapters described in models/external/README.md are installed.
MODELS="${MODELS:-lgca_net lgca_net_no_cross_attn lgca_net_cnn_only resnet50 densenet121 efficientnetv2 nextvit_small coatnet crossvit}"
SEEDS="${SEEDS:-0 1 2 3 4}"
MATCH_LEVEL="${MATCH_LEVEL:-site}"
MATCH_FEATURES="${MATCH_FEATURES:-lab_L}"
CALIPER="${CALIPER:-0.1}"
N_PER_CLASS="${N_PER_CLASS:-2000}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"
STAGES="${STAGES:-0 1 2 2b 3 4 5 6 5o 6o 7 7o 8 8b 9}"
GPU="${GPU:-0}"
PY="${PY:-python3}"
export CUDA_VISIBLE_DEVICES="$GPU"

mkdir -p "$RESULTS" "$TABLES" "$FIGURES" "$CKPT"
LOG="$RESULTS/run_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

want()   { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }
banner() { echo; echo "=============================================================="; echo " $*"; echo "=============================================================="; }
fail=0
run()    { echo "+ $*"; "$@" || { echo "[FAILED] $*"; fail=1; return 1; }; }

MATCHED=0

# ------------------------------------------------------------------ stage 0 --
if want 0; then
  banner "0. Model sanity check (expects 1,459,074 trainable parameters)"
  run $PY verify_model.py --check-params --ablations --forward || run $PY verify_model.py --dry-run
fi

# ------------------------------------------------------------------ stage 1 --
if want 1; then
  banner "1. Split manifests (StratifiedGroupKFold by hive proxy, seed 42) -> Table 3, Table S2"
  run $PY make_splits.py --data-root "$DATA_ROOT" --tasks $TASKS --out-dir "$RESULTS" --table-dir "$TABLES"
fi

# ------------------------------------------------------------------ stage 2 --
if want 2; then
  banner "2. Model-free image features (numpy/PIL/cv2 only)"
  run $PY extract_image_features.py --data-root "$DATA_ROOT" --tasks $TASKS --out-dir "$RESULTS"
fi

# ----------------------------------------------------------------- stage 2b --
if want 2b; then
  banner "2b. Full normal pool -> acquisition-matched subsets -> Table 7"
  if [ -d "$NORMAL_POOL" ]; then
    run $PY extract_image_features.py --pool-dir "$NORMAL_POOL" --out-dir "$RESULTS/pool"
    for task in $TASKS; do
      if run $PY rebuild_matched_subset.py --task "$task" \
            --positive-features "$RESULTS/features_${task}.csv" \
            --candidate-features "$RESULTS/pool/features_normal_pool.csv" \
            --out-dir "$RESULTS" --table-dir "$TABLES" \
            --match-level "$MATCH_LEVEL" --match-features $MATCH_FEATURES \
            --caliper "$CALIPER" --n-per-class "$N_PER_CLASS"; then
        MATCHED=1
      else
        echo "[warn] $task: fewer than $N_PER_CLASS matched pairs. Inspect $TABLES/matched_subset_balance.tex;"
        echo "       widen CALIPER, enlarge NORMAL_POOL, or add --allow-fewer and report the achieved N in the paper."
      fi
    done
  else
    echo "[skip] NORMAL_POOL='$NORMAL_POOL' not found -> matched stages skipped; main-text Tables 7-16 stay [pending]."
  fi
else
  for task in $TASKS; do [ -f "$RESULTS/matched_subset_${task}.csv" ] && MATCHED=1; done
fi

# ------------------------------------------------------------------ stage 3 --
if want 3; then
  banner "3. Near-duplicate audit (pHash sweep + MD5) -> Table 6"
  run $PY near_duplicates.py --features-dir "$RESULTS" --manifest-dir "$RESULTS" --out-dir "$RESULTS" --table-dir "$TABLES" --tasks $TASKS
fi

# ------------------------------------------------------------------ stage 4 --
if want 4; then
  banner "4. Trivial baselines + global-statistics comparison -> Tables 4, 5"
  run $PY trivial_baseline.py --features-dir "$RESULTS" --manifest-dir "$RESULTS" --out-dir "$RESULTS" --table-dir "$TABLES" --tasks $TASKS
fi

if [ "$SKIP_TRAINING" = "1" ]; then
  echo; echo "SKIP_TRAINING=1 -> GPU stages 5-8b skipped."
else
  # ---------------------------------------------------------------- stage 5 --
  if want 5 && [ "$MATCHED" = "1" ]; then
    banner "5. HEADLINE run: hive-grouped stratified 3-fold CV on the matched subsets, per-model LR search"
    for task in $TASKS; do
      [ -f "$RESULTS/matched_subset_${task}.csv" ] || continue
      run $PY train_cv.py --data-root "$DATA_ROOT" --task "$task" --models $MODELS \
          --file-list "$RESULTS/matched_subset_${task}.csv" --grouped --run-tag matched \
          --out-dir "$RESULTS" --ckpt-dir "$CKPT"
    done
  fi
  if want 6 && [ "$MATCHED" = "1" ]; then
    banner "6. Evaluate matched run -> Tables 8, 9, 10, 11, 14"
    run $PY evaluate_cv.py --results-dir "$RESULTS" --out-dir "$RESULTS" --table-dir "$TABLES" --run-tag matched --reference lgca_net
  fi

  # --------------------------------------------------------------- stage 5o --
  if want 5o; then
    banner "5o. Original (unmatched) subsets, same protocol -> Table S6"
    for task in $TASKS; do
      run $PY train_cv.py --data-root "$DATA_ROOT" --task "$task" --models $MODELS \
          --grouped --run-tag original --manifest-dir "$RESULTS" --out-dir "$RESULTS" --ckpt-dir "$CKPT"
    done
  fi
  if want 6o; then
    run $PY evaluate_cv.py --results-dir "$RESULTS" --out-dir "$RESULTS" --table-dir "$TABLES" --run-tag original --reference lgca_net
  fi

  # ---------------------------------------------------------------- stage 7 --
  if want 7 && [ "$MATCHED" = "1" ]; then
    banner "7. Five-seed runs on the fixed fold-0 matched split -> Tables 12, 13, Fig 5"
    for task in $TASKS; do
      MF="$RESULTS/split_manifest_matched_subset_${task}_matched.csv"
      [ -f "$MF" ] || { echo "[warn] $MF missing (run stage 5 first)"; continue; }
      run $PY train_seeds.py --data-root "$DATA_ROOT" --task "$task" --models $MODELS --seeds $SEEDS \
          --fold 0 --manifest "$MF" --run-tag matched --out-dir "$RESULTS" --ckpt-dir "$CKPT"
    done
    run $PY seed_stats.py --results-dir "$RESULTS" --tasks $TASKS --run-tag matched \
        --table-dir "$TABLES" --figure-dir "$FIGURES" --out-dir "$RESULTS" --reference lgca_net
  fi

  # ---------------------------------------------------------------- stage 8 --
  if want 8 && [ "$MATCHED" = "1" ]; then
    banner "8. XAI localisation on HELD-OUT positives only -> Table 15, Fig 8  (set BOX_KEY_PATH in the script first)"
    for task in $TASKS; do
      MF="$RESULTS/split_manifest_matched_subset_${task}_matched.csv"
      run $PY xai_localization_foldwise.py --data-root "$DATA_ROOT" --task "$task" --model lgca_net \
          --annotation-dir "$ANNOTATIONS" --manifest "$MF" --run-tag matched \
          --ckpt-dir "$CKPT" --out-dir "$RESULTS" --table-dir "$TABLES" --figure-dir "$FIGURES"
    done
  fi
  if want 8b && [ "$MATCHED" = "1" ]; then
    banner "8b. XAI faithfulness (perturbation AUCs, Average Drop, Increase in Confidence) -> Table 16"
    for task in $TASKS; do
      MF="$RESULTS/split_manifest_matched_subset_${task}_matched.csv"
      run $PY faithfulness.py --data-root "$DATA_ROOT" --task "$task" --model lgca_net \
          --manifest "$MF" --run-tag matched --ckpt-dir "$CKPT" --out-dir "$RESULTS" --table-dir "$TABLES"
    done
  fi
fi

# ----------------------------------------------------------------- stage 7o --
if want 7o; then
  banner "7o. Seed statistics on the ORIGINAL subsets -> Tables S3, S4 (raw per-seed CSV preferred; legacy 95CI CSV fallback)"
  SEED_ARGS=""
  for task in $TASKS; do
    [ -f "$RESULTS/seeds_${task}.csv" ] && continue
    [ -f "$RESULTS/seed_table_${task}_95CI.csv" ] && SEED_ARGS="$SEED_ARGS ${task}=$RESULTS/seed_table_${task}_95CI.csv"
  done
  if [ -n "$SEED_ARGS" ]; then
    run $PY seed_stats.py --results-dir "$RESULTS" --tasks $TASKS --table-dir "$TABLES" --figure-dir "$FIGURES" --out-dir "$RESULTS" --summary-csv $SEED_ARGS
  else
    run $PY seed_stats.py --results-dir "$RESULTS" --tasks $TASKS --table-dir "$TABLES" --figure-dir "$FIGURES" --out-dir "$RESULTS" \
      || echo "[warn] no seed results for the original subsets; Tables S3/S4 stay [pending]"
  fi
fi

# ------------------------------------------------------------------ stage 9 --
if want 9; then
  banner "9. LaTeX fragments -> $MANUSCRIPT/tables, then rebuild main.pdf and supplementary.pdf"
  run $PY make_tables.py --data-root "$DATA_ROOT" --results-dir "$RESULTS" --table-dir "$TABLES" --figure-dir "$FIGURES"
  if [ -d "$MANUSCRIPT" ]; then
    mkdir -p "$MANUSCRIPT/tables"
    cp "$TABLES"/*.tex "$MANUSCRIPT/tables/"
    [ -f "$FIGURES/seed_forest_matched.png" ] && cp "$FIGURES/seed_forest_matched.png" "$MANUSCRIPT/seed_forest.png"
    [ -f "$FIGURES/xai_localization.png" ]   && cp "$FIGURES/xai_localization.png"   "$MANUSCRIPT/xai_localization.png"
    if command -v pdflatex >/dev/null; then
      ( cd "$MANUSCRIPT" \
        && pdflatex -interaction=nonstopmode main.tex >/dev/null \
        && bibtex main >/dev/null \
        && pdflatex -interaction=nonstopmode main.tex >/dev/null \
        && pdflatex -interaction=nonstopmode main.tex >/dev/null \
        && pdflatex -interaction=nonstopmode supplementary.tex >/dev/null \
        && pdflatex -interaction=nonstopmode supplementary.tex >/dev/null \
        && echo "built $MANUSCRIPT/main.pdf and $MANUSCRIPT/supplementary.pdf" ) || { echo "[FAILED] LaTeX build"; fail=1; }
      if command -v pdftotext >/dev/null; then
        NP=$(pdftotext "$MANUSCRIPT/main.pdf" - 2>/dev/null | grep -c "pending" || true)
        NS=$(pdftotext "$MANUSCRIPT/supplementary.pdf" - 2>/dev/null | grep -c "pending" || true)
        NB=$(pdftotext "$MANUSCRIPT/main.pdf" - 2>/dev/null | grep -c -i -E "significantly outperform|drastically|overwhelming|rigorous|Our Proposal|Co-Corresponding|Keywords:.*Honey bee" || true)
        echo; echo "PENDING MARKERS : main.pdf=$NP  supplementary.pdf=$NS   (both must be 0 before submission)"
        echo "BANNED PHRASES  : main.pdf=$NB   (must be 0)"
      fi
    else
      echo "[warn] pdflatex not found; fragments copied, compile the manuscript elsewhere."
    fi
  fi
fi

echo
if [ "$fail" = "0" ]; then echo "run_all.sh finished without failures. Log: $LOG"
else echo "run_all.sh finished WITH failures (see [FAILED] lines above). Log: $LOG"; fi
exit $fail
