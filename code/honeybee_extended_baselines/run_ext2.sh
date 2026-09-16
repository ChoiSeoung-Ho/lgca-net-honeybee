#!/usr/bin/env bash
BEE=${BEE:-/path/to/code}                 # root of this code/ folder
EXT=${EXT:-/path/to/external_repos}       # sibling folder holding the LSNet and Conformer repositories
cd "$BEE" || exit 1
export PYTHONPATH="$EXT/lsnet:$EXT/Conformer:${PYTHONPATH:-}"
export EXTERNAL_CKPT_DIR="$BEE/checkpoints"
export HF_HOME="$BEE/.hf_cache"
export USE_TF=0 USE_JAX=0
GPU=${GPU:-1}

echo "=== [A] foulbrood 남은 2종 학습 (stage 2) ==="
CODE_ROOT=. DATA_ROOT=./data GPU=$GPU \
  TASKS="you_foulbrood" NEW_MODELS="lsnet_t mambavision" \
  STAGES="2" FIGURES=./figures_extended \
  bash honeybee_extended_baselines/run_extended_baselines.sh || { echo "[A 실패]"; exit 2; }

echo "=== [B] 표·그림 생성 (stage 3,4) ==="
CODE_ROOT=. DATA_ROOT=./data GPU=$GPU \
  TASKS="you_chalk_brood you_foulbrood" \
  NEW_MODELS="conformer lsnet_t mambavision" \
  STAGES="3 4" FIGURES=./figures_extended \
  bash honeybee_extended_baselines/run_extended_baselines.sh || { echo "[B 실패]"; exit 3; }

echo "=== 전부 완료 $(date +%F\ %T) ==="
