#!/usr/bin/env bash
BEE=${BEE:-/path/to/code}                 # root of this code/ folder
EXT=${EXT:-/path/to/external_repos}       # sibling folder holding the LSNet and Conformer repositories
cd "$BEE" || exit 1

echo "── 외부 경로 사전 확인"
for p in "$EXT/lsnet/model/lsnet.py" \
         "$EXT/Conformer/conformer.py" \
         "$BEE/checkpoints/Conformer_small_patch16.pth" \
         "$BEE/models/external/adapters.py" \
         "$BEE/models/__init__.py"; do
  if [ -e "$p" ]; then echo "   ok  $p"
  else echo "   ✗  없음: $p"; exit 1; fi
done

export PYTHONPATH="$EXT/lsnet:$EXT/Conformer:${PYTHONPATH:-}"
export EXTERNAL_CKPT_DIR="$BEE/checkpoints"
export HF_HOME="$BEE/.hf_cache"
export USE_TF=0 USE_JAX=0

CODE_ROOT=. DATA_ROOT=./data GPU=1 \
  NEW_MODELS="conformer lsnet_t mambavision" \
  FIGURES=./figures_extended \
  bash honeybee_extended_baselines/run_extended_baselines.sh
