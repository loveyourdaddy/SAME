#!/bin/bash
# Train SAME on the VT group folds, one fold after another.
#
#   ./run_folds.sh [folds] [device] [seed]
#
# examples
#   conda activate same
#   ./run_folds.sh                  # folds 0 1 2 on cuda:0
#
# Without activating the env first, point PYTHON at it:
#   PYTHON=~/anaconda3/envs/same/bin/python ./run_folds.sh
#   ./run_folds.sh "0 2" cuda:0     # only folds 0 and 2
#
# Each fold reads config/260803_cfg_VT_fold<N>.yml (which differs from the
# 260803_cfg_VT_split run only in train_data.pairs_txt) and writes
#   result/260803_cfg_VT_fold<N>/{model_*.pt,last_model.pt,logs/,train.log}
#
# Folds run sequentially on purpose: one 10GB GPU will not hold three
# batch_size=128 runs at once.
set -eo pipefail

FOLDS=${1:-"0 1 2"}
DEVICE=${2:-cuda:0}
SEED=${3:-0}
PYTHON=${PYTHON:-python}

cd "$(dirname "$0")"                          # repo root (SAME_original)
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

for f in $FOLDS; do
    EXP="260803_cfg_VT_fold${f}"
    [ -f "config/${EXP}.yml" ] || { echo "missing config/${EXP}.yml"; exit 1; }
    PAIRS=$(grep -m1 'pairs_txt:' "config/${EXP}.yml" | awk '{print $2}')
    [ -f "data/Trueboness_processed_byVT/processed/${PAIRS}" ] \
        || { echo "missing pair list ${PAIRS}"; exit 1; }

    mkdir -p "result/$EXP"
    echo ""
    echo "=== fold $f | cfg ${EXP}.yml | pairs ${PAIRS} | $DEVICE seed $SEED ==="
    "$PYTHON" src/same/train.py --exp "$EXP" --cfg "$EXP" \
        --device "$DEVICE" --seed "$SEED" 2>&1 | tee "result/$EXP/train.log"
    echo "=== fold $f done -> result/$EXP/last_model.pt ==="
done

echo ""
echo "all folds done. to score one:"
echo "  ./run_test.sh 260803_cfg_VT_fold0 truebones_vt_groups_fold0_test.txt \\"
echo "      Trueboness_processed_byVT/processed/"
