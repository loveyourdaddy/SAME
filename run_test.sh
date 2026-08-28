#!/bin/bash
# Retarget the test pairs with a trained SAME model, then score the BVH output.
#
#   ./run_test.sh <model_epoch> [pairs_txt] [data_dir] [device]
#
# examples
#   conda activate same
#   ./run_test.sh 260718_truebone
#   ./run_test.sh 260718_truebone truebones_test.txt \
#       "TruebonesZoo_processed_byJH/motion/processed/"
#   ./run_test.sh truecycle/best truebones_test.txt \
#       "TruebonesZoo_processed_byJH/motion/processed/" cuda:1
#
# Outputs (both under result/<model_epoch prefix>/test/):
#   pair<idx>__..__SRC/TGT/OUT.bvh + retarget_log.csv   (from src/same/test.py)
#   metrics.csv                                          (from metric/metric.py)
set -e

MODEL_EPOCH=${1:?"usage: ./run_test.sh <model_epoch> [pairs_txt] [data_dir] [device]"}
PAIRS=${2:-truebones_test.txt}
DATA_DIR=${3:-"TruebonesZoo_processed_byJH/motion/processed/"}
DEVICE=${4:-cuda:0}

cd "$(dirname "$0")"                          # repo root (SAME_original)
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# mirror test.py's default: result/<model_epoch up to first '/'>/test
OUT_DIR="$(pwd)/result/${MODEL_EPOCH%%/*}/test"

echo "=== [1/2] retargeting (test.py) -> $OUT_DIR ==="
python src/same/test.py \
    --model_epoch "$MODEL_EPOCH" \
    --data_dir "$DATA_DIR" \
    --pairs_txt "$PAIRS" \
    --out_dir "$OUT_DIR" \
    --device "$DEVICE"

echo ""
echo "=== [2/2] evaluating metrics (metric.py) ==="
python metric/metric.py \
    --result_dir "$OUT_DIR" \
    --gt_dir data/Trueboness_processed_byVT/augmented \
    --pairs_txt  data/Trueboness_processed_byVT/processed/truebones_vt_groups_fold0_test.txt
    # --out_csv "$OUT_DIR/metrics.csv"

echo ""
echo "=== done -> $OUT_DIR/metrics.csv ==="
