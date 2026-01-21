#!/bin/bash

set -euo pipefail

echo "=========================================="
echo "SelfClean Audio - GTZAN Benchmark Runner"
echo "=========================================="
echo "- Evaluates methods on GTZAN known issues"
echo "- Saves per-method metrics in outputs/"
echo "=========================================="
echo

# Allow overriding paths via env vars
GTZAN_ROOT=${GTZAN_ROOT:-"./data/gtzan/genres"}
GTZAN_GT_FILE=${GTZAN_GT_FILE:-"./data/gtzan/gtzan_ground_truth_full_exact_rep.txt"}
GTZAN_PREP_FILE=${GTZAN_PREP_FILE:-"./data/gtzan/gtzan_ground_truth_prep.txt"}

echo "GTZAN_ROOT     : $GTZAN_ROOT"
echo "GTZAN_GT_FILE  : $GTZAN_GT_FILE (duplicates)"
echo "GTZAN_PREP_FILE: $GTZAN_PREP_FILE (mislabels)"
echo

if [ ! -d "$GTZAN_ROOT" ]; then
  echo "WARNING: GTZAN_ROOT not found: $GTZAN_ROOT"
  echo "Ensure GTZAN audio exists under genres/<class>/*.wav"
fi
if [ ! -f "$GTZAN_GT_FILE" ]; then
  echo "WARNING: GTZAN_GT_FILE not found: $GTZAN_GT_FILE"
fi
if [ ! -f "$GTZAN_PREP_FILE" ]; then
  echo "WARNING: GTZAN_PREP_FILE not found: $GTZAN_PREP_FILE"
fi

# Models and methods to evaluate
MODELS=("beats")
ND_METHODS=("embedding_distance" "dejavu")
LE_METHODS=("intra_extra_distance" "cleanlab")

# Duplicates benchmarking
echo "Running duplicates (near-duplicates) benchmarks on GTZAN..."
for model in "${MODELS[@]}"; do
  echo "Model: $model"
  for method in "${ND_METHODS[@]}"; do
    outdir="outputs/gtzan_${model}_${method}_duplicates"
    echo "  Method: $method -> $outdir"
    python3 -m selfclean_audio \
      --config "config/templates/gtzan_template.py" \
      --output-dir "$outdir" \
      EVAL_DATASET="gtzan" \
      ISSUE_TYPE="duplicates" \
      GTZAN_ROOT="$GTZAN_ROOT" \
      GTZAN_GT_FILE="$GTZAN_GT_FILE" \
      MODEL_TYPE="$model" \
      near_duplicate_method="$method"
  done
done

# Label errors benchmarking
echo
echo "Running label error benchmarks on GTZAN..."
if [ ! -f "$GTZAN_PREP_FILE" ]; then
  echo "WARNING: Skipping label error runs: GTZAN_PREP_FILE not found: $GTZAN_PREP_FILE"
else
  for model in "${MODELS[@]}"; do
    echo "Model: $model"
    for method in "${LE_METHODS[@]}"; do
      outdir="outputs/gtzan_${model}_${method}_label_errors"
      echo "  Method: $method -> $outdir"
      python3 -m selfclean_audio \
        --config "config/templates/gtzan_template.py" \
        --output-dir "$outdir" \
        EVAL_DATASET="gtzan" \
        ISSUE_TYPE="label_errors" \
        GTZAN_ROOT="$GTZAN_ROOT" \
        GTZAN_PREP_FILE="$GTZAN_PREP_FILE" \
        MODEL_TYPE="$model" \
        label_error_method="$method"
    done
  done
fi

echo
echo "GTZAN benchmarking completed."
echo "Per-method metrics saved as Score-<issue>.csv in each output directory."
