#!/bin/bash

echo "=========================================="
echo "SelfClean Audio - Master Results Runner"
echo "=========================================="

echo
echo "Running file-level issue experiments..."
./scripts/get_file_results.sh

echo
echo "Running baseline experiments..."
./scripts/baselines/get_baseline_results.sh

echo
echo "Running LoRA experiments..."
./scripts/finetuning/run_lora_grid.sh

echo
echo "Running GTZAN experiments..."
./scripts/get_gtzan_results.sh

echo
echo "Running CSEM experiments..."
./scripts/get_csem_rankings.sh

echo
echo "All experiments completed!"
echo "Results are saved in the outputs/ directory"

echo
echo "Aggregating and visualizing runs"
python3 scripts/collect_results.py --base-dir outputs --filter all || true
python3 scripts/collect_results.py --base-dir outputs --filter synthetic || true
python3 scripts/collect_results.py --base-dir outputs --filter gtzan || true
python3 scripts/visualize_all_curves.py --base-dir outputs || true
