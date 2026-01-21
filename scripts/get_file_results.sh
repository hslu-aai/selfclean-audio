#!/bin/bash

models=("beats" "cav-mae" "eat" "m2d" "clmr" "clmr-dataspecific")
# These would be "all" issues but we only consider the ones below for a faster execution
# issue_types=("duplicates" "noisy_duplicates" "cropped_duplicates" "mixed_duplicates" "combined_duplicates" "label_errors" "off_topic_noise" "off_topic_external" "off_topic_corrupted" "off_topic_combined")
issue_types=("combined_duplicates" "label_errors" "off_topic_combined")
frac_errors=("0.05" "0.1" "0.2")

for model in "${models[@]}"; do
  for issue_type in "${issue_types[@]}"; do
    for frac_error in "${frac_errors[@]}"; do
      echo "Running ${model} with ${issue_type} at frac_error=${frac_error}..."

      python3 -m selfclean_audio \
        --config "config/templates/file_template.py" \
        --output-dir "outputs/${model}_esc50_${issue_type}_frac${frac_error}" \
        MODEL_TYPE="$model" \
        ISSUE_TYPE="$issue_type" \
        FRAC_ERROR="$frac_error"

      echo "Completed ${model} with ${issue_type} at frac_error=${frac_error}"
    done
  done
done

echo "All file experiments completed!"

echo "Running CSEM rankings..."
bash "$(dirname "$0")/get_csem_rankings.sh"
