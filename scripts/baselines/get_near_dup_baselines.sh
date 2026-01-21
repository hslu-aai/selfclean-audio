#!/bin/bash

echo "Running near duplicate detection baseline evaluations..."

models=("beats" "cav-mae" "eat" "m2d" "clmr" "clmr-dataspecific")
frac_errors=("0.05" "0.1" "0.2")
duplicate_types=("duplicates" "noisy_duplicates" "cropped_duplicates" "mixed_duplicates" "combined_duplicates")

for model in "${models[@]}"; do
    echo "Processing model: $model"

    for dup_type in "${duplicate_types[@]}"; do
      echo "  Processing duplicate type: $dup_type"
      for frac_error in "${frac_errors[@]}"; do
        echo "    Running Audio Hash baseline for ${dup_type} at frac_error=$frac_error..."
        python3 -m selfclean_audio \
          --config "config/templates/file_template.py" \
          --output-dir "outputs/${model}_esc50_${dup_type}_frac${frac_error}_audio_hash" \
          MODEL_TYPE="$model" \
          ISSUE_TYPE="$dup_type" \
          FRAC_ERROR="$frac_error" \
          near_duplicate_method="audio_hash" \
          near_duplicate_params="{}"

        echo "    Running Dejavu baseline for ${dup_type} at frac_error=$frac_error..."
        python3 -m selfclean_audio \
          --config "config/templates/file_template.py" \
          --output-dir "outputs/${model}_esc50_${dup_type}_frac${frac_error}_dejavu" \
          MODEL_TYPE="$model" \
          ISSUE_TYPE="$dup_type" \
          FRAC_ERROR="$frac_error" \
          near_duplicate_method="dejavu" \
          near_duplicate_params="{}"
      done
    done
done

echo "Near duplicate detection baseline evaluations completed!"
