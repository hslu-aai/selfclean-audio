#!/bin/bash

echo "Running off-topic detection baseline evaluations..."

models=("beats" "cav-mae" "eat" "m2d" "clmr" "clmr-dataspecific")
frac_errors=("0.05" "0.1" "0.2")
off_topic_types=("off_topic_noise" "off_topic_external" "off_topic_corrupted" "off_topic_combined")

for model in "${models[@]}"; do
    echo "Processing model: $model"

    for off_topic_type in "${off_topic_types[@]}"; do
        for frac_error in "${frac_errors[@]}"; do
            # Off-topic detection baselines
            echo "  Running IsolationForest baseline for ${off_topic_type} at frac_error=$frac_error..."
            python3 -m selfclean_audio \
              --config "config/templates/file_template.py" \
              --output-dir "outputs/${model}_esc50_${off_topic_type}_frac${frac_error}_isolation_forest" \
              MODEL_TYPE="$model" \
              ISSUE_TYPE="$off_topic_type" \
              FRAC_ERROR="$frac_error" \
              off_topic_method="isolation_forest" \
              off_topic_params="{}"

            echo "  Running CleanLab off-topic baseline for ${off_topic_type} at frac_error=$frac_error..."
            python3 -m selfclean_audio \
              --config "config/templates/file_template.py" \
              --output-dir "outputs/${model}_esc50_${off_topic_type}_frac${frac_error}_cleanlab_off_topic" \
              MODEL_TYPE="$model" \
              ISSUE_TYPE="$off_topic_type" \
              FRAC_ERROR="$frac_error" \
              off_topic_method="cleanlab" \
              off_topic_params="{}"
        done
    done
done

echo "Off-topic detection baseline evaluations completed!"
