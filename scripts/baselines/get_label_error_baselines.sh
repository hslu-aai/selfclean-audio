#!/bin/bash

echo "Running label error detection baseline evaluations..."

# Define model names and frac_errors
models=("beats" "cav-mae" "eat" "m2d" "clmr" "clmr-dataspecific")
frac_errors=("0.05" "0.1" "0.2")

# Loop over models
for model in "${models[@]}"; do
    echo "Processing model: $model"

    # Label error detection baselines
    for frac_error in "${frac_errors[@]}"; do
        echo "  Running CleanLab label error baseline for frac_error=$frac_error..."
        python3 -m selfclean_audio \
          --config "config/templates/file_template.py" \
          --output-dir "outputs/${model}_esc50_label_errors_frac${frac_error}_cleanlab" \
          MODEL_TYPE="$model" \
          ISSUE_TYPE="label_errors" \
          FRAC_ERROR="$frac_error" \
          label_error_method="cleanlab" \
          label_error_params="{}"
    done
done

echo "Label error detection baseline evaluations completed!"
