#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT=${OUT_ROOT:-outputs/universal_strategy}
mkdir -p "$OUT_ROOT"

MODELS=${MODELS:-"m2d"}
ISSUES=${ISSUES:-"combined_duplicates off_topic_combined label_errors"}
FRACS=${FRACS:-"0.20"}

# Universal strategy:
# r=16 alpha=48: high capacity adaptation
# temp=0.03: very tight contrastive learning
# aug=true: semantic robustness
# proj=768: large representational space
# ep=8 lr=6e-5: proven aggressive training schedule
UNIVERSAL_CONFIG="objective=infonce r=16 alpha=48 lr=6e-5 ep=8 temp=0.03 aug=true proj=768"

LORA_DROPOUT=${LORA_DROPOUT:-0.05}
MAX_STEPS=${MAX_STEPS:-8000}
ADAPT_WEIGHT_DECAY=${ADAPT_WEIGHT_DECAY:-5e-5}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
BATCH_SIZE=${BATCH_SIZE:-16}

echo "[Info] Testing universal strategy across all issues"
echo "[Info] Config: $UNIVERSAL_CONFIG"
echo "[Info] Outputs under: $OUT_ROOT"

run_universal_job() {
  local model="$1"; shift
  local issue="$1"; shift
  local frac="$1"; shift

  # Parse universal config
  objective=$(echo "$UNIVERSAL_CONFIG" | awk '{for(i=1;i<=NF;i++) if($i~"^objective=") {split($i,a,"="); print a[2]}}')
  r=$(echo "$UNIVERSAL_CONFIG" | awk '{for(i=1;i<=NF;i++) if($i~"^r=") {split($i,a,"="); print a[2]}}')
  alpha=$(echo "$UNIVERSAL_CONFIG" | awk '{for(i=1;i<=NF;i++) if($i~"^alpha=") {split($i,a,"="); print a[2]}}')
  lr=$(echo "$UNIVERSAL_CONFIG" | awk '{for(i=1;i<=NF;i++) if($i~"^lr=") {split($i,a,"="); print a[2]}}')
  ep=$(echo "$UNIVERSAL_CONFIG" | awk '{for(i=1;i<=NF;i++) if($i~"^ep=") {split($i,a,"="); print a[2]}}')
  temp=$(echo "$UNIVERSAL_CONFIG" | awk '{for(i=1;i<=NF;i++) if($i~"^temp=") {split($i,a,"="); print a[2]}}')
  strong_aug=$(echo "$UNIVERSAL_CONFIG" | awk '{for(i=1;i<=NF;i++) if($i~"^aug=") {split($i,a,"="); print a[2]}}')
  proj_dim=$(echo "$UNIVERSAL_CONFIG" | awk '{for(i=1;i<=NF;i++) if($i~"^proj=") {split($i,a,"="); print a[2]}}')

  local tag_base="${model}_${issue}_frac${frac}"
  local tag_extra="universal__r-${r}__alpha-${alpha}__lr-${lr}__ep-${ep}__temp-${temp}"
  local outdir="$OUT_ROOT/${tag_base}__${tag_extra}__aug-${strong_aug}__proj-${proj_dim}"

  echo "[Run] $issue -> Universal Strategy"

  declare -a OV
  OV+=("MODEL_TYPE=${model}")
  OV+=("ISSUE_TYPE=${issue}")
  OV+=("FRAC_ERROR=${frac}")
  OV+=("dataloader.batch_size=${BATCH_SIZE}")

  # LoRA configuration
  OV+=("selfclean_audio.lora_enable=true")
  OV+=("selfclean_audio.lora_r=${r}")
  OV+=("selfclean_audio.lora_alpha=${alpha}")
  OV+=("selfclean_audio.lora_dropout=${LORA_DROPOUT}")
  OV+=("selfclean_audio.adapt_epochs=${ep}")
  OV+=("selfclean_audio.adapt_lr=${lr}")
  OV+=("selfclean_audio.adapt_weight_decay=${ADAPT_WEIGHT_DECAY}")
  OV+=("selfclean_audio.adapt_projection_dim=${proj_dim}")
  OV+=("selfclean_audio.adapt_objective=${objective}")
  OV+=("selfclean_audio.adapt_strong_aug=${strong_aug}")
  OV+=("selfclean_audio.adapt_max_steps=${MAX_STEPS}")
  OV+=("selfclean_audio.adapt_temperature=${temp}")
  OV+=("selfclean_audio.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}")

  python -m selfclean_audio \
    --config config/templates/file_template.py \
    --output-dir "$outdir" \
    "${OV[@]}"
}

echo "=== Running Universal Strategy Test ==="

for model in ${MODELS}; do
  for issue in ${ISSUES}; do
    for frac in ${FRACS}; do
      run_universal_job "$model" "$issue" "$frac"
    done
  done
done

echo "[Done] Universal strategy test completed"
echo "Results in: $OUT_ROOT"
