#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT=${OUT_ROOT:-outputs/lora_grid}
mkdir -p "$OUT_ROOT"

MODELS=${MODELS:-"beats"}
ISSUES=${ISSUES:-"combined_duplicates off_topic_combined label_errors"}
FRACS=${FRACS:-"0.1"}

UNIVERSAL_STRATEGY=${UNIVERSAL_STRATEGY:-"objective=infonce r=16 alpha=48 lr=6e-5 ep=8 temp=0.03 aug=true proj=768"}
CANDIDATES=${CANDIDATES:-"$UNIVERSAL_STRATEGY | objective=vicreg r=6 alpha=18 lr=8e-5 ep=4 temp=0.0 aug=false proj=384 | objective=infonce r=12 alpha=24 lr=8e-5 ep=4 temp=0.08 aug=false proj=384"}

LORA_DROPOUT=${LORA_DROPOUT:-0.05}
MAX_STEPS=${MAX_STEPS:-8000}
ADAPT_WEIGHT_DECAY=${ADAPT_WEIGHT_DECAY:-5e-5}
BATCH_SIZE=${BATCH_SIZE:-16}
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-""}

echo "[Info] Outputs under: $OUT_ROOT"

run_job() {
  local model="$1"; shift
  local issue="$1"; shift
  local frac="$1"; shift
  local lora_enable="$1"; shift
  local objective="$1"; shift
  local r="$1"; shift
  local alpha="$1"; shift
  local lr="$1"; shift
  local epochs="$1"; shift
  local temp="$1"; shift
  local strong_aug="$1"; shift
  local proj_dim="$1"; shift

  local tag_base="${model}_${issue}_frac${frac}"
  local tag_extra
  if [[ "$lora_enable" == "true" ]]; then
    tag_extra="obj-${objective}__r-${r}__alpha-${alpha}__lr-${lr}__ep-${epochs}"
    if [[ "$objective" == "infonce" ]]; then
      tag_extra+="__temp-${temp}"
    fi
  else
    tag_extra="baseline"
  fi

  local outdir="$OUT_ROOT/${tag_base}__${tag_extra}__aug-${strong_aug}__proj-${proj_dim}"

  echo "[Run] $tag_base  ->  $tag_extra"

  # Build override array for python -m selfclean_audio
  # Note: all overrides must be separate args (no commas)
  declare -a OV
  OV+=("MODEL_TYPE=${model}")
  OV+=("ISSUE_TYPE=${issue}")
  OV+=("FRAC_ERROR=${frac}")
  OV+=("dataloader.batch_size=${BATCH_SIZE}")

  if [[ "$lora_enable" == "true" ]]; then
    OV+=("selfclean_audio.lora_enable=true")
    OV+=("selfclean_audio.lora_r=${r}")
    OV+=("selfclean_audio.lora_alpha=${alpha}")
    OV+=("selfclean_audio.lora_dropout=${LORA_DROPOUT}")
    OV+=("selfclean_audio.adapt_epochs=${epochs}")
    OV+=("selfclean_audio.adapt_lr=${lr}")
    OV+=("selfclean_audio.adapt_weight_decay=${ADAPT_WEIGHT_DECAY}")
    OV+=("selfclean_audio.adapt_projection_dim=${proj_dim}")
    OV+=("selfclean_audio.adapt_objective=${objective}")
    OV+=("selfclean_audio.adapt_strong_aug=${strong_aug}")
    if [[ -n "${MAX_STEPS}" ]]; then
      OV+=("selfclean_audio.adapt_max_steps=${MAX_STEPS}")
    fi
    if [[ "${objective}" == "infonce" ]]; then
      OV+=("selfclean_audio.adapt_temperature=${temp}")
    fi
    # Add extra overrides (split on spaces; ensure your EXTRA_OVERRIDES string has spaces between tokens)
    if [[ -n "${EXTRA_OVERRIDES}" ]]; then
      for tok in ${EXTRA_OVERRIDES}; do
        OV+=("${tok}")
      done
    fi
  else
    # Baseline: make sure adaptation is off
    OV+=("selfclean_audio.lora_enable=false")
    OV+=("selfclean_audio.adapt_epochs=0")
  fi

  python -m selfclean_audio \
    --config config/templates/file_template.py \
    --output-dir "$outdir" \
    "${OV[@]}"
}

echo "=== Running 3-candidate LoRA sweep (all issues) ==="

candidates_for_issue() {
  local issue="$1"
  case "$issue" in
    off_topic_*|offtopic*|off-topic*)
      # OT Strategy: Combat -4% underperformance with semantic specialization
      # High capacity + tight contrastive + semantic augmentation + longer training
      echo "objective=infonce r=16 alpha=48 lr=6e-5 ep=8 temp=0.03 aug=true proj=768 | objective=infonce r=20 alpha=40 lr=4e-5 ep=10 temp=0.05 aug=true proj=512 | objective=infonce r=12 alpha=36 lr=8e-5 ep=6 temp=0.04 aug=true proj=640"
      ;;
    *duplicates*)
      # ND Strategy: Build on +0.5% success - careful scaling without breaking what works
      # Extend training duration + modest capacity increase + fine-tune temperature
      echo "objective=infonce r=4 alpha=8 lr=2e-5 ep=12 temp=0.08 aug=false proj=256 | objective=infonce r=6 alpha=12 lr=3e-5 ep=10 temp=0.09 aug=false proj=320 | objective=infonce r=4 alpha=10 lr=2.5e-5 ep=15 temp=0.07 aug=false proj=288"
      ;;
    label_errors)
      # LE Strategy: Build on +0.5% success with label-discrimination focus
      # Balanced capacity + label-aware contrastive + discriminative training
      echo "objective=infonce r=8 alpha=20 lr=4e-5 ep=8 temp=0.12 aug=true proj=448 | objective=infonce r=10 alpha=25 lr=3e-5 ep=10 temp=0.10 aug=false proj=512 | objective=infonce r=6 alpha=18 lr=5e-5 ep=6 temp=0.14 aug=true proj=384"
      ;;
    *)
      # Fallback to universal strategy for unknown issue types
      echo "$UNIVERSAL_STRATEGY"
      ;;
  esac
}
for model in ${MODELS}; do
  for issue in ${ISSUES}; do
    for frac in ${FRACS}; do
      IFS='|' read -r -a cand_arr <<< "$CANDIDATES"
      # Override generic candidates with issue-specific presets when available
      CSET=$(candidates_for_issue "$issue")
      if [[ -n "$CSET" ]]; then IFS='|' read -r -a cand_arr <<< "$CSET"; fi
      for cand in "${cand_arr[@]}"; do
        # Normalize spaces
        cand=$(echo "$cand" | sed 's/^ *//;s/ *$//;s/  */ /g')
        # Extract fields with defaults
        objective=$(echo "$cand" | awk '{for(i=1;i<=NF;i++) if($i~"^objective=") {split($i,a,"="); print a[2]}}')
        r=$(echo "$cand" | awk '{for(i=1;i<=NF;i++) if($i~"^r=") {split($i,a,"="); print a[2]}}')
        alpha=$(echo "$cand" | awk '{for(i=1;i<=NF;i++) if($i~"^alpha=") {split($i,a,"="); print a[2]}}')
        lr=$(echo "$cand" | awk '{for(i=1;i<=NF;i++) if($i~"^lr=") {split($i,a,"="); print a[2]}}')
        ep=$(echo "$cand" | awk '{for(i=1;i<=NF;i++) if($i~"^ep=") {split($i,a,"="); print a[2]}}')
        temp=$(echo "$cand" | awk '{for(i=1;i<=NF;i++) if($i~"^temp=") {split($i,a,"="); print a[2]}}')
        strong_aug=$(echo "$cand" | awk '{for(i=1;i<=NF;i++) if($i~"^aug=") {split($i,a,"="); print a[2]}}')
        proj_dim=$(echo "$cand" | awk '{for(i=1;i<=NF;i++) if($i~"^proj=") {split($i,a,"="); print a[2]}}')
        # Fallbacks in case parsing fails
        objective=${objective:-infonce}
        r=${r:-8}
        alpha=${alpha:-16}
        lr=${lr:-1e-4}
        ep=${ep:-1}
        temp=${temp:-0.2}
        # Enforce weaker aug for duplicates unless user explicitly overrides via EXTRA_OVERRIDES
        if [[ "$issue" == *duplicates* && -z "$strong_aug" ]]; then strong_aug=false; fi
        strong_aug=${strong_aug:-true}
        proj_dim=${proj_dim:-256}
        run_job "$model" "$issue" "$frac" true "$objective" "$r" "$alpha" "$lr" "$ep" "$temp" "$strong_aug" "$proj_dim"
      done
    done
  done
done

echo "[Done] All runs finished."
