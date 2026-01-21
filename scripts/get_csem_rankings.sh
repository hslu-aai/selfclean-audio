#!/usr/bin/env bash
set -euo pipefail

# Build rankings for the CSEM Membrane Pump dataset using different methods.
# - SelfClean with BEATS embeddings (all issue combinations)
# - SelfClean with Adapted BEATS embeddings (universal LoRA strategy)
# - BEATS baselines: IForest (OT), Dejavu (ND), CLearning (LB)

TEMPLATE=${TEMPLATE:-config/templates/csem_template.py}
ADAPTED_TEMPLATE=${ADAPTED_TEMPLATE:-config/templates/csem_adapted_template.py}
OUTDIR=${OUTDIR:-data/CSEM/membranepumps_rankings}

echo "[CSEM] Writing rankings to: ${OUTDIR}"

# SelfClean with BEATS: All issue combinations with default methods
echo "[CSEM] === Running SelfClean with BEATS embeddings ==="
echo "[CSEM] SelfClean BEATS - all issues with default methods"
python scripts/build_csem_rankings.py \
  --config "${TEMPLATE}" \
  --output-dir "${OUTDIR}"

# SelfClean with Adapted BEATS: Universal LoRA strategy
echo "[CSEM] SelfClean Adapted BEATS - universal LoRA strategy (r=16, alpha=48)"
python scripts/build_csem_rankings.py \
  --config "${ADAPTED_TEMPLATE}" \
  --output-dir "${OUTDIR}"

# BEATS Baselines: Specific methods for each issue type
echo "[CSEM] === Running BEATS baselines ==="

# Isolation Forest for Off-topic (OT)
echo "[CSEM] BEATS Off-topic: isolation_forest"
python scripts/build_csem_rankings.py \
  --config "${TEMPLATE}" \
  --issues off_topic_samples \
  --output-dir "${OUTDIR}" \
  -- \
  selfclean_audio.off_topic_method=isolation_forest

# Dejavu for Near-duplicates (ND)
echo "[CSEM] BEATS Near-duplicates: dejavu"
python scripts/build_csem_rankings.py \
  --config "${TEMPLATE}" \
  --issues near_duplicates \
  --output-dir "${OUTDIR}" \
  -- \
  selfclean_audio.near_duplicate_method=dejavu

# CLearning (cleanlab) for Label errors (LB)
echo "[CSEM] BEATS Label errors: cleanlab (confident learning)"
python scripts/build_csem_rankings.py \
  --config "${TEMPLATE}" \
  --issues label_errors \
  --output-dir "${OUTDIR}" \
  -- \
  selfclean_audio.label_error_method=cleanlab

echo "[CSEM] All rankings complete."
