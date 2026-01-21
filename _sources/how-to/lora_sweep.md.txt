# Run LoRA Adaptation Grid

This guide runs a LoRA fine-tuning sweep (unsupervised SimCLR/VICReg) to adapt
the backbone's representations to the target dataset, then evaluates SelfClean
metrics.

## Prerequisites

- Datasets configured in `config/templates/file_template.py` (`ESC50_ROOT`,
  `ESC50_META`, `NOISE_ROOT`).
- Backbone weights available at paths in `MODEL_PATHS`.

## Quick Start (Stage A)

Baselines and a coarse LoRA grid over BEATs and EAT across duplicates, off-topic,
and label errors at 10% corruption:

```
scripts/finetuning/run_lora_grid.sh
```

## What It Does

- Writes runs to `outputs/lora_grid/<model>_<issue>_frac<...>__...`.
- Saves `config.yaml` and per-issue `Score-*.csv` compatible with
  `scripts/collect_results.py`.

## Customize the Sweep (Environment Variables)

- `MODELS`: space-separated backbones to try (default: `"beats eat"`).
- `ISSUES`: detection tasks (default: `"duplicates off_topic_noise label_errors"`).
- `FRACS`: corruption fractions (default: `"0.1"`).
- `OBJECTIVES`: `infonce vicreg`.
- `RS`: LoRA rank (default: `"8 16"`).
- `ALPHAS`: LoRA alpha; if empty uses `{r, 2r}`.
- `LRS`: learning rates (Stage A default: `"5e-5 1e-4 3e-4"`).
- `EPOCHS`: adaptation epochs (Stage A: `"1 3"`).
- `TEMPS`: InfoNCE temperatures (Stage A: `"0.07 0.2 0.5"`).
- `MAX_STEPS`: cap on optimizer steps per run (default: `200`).
- `EXTRA_OVERRIDES`: additional dotlist overrides (default enables strong augs).

## Examples

```
# Smaller dev sweep
MODELS="beats" ISSUES="duplicates off_topic_noise" EPOCHS="1" MAX_STEPS=100 \
  scripts/finetuning/run_lora_grid.sh

# VicReg only, higher rank
OBJECTIVES="vicreg" RS="16" LRS="3e-4" EPOCHS="3" \
  scripts/finetuning/run_lora_grid.sh

# Add/modify augmentations
EXTRA_OVERRIDES="selfclean_audio.adapt_strong_aug=true selfclean_audio.adapt_eq_prob=0.5" \
  scripts/finetuning/run_lora_grid.sh
```

## Aggregate Results

```
python scripts/collect_results.py --base-dir outputs
```

## Notes

- The saved config logs requested overrides and actual initialization details.
- If PEFT is unavailable, the run falls back to frozen base with projection head only.
- Use `MAX_STEPS` to keep wallclock manageable while comparing many configurations.
