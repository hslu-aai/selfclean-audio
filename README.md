# SelfClean for Audio

SelfClean-Audio detects dataset issues in audio (near-duplicates, off-topic samples, and label errors).
It includes strong baselines, an end‑to‑end runner, and automatic aggregation of evaluation metrics.

- Source: https://github.com/hslu-aai/selfclean-audio
- Docs: https://hslu-aai.github.io/selfclean-audio
- License: Apache 2.0

<!-- README only content -->

## Quick Start (Docker Compose)

No local Python/conda setup required.
Use Docker Compose to bring up the app and a Postgres database for the Dejavu near‑duplicate baseline.

Prerequisites
- Docker and Docker Compose

Workflow
1) Build + start the stack (app + Postgres)
- `make up`

2) (Optional) Verify the environment inside the container
- `make check`

3) Open a shell inside the app container
- `make bash`

4) Run experiments (inside the container)
- `scripts/get_results.sh`  (file experiments + baselines + aggregation)
- Or individual sets:
  - `scripts/get_file_results.sh`  (Duplicates / Off‑topic / Label errors)
  - `scripts/baselines/get_baseline_results.sh`  (All baselines)

5) Inspect aggregated results (written to `outputs/aggregates/`)
- `all_results.csv` — one row per run per issue with metrics + config context (via `scripts/collect_results.py`)

6) Stop the stack when done
- `make down`

## LoRA Adaptation Grid (Fine‑Tuning)

Run an unsupervised LoRA adaptation sweep (InfoNCE/VICReg) to tailor embeddings to the target dataset, then evaluate SelfClean metrics:

```
scripts/finetuning/run_lora_grid.sh
```

Customize with env vars (examples, Stage A defaults in parentheses):
- `MODELS="beats eat"` — backbones to try
- `ISSUES="duplicates off_topic_noise label_errors"` — detection tasks
- `EPOCHS="1 3"` `RS="8 16"` `LRS="5e-5 1e-4 3e-4"` `TEMPS="0.07 0.2 0.5"`
- `MAX_STEPS=200` to cap optimizer steps per run

Aggregate all results after runs:

```
python scripts/collect_results.py --base-dir outputs
```

## Configuration and Model Weights

Default experiment template: `config/templates/file_template.py`.

Override key parameters via dotlist CLI args:
- `MODEL_TYPE` — embedding backbone (`beats`, `cav-mae`, `eat`, `m2d`).
- `ISSUE_TYPE` — dataset corruption type:
  - `duplicates`, `noisy_duplicates`, `cropped_duplicates`, `mixed_duplicates`, `combined_duplicates`
  - `label_errors`
  - `off_topic_noise`, `off_topic_external`, `off_topic_corrupted`, `off_topic_combined`
- `FRAC_ERROR` — fraction of samples to corrupt (e.g., `0.05`, `0.1`, `0.2`).
- Baseline methods and their params:
  - `near_duplicate_method` (`embedding_distance`, `audio_hash`, `dejavu`)
  - `off_topic_method` (`lad`, `quantile`, `isolation_forest`, `cleanlab`)
  - `label_error_method` (`intra_extra_distance`, `cleanlab` [confident learning])
  - `*_params` — dicts for method‑specific knobs

Model weights: edit `MODEL_PATHS` in `config/templates/file_template.py` to point to your local checkpoints. These are selected automatically based on `MODEL_TYPE`.

Example
```
python -m selfclean_audio \
  --config config/templates/file_template.py \
  --output-dir outputs/beats_esc50_duplicates_frac0.1 \
  MODEL_TYPE=beats ISSUE_TYPE=duplicates FRAC_ERROR=0.1 \
  near_duplicate_method=audio_hash 'near_duplicate_params={hash_method:mfcc,hash_size:32}'
```

## Synthetic Noise Types (Ground Truth Generators)

We provide controlled synthetic corruptions to benchmark detection quality. Each generator returns ground truth needed for evaluation.

- Duplicates (`DuplicateDataset`)
  - `duplicates` — exact duplicates (identical copy appended)
  - `noisy_duplicates` — original + copy with additive noise
  - `cropped_duplicates` — original + copy with a randomly zeroed/cropped segment
  - `mixed_duplicates` — original + copy softly mixed with noise
  - `combined_duplicates` — randomly selects one of the above strategies for each duplicate
  - Implementation detail: Synthetic duplicates are written as real `.wav` files into a temp folder so audio‑hash / Dejavu can operate on the actual audio content. Files are automatically cleaned up after the run.
  - Ground truth: set of (original_idx, duplicate_idx) pairs via `get_errors()`.

- Label Errors (`LabelErrorDataset`)
  - Randomly reassigns labels for a fraction of samples (optionally per class).
  - Ground truth: a 0/1 list indicating label errors via `get_errors()`.

- Off‑Topic (`OffTopicDataset`)
  - `off_topic_noise` — replaces selected samples with pure noise (Gaussian/uniform)
  - `off_topic_external` — injects samples from a separate dataset, resized to match
  - `off_topic_corrupted` — heavily corrupts selected originals (e.g., strong noise)
  - `off_topic_combined` — randomly selects one of the above strategies for each off-topic sample
  - Ground truth: a 0/1 list indicating off‑topic samples via `get_errors()`.

## Baselines and Evaluation

All baselines return rankings (no thresholds). Metrics (AUROC, AP, Recall@K, Precision@K) are computed from rankings and saved as `Score-<issue>.csv` next to each run.

- Near‑duplicates: `embedding_distance`, `audio_hash`, `dejavu` (requires Postgres).
- Off‑topic: `lad`, `quantile`, `isolation_forest`, `cleanlab` (supervised confidence: score = `1 − max_c P(c|x)`).
- Label errors: `intra_extra_distance`, `cleanlab` (confident learning: score = `1 − P(observed label|x)`).

The CLI logs both requested (from config) and actual initialized parameters (model, methods, params, dataset) so you can confirm that lazy loading behaved as intended.

## Development

This project uses **Docker-first development** for complete dependency isolation and **tox** for consistent test/build environments.

### Recommended Development Workflow (Docker)

Due to complex system dependencies (PostgreSQL, PyTorch, audio libraries), we recommend using Docker for development:

```bash
# Clone and initialize
git clone https://github.com/hslu-aai/selfclean-audio
cd selfclean-audio

# Start the development environment
make up

# Run development tasks inside the container
make test      # run tests with coverage inside container
make lint      # run linting and formatting inside container
make typecheck # run type checking inside container
make docs      # build documentation inside container
make build     # build and validate distribution inside container
make check     # run lint + typecheck + test inside container
make all       # run all tox environments inside container

# Stop when done
make down
```

All development commands automatically run inside the Docker container.
If the container isn't running, commands will prompt you to start it with `make up`.

### Available Make Targets

**Development (Docker-based):**
- `make test` — run tests with coverage inside container
- `make lint` — run linting and formatting inside container
- `make typecheck` — run type checking inside container
- `make docs` — build documentation inside container
- `make build` — build and validate packages inside container
- `make check` — run lint + typecheck + test inside container
- `make all` — run all tox environments inside container
- `make clean` — clean build artifacts and cache (local)

**Docker Management:**
- `make up` — start development environment (app + database)
- `make down` — stop development environment
- `make bash` — open shell inside container
- `make logs` — view container logs
- `make check` — verify environment setup inside container
