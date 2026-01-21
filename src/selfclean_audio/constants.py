# Copyright (c) Lucerne University of Applied Sciences and Arts.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""Constants used throughout the selfclean_audio package."""

# Logging and Output
DEFAULT_OUTPUT_DIR = "./outputs"
LOG_FORMAT = "{time} {level} {message}"
LOG_SEPARATOR = "=" * 40

# Visualization Constants
DEFAULT_FIGURE_SIZE = (10, 8)
DEFAULT_HISTOGRAM_BINS = 30
DEFAULT_HISTOGRAM_ALPHA = 0.6
DEFAULT_TEMPORAL_STATS_BINS = 30

# t-SNE Parameters
DEFAULT_TSNE_PERPLEXITY_MIN = 2
DEFAULT_TSNE_PERPLEXITY_MAX = 30

# Model and Training Defaults
DEFAULT_RANDOM_SEED = 42
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_TARGET_DURATION_SEC = 30.0

# Data Processing
DEFAULT_CHUNK_SIZE = 100
DEFAULT_SNR_DB = 15.0
DEFAULT_TEMPO_MIN = 0.9
DEFAULT_TEMPO_MAX = 1.1
DEFAULT_PITCH_SEMITONES = 2.0
DEFAULT_TIME_SHIFT_MAX = 0.1

# Validation Sets - Required Configuration Parameters
REQUIRED_BASE_PARAMS = ["ISSUE_TYPE", "FRAC_ERROR"]
REQUIRED_DATASET_PATHS = ["ESC50_ROOT", "ESC50_META", "NOISE_ROOT"]
REQUIRED_GTZAN_PARAMS = ["GTZAN_ROOT", "ISSUE_TYPE"]
REQUIRED_DATALOADER_PARAMS = ["num_workers", "batch_size", "drop_last", "pin_memory"]

# Baseline Parameter Names
BASELINE_PARAM_NAMES = [
    "near_duplicate_method",
    "near_duplicate_params",
    "off_topic_method",
    "off_topic_params",
    "label_error_method",
    "label_error_params",
]

# Issue Type Mappings
DUPLICATE_STRATEGY_MAP = {
    "duplicates": "exact",
    "noisy_duplicates": "noisy",
    "cropped_duplicates": "cropped",
    "mixed_duplicates": "mixed",
    "combined_duplicates": "combined",
}

OFF_TOPIC_STRATEGY_MAP = {
    "off_topic_noise": "noise",
    "off_topic_external": "external",
    "off_topic_corrupted": "corrupted",
    "off_topic_combined": "combined",
}

# Feature Names for Temporal Statistics
TEMPORAL_FEATURE_NAMES = [
    "mean_norm",
    "std_norm",
    "mean_delta",
    "std_delta",
    "mean_feat_mean",
    "mean_feat_std",
    "var_feat_mean",
    "var_feat_std",
]
