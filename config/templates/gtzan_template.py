import numpy as np

from selfclean_audio.config import LazyCall as L
from selfclean_audio.selfclean_audio import PretrainingSSL, SelfCleanAudio

# Dataset and experiment selection
EVAL_DATASET = "gtzan"  # Special path in factory uses known GTZAN issues
ISSUE_TYPE = "duplicates"  # Options: "duplicates" or "label_errors"
MODEL_TYPE = "beats"
SEED = 42

# Baseline method parameters - can be overridden via CLI
near_duplicate_method = (
    "embedding_distance"  # "embedding_distance", "cleanlab", "dejavu", "audio_hash"
)
near_duplicate_params = {}
label_error_method = "intra_extra_distance"  # "intra_extra_distance", "cleanlab"
label_error_params = {}
off_topic_method = "lad"  # Not used for GTZAN known-issues eval
off_topic_params = {}

# GTZAN paths (local folder with genres/<class>/*.wav)
GTZAN_ROOT = "./data/gtzan/genres/"

# Known-issues CSVs from external_code (duplicates and mislabels)
GTZAN_GT_FILE = "./data/gtzan/gtzan_ground_truth_full_exact_rep.txt"
GTZAN_PREP_FILE = "./data/gtzan/gtzan_ground_truth_prep.txt"

# Model configuration (same mapping as file_template)
MODEL_PATHS = {
    "beats": "./data/Weights/BEATs_iter3_plus_AS2M.pt",
    "cav-mae": "./data/Weights/cav-mae.pth",
    "eat": "./data/Weights/EAT-base_epoch30_ft.pt",
    "m2d": "./data/Weights/m2d_clap_vit_base-80x608p16x16-240128/checkpoint300.pth",
    "clmr-dataspecific": "./data/Weights/CMLR_ESC_50/epoch=1300-step=53341.ckpt",
    "clmr": "./data/Weights/clmr_checkpoint_10000.pt",
}

MODEL_ENUMS = {
    "beats": PretrainingSSL.BEATS,
    "cav-mae": PretrainingSSL.CAVMAE,
    "eat": PretrainingSSL.EAT_BASE_PRETRAIN,
    "m2d": PretrainingSSL.M2D,
    "clmr-dataspecific": PretrainingSSL.CLMR,
    "clmr": PretrainingSSL.CLMR,
}

dataloader = dict(num_workers=8, batch_size=16, drop_last=False, pin_memory=True)

selfclean_audio = L(SelfCleanAudio)(
    # distance calculation
    distance_function_path="sklearn.metrics.pairwise.",
    distance_function_name="cosine_similarity",
    chunk_size=100,
    precision_type_distance=np.float32,
    # memory management
    memmap=False,
    memmap_path=None,
    # plotting
    plot_distribution=False,
    plot_top_N=None,
    output_path=None,
    figsize=(10, 8),
    # model
    pretraining_ssl=MODEL_ENUMS[MODEL_TYPE],
    model_path=MODEL_PATHS[MODEL_TYPE],
    # baseline methods (overridable)
    near_duplicate_method=near_duplicate_method,
    near_duplicate_params=near_duplicate_params,
    label_error_method=label_error_method,
    label_error_params=label_error_params,
    off_topic_method=off_topic_method,
    off_topic_params=off_topic_params,
    # utils
    random_seed=SEED,
    device="cuda",
)

params = dict(
    seed=SEED,
    cudnn_benchmark=True,
    cudnn_deterministic=False,
)
