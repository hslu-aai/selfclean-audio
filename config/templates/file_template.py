import numpy as np

from selfclean_audio.config import LazyCall as L
from selfclean_audio.selfclean_audio import PretrainingSSL, SelfCleanAudio

# Default parameters - can be overridden via command line
ISSUE_TYPE = "duplicates"
MODEL_TYPE = "beats"
FRAC_ERROR = 0.1
SEED = 42

# Baseline method parameters - can be overridden for baseline evaluations
near_duplicate_method = "embedding_distance"  # Options: "embedding_distance", "cleanlab", "dejavu", "audio_hash"
near_duplicate_params = {}
off_topic_method = "lad"  # Options: "lad", "quantile", "isolation_forest", "cleanlab"
off_topic_params = {}
label_error_method = (
    "intra_extra_distance"  # Options: "intra_extra_distance", "cleanlab"
)
label_error_params = {}

# Dataset paths - used by LazyFactory
ESC50_ROOT = "./data/ESC-50-master/audio/"
ESC50_META = "./data/ESC-50-master/meta/esc50.csv"
NOISE_ROOT = "./data/wmms/"

# Model configuration
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
    # baseline methods
    near_duplicate_method=near_duplicate_method,
    near_duplicate_params=near_duplicate_params,
    off_topic_method=off_topic_method,
    off_topic_params=off_topic_params,
    label_error_method=label_error_method,
    label_error_params=label_error_params,
    # utils
    random_seed=SEED,
    device="cuda",
)

params = dict(
    seed=SEED,
    cudnn_benchmark=True,
    cudnn_deterministic=False,
)
