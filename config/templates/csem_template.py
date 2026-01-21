from selfclean_audio.config import LazyCall as L
from selfclean_audio.selfclean_audio import PretrainingSSL, SelfCleanAudio

# Evaluation dataset selector
EVAL_DATASET = "CSEM"

# Paths
CSEM_ROOT = "data/CSEM/dataset_membranepumps"

# Model selection
MODEL_ENUMS = {"BEATS": PretrainingSSL.BEATS}
MODEL_PATHS = {
    # Update if you prefer BEATs_iter3.pt
    "BEATS": "data/Weights/BEATs_iter3_plus_AS2M.pt",
}
MODEL_TYPE = "BEATS"

# Dataloader parameters
dataloader = {
    "num_workers": 4,
    "batch_size": 8,
    "drop_last": False,
    "pin_memory": True,
}

# SelfCleanAudio setup. MODEL_TYPE and MODEL_PATHS are applied in __main__.
# Leave issues_to_detect=None to run all three issues.
selfclean_audio = L(SelfCleanAudio)(
    pretraining_ssl=PretrainingSSL.BEATS,
    model_path=MODEL_PATHS[MODEL_TYPE],
    device="cuda",
    memmap=True,
    memmap_path=None,
    # Default baseline methods
    off_topic_method="lad",
    off_topic_params=None,
    near_duplicate_method="embedding_distance",
    near_duplicate_params=None,
    label_error_method="intra_extra_distance",
    label_error_params=None,
    # Keep None to include all issues by default
    issues_to_detect=None,
)
