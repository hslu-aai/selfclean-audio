# Copyright (c) Lucerne University of Applied Sciences and Arts.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import random

import numpy as np
import torch
from omegaconf import DictConfig


def set_seeds(_C: DictConfig):
    """
    Set the random seed for reproducibility across various libraries and frameworks.

    This function sets the seed for the Python ``random`` module, the ``numpy`` library, and
    the ``torch`` library to ensure deterministic behavior. Additionally, it configures
    CuDNN's deterministic and benchmarking behavior based on the provided configuration.

    Args:
        _C (DictConfig): A dict configuration that contains the random seed and
            CuDNN settings (``seed``, ``cudnn_deterministic``, and ``cudnn_benchmark``).

    Notes:
        - ``_C.params.seed`` should be an integer value used for seeding.
        - ``_C.params.cudnn_deterministic`` is a boolean flag that enforces deterministic behavior in CuDNN.
        - ``_C.params.cudnn_benchmark`` is a boolean flag that enables CuDNN's autotuner for optimal performance.
    """
    # Try multiple locations for the seed to support different templates
    seed = None
    try:
        if hasattr(_C, "params") and hasattr(_C.params, "seed"):
            seed = int(_C.params.seed)
        elif hasattr(_C, "SEED"):
            seed = int(_C.SEED)
        elif hasattr(_C, "selfclean_audio") and hasattr(
            _C.selfclean_audio, "random_seed"
        ):
            seed = int(_C.selfclean_audio.random_seed)
    except Exception:
        pass

    if seed is None:
        # Fallback to a reasonable default but warn via comment (no logging here to avoid side-effects)
        seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CuDNN flags: default to deterministic=False, benchmark=False if missing
    det = False
    bench = False
    try:
        if hasattr(_C, "params"):
            det = bool(getattr(_C.params, "cudnn_deterministic", det))
            bench = bool(getattr(_C.params, "cudnn_benchmark", bench))
    except Exception:
        pass
    torch.backends.cudnn.deterministic = det
    torch.backends.cudnn.benchmark = bench


def ensure_dictconfig(_C):
    """
    Ensure the provided configuration is an instance of DictConfig.

    This function checks if the given object ``_C`` is an instance of the ``DictConfig`` class.
    If not, a `TypeError` is raised.

    Args:
        _C (object): The configuration object to check.

    Returns:
        DictConfig: The original `_C` object if it is an instance of `DictConfig`.

    Raises:
        TypeError: If `_C` is not an instance of `DictConfig`.

    """
    if not isinstance(_C, DictConfig):
        raise TypeError(f"Expected DictConfig, got {type(_C)}")
    return _C
