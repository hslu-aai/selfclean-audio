from __future__ import annotations

import random as pyrandom

import numpy as np
import torch
from omegaconf import OmegaConf

from selfclean_audio.utils.sample import extract_sample
from selfclean_audio.utils.types import set_seeds


def _snap_state():
    return (
        pyrandom.random(),
        np.random.randint(0, 100000),
        int(torch.randint(0, 100000, (1,)).item()),
    )


def test_set_seeds_supports_multiple_config_layouts():
    # Case 1: _C.params.seed
    cfg1 = OmegaConf.create(
        {"params": {"seed": 7, "cudnn_deterministic": False, "cudnn_benchmark": False}}
    )
    set_seeds(cfg1)  # type: ignore[arg-type]
    s1 = _snap_state()

    # Change RNG
    set_seeds(OmegaConf.create({"params": {"seed": 8}}))  # type: ignore[arg-type]
    s2 = _snap_state()
    assert s1 != s2

    # Case 2: _C.SEED
    cfg2 = OmegaConf.create({"SEED": 7})
    set_seeds(cfg2)  # type: ignore[arg-type]
    s3 = _snap_state()
    assert s1 == s3  # same seed -> same sequence

    # Case 3: _C.selfclean_audio.random_seed
    cfg3 = OmegaConf.create({"selfclean_audio": {"random_seed": 9}})
    set_seeds(cfg3)  # type: ignore[arg-type]
    s4 = _snap_state()
    assert s4 != s1


def test_extract_sample_normalizes_tuples():
    import torch as T

    # 4-tuple path
    audio = T.zeros(1, 10)
    path = "/tmp/a.wav"
    label = T.tensor(3)
    noisy = T.tensor(1)
    a, p, label_value, n = extract_sample((audio, path, label, noisy))
    assert a is audio and p == path and label_value == 3 and int(n.item()) == 1

    # 3-tuple path -> adds noisy=0
    a2, p2, l2, n2 = extract_sample((audio, path, 5))
    assert a2 is audio and p2 == path and l2 == 5 and int(n2.item()) == 0
