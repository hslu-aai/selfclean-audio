# Copyright (c) Lucerne University of Applied Sciences and Arts.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

from typing import Optional, Tuple

import torch


def extract_sample(
    sample_tuple: tuple,
) -> Tuple[torch.Tensor, Optional[str], int, torch.Tensor]:
    """
    Normalize dataset samples to a common shape.

    Accept tuples returned by various dataset implementations and return a
    tuple of (waveform, path, label, noisy_label) where:
    - waveform: torch.Tensor, shape (C, T) or (T,)
    - path: Optional[str] (file path or None if unavailable)
    - label: int (class index)
    - noisy_label: torch.Tensor scalar long, 0 for clean, 1 for noisy

    This helper avoids repeated ad-hoc tuple length checks across the codebase.
    """
    n = len(sample_tuple)
    if n == 4:
        # (audio, path, label, noisy_label)
        audio, path, label, noisy = sample_tuple
        if not isinstance(noisy, torch.Tensor):
            noisy = torch.tensor(int(noisy), dtype=torch.long)
        return (
            audio,
            path,
            int(label) if not isinstance(label, torch.Tensor) else int(label.item()),
            noisy,
        )

    if n == 3:
        # (audio, path, label) -> add noisy_label=0
        audio, path, label = sample_tuple
        return (
            audio,
            path,
            int(label) if not isinstance(label, torch.Tensor) else int(label.item()),
            torch.tensor(0, dtype=torch.long),
        )

    if n == 2:
        # (audio, label) -> no path information
        audio, label = sample_tuple
        return (
            audio,
            None,
            int(label) if not isinstance(label, torch.Tensor) else int(label.item()),
            torch.tensor(0, dtype=torch.long),
        )

    raise ValueError(
        f"Unsupported sample tuple format of length {n}; expected 2, 3 or 4 elements"
    )
