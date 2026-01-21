# Copyright (c) Lucerne University of Applied Sciences and Arts.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from selfclean_audio.datasets.base import BaseAudioDataset


class NoisyDataset(BaseAudioDataset):
    def __init__(
        self,
        root: str | Path,
        transforms: Callable | None = None,
        convert_mono: bool = True,
        sample_rate: int = 44100,
        label_error: bool = False,
        alpha: float = 0.1,
    ):
        """
        A PyTorch Dataset for loading audio files with optional label noise.

        Args:
            root (str | Path): Root directory containing audio files and `metadata.csv`.
            transforms (Callable, optional): Optional waveform transforms.
            convert_mono (bool): Convert stereo audio to mono.
            sample_rate (int): Target sample rate (audio will be resampled if needed).
            label_error (bool): Whether to inject random label noise.
            alpha (float): Fraction of samples to inject label noise into.
        """
        self.root = Path(root)
        metadata_path = self.root / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.csv not found in {self.root}")

        super().__init__(
            root=str(root), convert_mono=convert_mono, sample_rate=sample_rate
        )

        self.df = pd.read_csv(metadata_path)
        self.transforms = transforms
        self.label_error = label_error
        self.alpha = alpha

        self.class_to_idx = {
            cls: idx for idx, cls in enumerate(sorted(self.df["label"].unique()))
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        if self.label_error:
            self._generate_label_error_mask()

    @property
    def label_error_indices(self):
        """Return indices of samples with label errors for evaluation."""
        if hasattr(self, "mask"):
            return set(np.where(self.mask)[0])
        return set()

    def _generate_label_error_mask(self):
        """Generates a binary mask to inject label noise."""
        N = len(self.df)
        num_noisy = int(self.alpha * N)
        self.mask = np.zeros(N, dtype=bool)
        noisy_indices = np.random.choice(N, num_noisy, replace=False)
        self.mask[noisy_indices] = True

    def _get_noisy_label(self, original_label: int) -> int:
        """Returns a noisy label different from the original one."""
        all_labels = list(self.class_to_idx.values())
        all_labels.remove(original_label)
        return np.random.choice(all_labels)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, str, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        filepath = self.root / row["filename"]

        # Use base class method for consistent audio loading and preprocessing
        waveform, _ = self._load_and_preprocess_audio(filepath)

        if self.transforms:
            waveform = self.transforms(waveform)

        original_label = self.class_to_idx[row["label"]]

        # Determine if this sample is "noisy"
        if self.label_error:
            if self.mask[idx]:
                label = self._get_noisy_label(original_label)
                noisy_label = torch.tensor(1, dtype=torch.long)
            else:
                label = original_label
                noisy_label = torch.tensor(0, dtype=torch.long)
        else:
            noisy_label = torch.tensor(0, dtype=torch.long)
            if "noise_strategy" in row and row["noise_strategy"] != "Clean":
                noisy_label = torch.tensor(1, dtype=torch.long)
            label = original_label

        return waveform, str(filepath), torch.tensor(label), noisy_label

    def __len__(self) -> int:
        return len(self.df)
