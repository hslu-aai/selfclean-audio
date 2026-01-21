# Copyright (c) Lucerne University of Applied Sciences and Arts.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""Base audio dataset class providing common functionality for audio loading and preprocessing."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset


class BaseAudioDataset(Dataset, ABC):
    """
    Base class for audio datasets with common preprocessing functionality.

    Provides standardized audio loading, mono conversion, resampling, and duration handling.
    """

    def __init__(
        self,
        root: str | None = None,
        convert_mono: bool = True,
        sample_rate: int = 44100,
        target_duration_sec: float | None = None,
    ):
        """
        Initialize base audio dataset.

        Args:
            root: Root directory path for the dataset
            convert_mono: Convert stereo audio to mono if True
            sample_rate: Target sample rate for audio (will resample if needed)
            target_duration_sec: Target duration in seconds (will pad/trim if specified)
        """
        self.root = Path(root) if root else Path("./")
        self.convert_mono = convert_mono
        self.sample_rate = sample_rate
        self.target_duration_sec = target_duration_sec
        self.target_length = (
            int(target_duration_sec * sample_rate) if target_duration_sec else None
        )

    def _load_and_preprocess_audio(
        self, audio_path: str | Path
    ) -> tuple[torch.Tensor, int]:
        """
        Load and preprocess audio file with standardized transformations.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (waveform, sample_rate) where waveform is preprocessed
        """
        audio_path = Path(audio_path)

        # Load audio file
        waveform, file_sample_rate = torchaudio.load(str(audio_path))

        # Convert to mono if requested
        if self.convert_mono and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if file_sample_rate != self.sample_rate:
            waveform = T.Resample(
                orig_freq=file_sample_rate, new_freq=self.sample_rate
            )(waveform)

        # Handle target duration (pad or trim)
        if self.target_length is not None:
            waveform = self._resize_waveform(waveform, self.target_length)

        return waveform, self.sample_rate

    def _resize_waveform(
        self, waveform: torch.Tensor, target_length: int
    ) -> torch.Tensor:
        """
        Resize waveform to target length by padding or trimming.

        Args:
            waveform: Input waveform tensor of shape (C, T)
            target_length: Target length in samples

        Returns:
            Resized waveform tensor
        """
        current_length = waveform.size(1)

        if current_length < target_length:
            # Pad with zeros
            padding = target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_length > target_length:
            # Trim to target length
            waveform = waveform[:, :target_length]

        return waveform

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple:
        """Get a sample from the dataset."""
        pass
