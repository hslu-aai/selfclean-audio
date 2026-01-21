# Copyright (c) Lucerne University of Applied Sciences and Arts.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import torch

from selfclean_audio.datasets.base import BaseAudioDataset


class FolderAudioDataset(BaseAudioDataset):
    def __init__(
        self,
        root: Path | str,
        transforms: Callable | None = None,
        convert_mono: bool = True,
        sample_rate: int = 44100,
        target_duration_sec: int = 5,
        extensions=(".wav", ".mp3", ".flac"),
    ):
        """
        Dataset for loading audio files organized in class-labeled folders.

        Args:
            root (Path or str): Root directory with subfolders for each class.
            transforms (Callable, optional): Function to apply to each waveform.
            convert_mono (bool): Convert stereo to mono if True (default: True).
            sample_rate (int): Target sample rate for audio (default: 44100).
            target_duration_sec (int): Target duration of audio in seconds (default: 5).
            extensions (tuple): Valid audio file extensions (default: ('.wav', '.mp3', '.flac')).
        """
        super().__init__(
            root=str(root),
            convert_mono=convert_mono,
            sample_rate=sample_rate,
            target_duration_sec=target_duration_sec,
        )
        self.transforms = transforms
        self.extensions = extensions
        self.samples = self._load_samples()
        self.class_to_idx = {
            cls: idx for idx, cls in enumerate(sorted(self.samples.keys()))
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def _load_samples(self):
        """
        Recursively scan root directory for audio files grouped by class labels.

        Returns:
            dict: Mapping of class label to list of audio file paths.
        """
        samples = defaultdict(list)
        for root, _, files in os.walk(self.root):
            for file in files:
                if file.lower().endswith(self.extensions):
                    label = os.path.basename(root)
                    path = os.path.join(root, file)
                    samples[label].append(path)
        if not samples:
            raise RuntimeError(f"No valid audio files found in {self.root}.")
        return samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, int]:
        """
        Retrieve and preprocess audio sample by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (waveform tensor, audio file path, class index)
        """
        # Find the class for this index
        class_idx = idx % len(
            self.class_to_idx
        )  # Get the class index based on the total number of classes
        class_name = self.idx_to_class[class_idx]

        # Get the list of sample paths for this class
        sample_paths = self.samples[class_name]

        # Get the actual sample index within the class
        sample_idx = idx // len(
            self.class_to_idx
        )  # Find the sample index within the class

        # Ensure the sample index is valid
        if sample_idx >= len(sample_paths):
            sample_idx = len(sample_paths) - 1

        path = sample_paths[sample_idx]

        waveform, _ = self._load_and_preprocess_audio(path)

        # Apply transforms if provided
        if self.transforms:
            waveform = self.transforms(waveform)

        return waveform, path, self.class_to_idx[class_name]

    def __len__(self) -> int:
        """
        Returns:
            int: Total number of audio samples in the dataset.
        """
        return sum(len(paths) for paths in self.samples.values())
