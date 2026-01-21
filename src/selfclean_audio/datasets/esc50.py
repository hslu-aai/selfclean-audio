# Copyright (c) Lucerne University of Applied Sciences and Arts.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os

import pandas as pd
import torch

from selfclean_audio.datasets.base import BaseAudioDataset


class ESC50(BaseAudioDataset):
    def __init__(
        self,
        root=None,
        dataframe=None,
        convert_mono: bool = True,
        resample: int = 44100,
    ):
        """
        AudioSet20KDataset paired with JSON label files.

        Args:
            root (str): Directory containing audio and JSON files.
            dataframe (str): Path to the dataframe with the information of the files.
            convert_mono (bool): Convert audio to mono if True.
            resample (int): Target sample rate for audio.
            target_duration_sec (int): Duration to pad or trim audio (seconds).
        """
        super().__init__(root=root, convert_mono=convert_mono, sample_rate=resample)
        self.df = pd.read_csv(dataframe)

    def __len__(self) -> int:
        """
        Returns:
            int: Number of audio files in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, int]:
        """
        Load and preprocess an audio sample and its labels.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (waveform tensor, audio file path, label tensor)
        """
        audio_path = os.path.join(self.root, self.df["filename"][idx])
        label = self.df["target"][idx]

        # Use base class method for consistent audio loading and preprocessing
        waveform, _ = self._load_and_preprocess_audio(audio_path)

        # Convert labels to torch tensor (long dtype)
        labels = torch.tensor(label, dtype=torch.long)

        return waveform, audio_path, labels
