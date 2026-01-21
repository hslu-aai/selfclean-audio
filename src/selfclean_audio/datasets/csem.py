# Copyright (c) Lucerne University of Applied Sciences and Arts.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path

import pandas as pd
import torch

from selfclean_audio.datasets.base import BaseAudioDataset


class CSEMMembranePumps(BaseAudioDataset):
    """
    CSEM Membrane Pump Audio Dataset loader.

    Expects the following structure under ``root`` (see ``data/CSEM/README.md``)::

        root/
            files/
                {guid}.wav
                ...
            index.csv   # columns: id, filename, label

    The dataset returns tuples of (waveform, absolute_path, label).
    ``noisy_label`` is not known and will be set by the synthetic/noise wrappers
    when applicable, otherwise considered 0 by downstream code.
    """

    def __init__(
        self,
        root: str | Path,
        convert_mono: bool = True,
        sample_rate: int = 16000,
        target_duration_sec: float | None = None,
        index_file: str | Path = "index.csv",
        files_dir: str | Path = "files",
    ):
        super().__init__(
            root=str(root),
            convert_mono=convert_mono,
            sample_rate=sample_rate,
            target_duration_sec=target_duration_sec,
        )
        self.root = Path(root)
        self.index_path = self.root / index_file
        self.files_dir = self.root / files_dir

        if not self.index_path.exists():
            raise FileNotFoundError(f"CSEM index file not found: {self.index_path}")
        if not self.files_dir.exists():
            raise FileNotFoundError(f"CSEM files directory not found: {self.files_dir}")

        self.df = pd.read_csv(self.index_path)
        # Basic validation of expected columns
        expected_cols = {"id", "filename", "label"}
        missing = expected_cols - set(self.df.columns)
        if missing:
            raise ValueError(
                f"CSEM index.csv is missing columns: {sorted(missing)}; "
                f"found columns: {list(self.df.columns)}"
            )

        # Create a mapping of label ids to ensure consistent behavior downstream
        # Labels are already integers per README; keep as-is
        self.labels = self.df["label"].astype(int).tolist()

        # Map for optional meta output in IssueManager
        classes = sorted(set(self.labels))
        self.idx_to_class = {idx: str(idx) for idx in classes}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, int, torch.Tensor]:
        row = self.df.iloc[idx]
        guid = str(row["filename"]).strip()
        # Files are stored as <guid>.wav
        audio_path = self.files_dir / f"{guid}.wav"
        if not audio_path.exists():
            # Also allow bare GUIDs containing extension in index
            alt_path = self.files_dir / guid
            if alt_path.exists():
                audio_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Audio file not found for row {idx}: {audio_path}"
                )

        label = int(row["label"])  # already numeric per README
        waveform, _ = self._load_and_preprocess_audio(audio_path)
        return waveform, str(audio_path), label, torch.tensor(0)

    def get_errors(self):
        """Return dummy ground truth for CSEM dataset (no ground truth available).

        Returns empty lists to bypass scoring requirements while allowing
        ranking generation to proceed.
        """
        # For near duplicates: return empty pairs
        # For other issues: return empty error indicators
        return [], []
