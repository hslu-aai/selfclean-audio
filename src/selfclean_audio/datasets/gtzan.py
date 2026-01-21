# Copyright (c) Lucerne University of Applied Sciences and Arts.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from selfclean_audio.datasets.base import BaseAudioDataset


@dataclass
class _GTZANGroundTruth:
    duplicate_pairs: set[tuple[int, int]]
    mislabeled_ids: set[str]


class GTZANKnownIssuesDataset(BaseAudioDataset):
    """
    GTZAN dataset wrapper with built-in access to known data quality issues.

    - Exposes audio samples from a local GTZAN folder (``genres/<class>/*.wav``)
    - Parses ground truth CSVs with known issues from external_code
    - Provides `get_errors()` compatible with SelfClean evaluation:
        - For ISSUE_TYPE "duplicates": returns (set of (idx_i, idx_j) pairs, [..labels..])
        - For ISSUE_TYPE "label_errors": returns a list[int] of 0/1 per sample
    """

    def __init__(
        self,
        root: str | Path,
        issue_type: str = "duplicates",
        gt_duplicates_file: str | Path | None = None,
        gt_prep_file: str | Path | None = None,
        convert_mono: bool = True,
        sample_rate: int = 16000,
        target_duration_sec: float | None = 30.0,
        extensions: tuple[str, ...] = (".wav", ".mp3", ".flac"),
    ) -> None:
        super().__init__(
            root=str(root),
            convert_mono=convert_mono,
            sample_rate=sample_rate,
            target_duration_sec=target_duration_sec,
        )
        self.issue_type = issue_type
        self.extensions = extensions

        # Scan files by class
        self.class_names: list[str] = []
        self.class_to_idx: dict[str, int] = {}
        self.samples: list[tuple[str, int]] = []  # (path, label_idx)

        self._scan_dataset()

        # Map from GTZAN id (e.g., "disco.00050") to dataset index
        self.id_to_index: dict[str, int] = {
            self._path_to_gtzan_id(path): i for i, (path, _) in enumerate(self.samples)
        }

        # Parse ground truth CSVs if provided
        self._gt = self._parse_ground_truth(
            gt_duplicates_file=gt_duplicates_file, gt_prep_file=gt_prep_file
        )

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, int, torch.Tensor]:
        path, label_idx = self.samples[idx]
        waveform, _ = self._load_and_preprocess_audio(path)

        # Optional: mark known mislabeled items as noisy=1 for visualization
        gtzan_id = self._path_to_gtzan_id(path)
        noisy = 1 if gtzan_id in self._gt.mislabeled_ids else 0
        return waveform, path, label_idx, torch.tensor(noisy)

    # ------------------------------------------------------------------
    # Ground truth interfacing used by SelfCleanAudio.calculate_scores
    # ------------------------------------------------------------------
    def get_errors(self):
        """
        Return evaluation ground truth in the format expected by SelfClean.

        - If evaluating duplicates: returns (set[(i, j)], [..labels..])
        - If evaluating label errors: returns list[int] of length len(self)
        """
        if "duplicates" in self.issue_type:
            return self._gt.duplicate_pairs, ["gtzan_original", "gtzan_duplicate"]

        if "label_errors" in self.issue_type:
            indicators = [0] * len(self)
            for gtzan_id in self._gt.mislabeled_ids:
                idx = self.id_to_index.get(gtzan_id)
                if idx is not None:
                    indicators[idx] = 1
            return indicators

        raise ValueError(
            f"Unsupported issue_type for GTZAN: {self.issue_type}. Use 'duplicates' or 'label_errors'."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _scan_dataset(self) -> None:
        classes = []
        files: list[tuple[str, int]] = []

        # Expect structure: root/genres/<class>/<file>
        for cls_name in sorted(os.listdir(self.root)):
            cls_path = os.path.join(self.root, cls_name)
            if not os.path.isdir(cls_path):
                continue
            classes.append(cls_name)
        self.class_names = sorted(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        for cls in self.class_names:
            cls_path = os.path.join(self.root, cls)
            for fname in sorted(os.listdir(cls_path)):
                if any(fname.lower().endswith(ext) for ext in self.extensions):
                    files.append(
                        (os.path.join(cls_path, fname), self.class_to_idx[cls])
                    )

        if len(files) == 0:
            raise RuntimeError(
                f"No audio files found under {self.root}. Expected GTZAN layout '.../genres/<class>/*.wav'"
            )

        self.samples = files

    @staticmethod
    def _path_to_gtzan_id(path: str | Path) -> str:
        p = Path(path)
        # e.g., genres/disco/disco.00050.wav -> disco.00050
        stem = p.stem  # disco.00050
        return stem

    def _parse_ground_truth(
        self,
        gt_duplicates_file: str | Path | None,
        gt_prep_file: str | Path | None,
    ) -> _GTZANGroundTruth:
        dup_pairs: set[tuple[int, int]] = set()
        mislabeled_ids: set[str] = set()

        # Parse duplicates ground truth
        if gt_duplicates_file is not None and Path(gt_duplicates_file).exists():
            try:
                groups: dict[str, list[str]] = {}
                with open(gt_duplicates_file, "r", encoding="utf-8") as f:
                    # Expect header: Id;Issue;Value
                    _ = f.readline()
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(";")
                        if len(parts) < 3:
                            continue
                        gtzan_id, issue, value = parts[0], parts[1], parts[2]
                        if issue.strip().lower().startswith("exact repetition"):
                            if value == "":
                                # Some entries have empty group ids; ignore those
                                continue
                            groups.setdefault(value, []).append(gtzan_id)

                # Build all unique pairs per group
                for members in groups.values():
                    # map to indices; skip ids not present locally
                    idxs: list[int] = [
                        self.id_to_index[m] for m in members if m in self.id_to_index
                    ]
                    for i, j in itertools.combinations(sorted(set(idxs)), 2):
                        dup_pairs.add((i, j))
            except Exception:
                # Keep empty set if parsing fails
                pass

        # Parse mislabeled indicators (optional)
        if gt_prep_file is not None and Path(gt_prep_file).exists():
            try:
                with open(gt_prep_file, "r", encoding="utf-8") as f:
                    # Header: Id;Artist Repetition;Distortions;Exact Repetition;Exact Rep Value;Mislabelings;...
                    _ = f.readline()
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(";")
                        if len(parts) < 6:
                            continue
                        gtzan_id = parts[0].strip()
                        mislabel = parts[5].strip().lower()
                        if mislabel == "yes":
                            if gtzan_id in self.id_to_index:
                                mislabeled_ids.add(gtzan_id)
            except Exception:
                # Ignore parsing errors; treat as unknown
                pass

        return _GTZANGroundTruth(
            duplicate_pairs=dup_pairs, mislabeled_ids=mislabeled_ids
        )
