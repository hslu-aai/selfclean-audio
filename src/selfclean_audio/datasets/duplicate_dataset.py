# Copyright (c) Lucerne University of Applied Sciences and Arts.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
import shutil
from typing import Literal

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

from selfclean_audio.utils.sample import extract_sample


class DuplicateDataset(Dataset):
    """
    Dataset that creates near-duplicates by appending modified versions to the original dataset.
    This follows the same approach as the image domain implementation.
    """

    def __init__(
        self,
        dataset: Dataset,
        frac_error: float = 0.1,
        n_errors: int | None = None,
        duplicate_strategy: Literal[
            "exact", "noisy", "cropped", "mixed", "combined"
        ] = "exact",
        noise_level: float = 0.05,
        crop_ratio_range: tuple[float, float] = (0.1, 0.25),
        random_state: int = 42,
        name: str | None = None,
        save_to_temp: bool = True,
        temp_dir: str | None = None,
    ):
        """
        Args:
            dataset: Original clean dataset
            frac_error: Fraction of samples to duplicate
            n_errors: Exact number of duplicates (overrides frac_error)
            duplicate_strategy: Type of duplicate to create
            noise_level: Noise level for noisy duplicates
            crop_ratio_range: Range of crop ratios for cropped duplicates
            random_state: Random seed for reproducibility
            name: Dataset name for logging
        """
        self.name = (
            name if name is not None else f"DuplicateDataset_{duplicate_strategy}"
        )
        self.original_dataset = dataset
        self.duplicate_strategy = duplicate_strategy
        self.noise_level = noise_level
        self.crop_ratio_range = crop_ratio_range

        # Forward sample_rate attribute from original dataset if available
        if hasattr(dataset, "sample_rate"):
            self.sample_rate = dataset.sample_rate
        else:
            # Default sample rate (commonly used in audio processing)
            self.sample_rate = 16000
            logger.warning(
                f"Original dataset has no sample_rate attribute, using default: {self.sample_rate}"
            )

        logger.info(f"Initializing {self.name}")

        self.save_to_temp = save_to_temp
        self.temp_dir = None
        if self.save_to_temp:
            import tempfile
            from pathlib import Path

            self.temp_dir = (
                Path(temp_dir)
                if temp_dir is not None
                else Path(tempfile.mkdtemp(prefix="synthetic_audio_"))
            )
            self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Set random seed for reproducibility
        np.random.seed(random_state)
        torch.manual_seed(random_state)

        # Determine how many samples to duplicate
        if n_errors is not None:
            n_duplicates = n_errors
        else:
            n_duplicates = math.ceil(frac_error * len(self.original_dataset))

        # Randomly select indices to duplicate
        idx_range = np.arange(0, len(self.original_dataset))
        self.duplicate_indices = np.random.choice(
            idx_range, size=n_duplicates, replace=False
        )

        # Pre-generate duplicates for consistency
        self.duplicates = []
        self.duplicate_pairs = []  # (original_idx, duplicate_idx) tuples

        for i, orig_idx in enumerate(self.duplicate_indices):
            duplicate_idx = len(self.original_dataset) + i
            self.duplicate_pairs.append((orig_idx, duplicate_idx))

            # Get original sample - handle variable number of return values
            original_data = self.original_dataset[orig_idx]
            original_audio, audio_path, label, _noisy = extract_sample(original_data)

            # Create duplicate based on strategy
            duplicate_audio = self._create_duplicate(original_audio)

            # Store duplicate with same format as original + noisy_label
            if audio_path is not None:
                # Build a real path for the duplicate so audio-hash/Dejavu can read it
                if self.save_to_temp and self.temp_dir is not None:
                    from pathlib import Path

                    import torchaudio

                    orig_stem = Path(audio_path).stem
                    duplicate_path = self.temp_dir / f"{orig_stem}_duplicate_{i}.wav"
                    # Ensure waveform is 2D (C, T)
                    wav = duplicate_audio
                    if wav.ndim == 1:
                        wav = wav.unsqueeze(0)
                    # Use original dataset sample rate if available
                    sample_rate = getattr(self.original_dataset, "sample_rate", 16000)
                    try:
                        torchaudio.save(str(duplicate_path), wav, sample_rate)
                    except Exception:
                        # Fallback: clamp dtype/values and retry
                        wav = wav.to(torch.float32)
                        torchaudio.save(str(duplicate_path), wav, sample_rate)
                    duplicate_path_str = str(duplicate_path)
                else:
                    duplicate_path_str = f"{audio_path}_duplicate_{i}"

                # Duplicates are considered "noisy" (1), originals are clean (0)
                self.duplicates.append(
                    (duplicate_audio, duplicate_path_str, label, torch.tensor(1))
                )
            else:
                self.duplicates.append((duplicate_audio, label, torch.tensor(1)))

    def _create_duplicate(self, audio: torch.Tensor) -> torch.Tensor:
        """Create a duplicate of the audio based on the specified strategy."""

        if self.duplicate_strategy == "exact":
            # Exact copy
            return audio.clone()

        elif self.duplicate_strategy == "noisy":
            # Add Gaussian noise
            duplicate = audio.clone()
            noise = torch.normal(0, self.noise_level, size=duplicate.shape)
            duplicate = duplicate + noise
            return torch.clamp(duplicate, -1.0, 1.0)

        elif self.duplicate_strategy == "cropped":
            # Randomly crop part of the audio
            duplicate = audio.clone()
            if len(duplicate.shape) == 2:  # [channels, length]
                audio_length = duplicate.shape[1]
            else:  # [length]
                audio_length = duplicate.shape[0]

            crop_ratio = np.random.uniform(*self.crop_ratio_range)
            crop_length = int(audio_length * crop_ratio)

            # Random crop position
            start_pos = np.random.randint(0, max(1, audio_length - crop_length))
            end_pos = start_pos + crop_length

            # Zero out the cropped section
            if len(duplicate.shape) == 2:
                duplicate[:, start_pos:end_pos] = 0
            else:
                duplicate[start_pos:end_pos] = 0

            return duplicate

        elif self.duplicate_strategy == "mixed":
            # Mix with small amount of noise
            duplicate = audio.clone()
            noise = torch.rand_like(duplicate) * self.noise_level
            mix_ratio = np.random.uniform(*self.crop_ratio_range)
            duplicate = (1 - mix_ratio) * duplicate + mix_ratio * noise
            return torch.clamp(duplicate, -1.0, 1.0)

        elif self.duplicate_strategy == "combined":
            # Randomly select one of the other strategies for each duplicate
            strategies = ["exact", "noisy", "cropped", "mixed"]
            selected_strategy = np.random.choice(strategies)

            # Temporarily change strategy and create duplicate
            original_strategy = self.duplicate_strategy
            self.duplicate_strategy = selected_strategy
            duplicate = self._create_duplicate(audio)
            self.duplicate_strategy = original_strategy

            return duplicate

        else:
            raise ValueError(f"Unknown duplicate strategy: {self.duplicate_strategy}")

    def __len__(self) -> int:
        """Total dataset size: original + duplicates"""
        return len(self.original_dataset) + len(self.duplicates)

    def __getitem__(self, idx: int):
        """Get item from either original dataset or duplicates"""
        if idx < len(self.original_dataset):
            # Return original sample with noisy_label=0 (clean)
            original_data = self.original_dataset[idx]
            if len(original_data) == 3:
                # ESC50 format: (audio, path, label) -> (audio, path, label, noisy_label)
                audio, path, label = original_data
                return audio, path, label, torch.tensor(0)  # 0 = clean/original
            else:
                # Other formats - extend as needed
                return original_data + (torch.tensor(0),)
        else:
            # Return duplicate sample (already has noisy_label=1)
            duplicate_idx = idx - len(self.original_dataset)
            return self.duplicates[duplicate_idx]

    def cleanup_temp_dir(self) -> None:
        """Remove the temporary directory containing synthetic duplicate files."""
        if getattr(self, "save_to_temp", False) and getattr(self, "temp_dir", None):
            try:
                shutil.rmtree(str(self.temp_dir), ignore_errors=True)
            except Exception:
                pass
            finally:
                self.temp_dir = None

    def get_errors(self) -> tuple[set[tuple[int, int]], list[str]]:
        """
        Return ground truth duplicate pairs.
        This matches the interface from the image domain.

        Returns:
            Set of (original_idx, duplicate_idx) tuples
            List of error type names
        """
        return set(self.duplicate_pairs), ["original", "duplicate"]

    def info(self):
        """Print dataset information"""
        print(f"Name: {self.name}")
        print(f"Original dataset size: {len(self.original_dataset)}")
        print(f"Number of duplicates: {len(self.duplicates)}")
        print(f"Total dataset size: {len(self)}")
        print(f"Duplicate strategy: {self.duplicate_strategy}")
        print(f"Duplicate pairs: {len(self.duplicate_pairs)}")
