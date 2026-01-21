# Copyright (c) Lucerne University of Applied Sciences and Arts.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Literal

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

from selfclean_audio.utils.sample import extract_sample


class OffTopicDataset(Dataset):
    """
    Dataset that creates off-topic samples by:
    1. Adding samples from an unrelated dataset
    2. Adding pure noise samples
    3. Adding heavily corrupted versions of original samples

    This follows the same approach as the image domain implementation.
    """

    def __init__(
        self,
        dataset: Dataset,
        contamination_dataset: Dataset | None = None,
        frac_error: float = 0.1,
        n_errors: int | None = None,
        contamination_strategy: Literal[
            "external", "noise", "corrupted", "combined"
        ] = "combined",
        noise_level: float = 0.5,  # For noise and corrupted strategies
        random_state: int = 42,
        name: str | None = None,
    ):
        """
        Args:
            dataset: Original clean dataset
            contamination_dataset: External dataset for contamination (e.g., MUSAN for ESC50)
            frac_error: Fraction of samples to contaminate
            n_errors: Exact number of contaminated samples (overrides frac_error)
            contamination_strategy: How to create off-topic samples
            noise_level: Level of noise/corruption (0-1)
            random_state: Random seed for reproducibility
            name: Dataset name for logging
        """
        self.name = (
            name if name is not None else f"OffTopicDataset_{contamination_strategy}"
        )
        self.original_dataset = dataset
        self.contamination_dataset = contamination_dataset
        self.contamination_strategy = contamination_strategy
        self.noise_level = noise_level
        logger.info(f"Initializing {self.name}")

        # Forward sample_rate attribute for downstream components (e.g., LoRA adaptation)
        # Prefer the original dataset's attribute if present, otherwise fallback to common default
        if hasattr(self.original_dataset, "sample_rate"):
            try:
                self.sample_rate = int(getattr(self.original_dataset, "sample_rate"))
            except Exception:  # pragma: no cover - defensive
                self.sample_rate = 16000
        else:
            self.sample_rate = 16000

        # Set random seed for reproducibility
        np.random.seed(random_state)
        torch.manual_seed(random_state)

        # Determine how many samples to contaminate
        if n_errors is not None:
            n_contaminated = n_errors
        else:
            n_contaminated = math.ceil(frac_error * len(self.original_dataset))

        # Randomly select indices to contaminate
        idx_range = np.arange(0, len(self.original_dataset))
        self.contaminated_indices = set(
            np.random.choice(idx_range, size=n_contaminated, replace=False)
        )

        # Pre-generate contaminated samples for consistency
        self.contaminated_samples = {}
        for idx in self.contaminated_indices:
            audio, path, label, _ = extract_sample(self.original_dataset[idx])

            # Create off-topic sample
            contaminated_audio = self._create_off_topic_sample(audio)
            contaminated_path = f"{path}_off_topic"

            # Store contaminated version (off-topic audio, modified path, original label, noisy_label=1)
            self.contaminated_samples[idx] = (
                contaminated_audio,
                contaminated_path,
                label,
                torch.tensor(1),
            )

    def _create_off_topic_sample(self, original_audio: torch.Tensor) -> torch.Tensor:
        """Create an off-topic sample based on the contamination strategy"""

        if self.contamination_strategy == "external":
            # Use sample from external dataset
            if self.contamination_dataset is not None:
                ext_idx = np.random.randint(0, len(self.contamination_dataset))
                if hasattr(self.contamination_dataset, "__getitem__"):
                    try:
                        ext_data = self.contamination_dataset[ext_idx]
                        if isinstance(ext_data, tuple) and len(ext_data) > 0:
                            ext_audio = ext_data[0]  # First element should be audio
                            # Ensure same shape as original
                            if ext_audio.shape != original_audio.shape:
                                # Simple resize/pad/crop to match shape
                                ext_audio = self._match_audio_shape(
                                    ext_audio, original_audio.shape
                                )
                            return ext_audio
                    except Exception as e:
                        logger.error(f"Failed to get item from cont. dataset: {e}")

            # Fallback to noise if external dataset not available/working
            return self._create_noise_sample(original_audio)

        elif self.contamination_strategy == "noise":
            # Pure noise
            return self._create_noise_sample(original_audio)

        elif self.contamination_strategy == "corrupted":
            # Heavily corrupted version of original
            corrupted = original_audio.clone()
            noise = torch.normal(0, self.noise_level, size=corrupted.shape)
            # High corruption level
            corrupted = corrupted + noise
            return torch.clamp(corrupted, -1.0, 1.0)

        elif self.contamination_strategy == "combined":
            # Randomly choose between strategies
            strategy = np.random.choice(["external", "noise", "corrupted"])
            return self._create_off_topic_sample_strategy(original_audio, strategy)

        else:
            raise ValueError(
                f"Unknown contamination strategy: {self.contamination_strategy}"
            )

    def _create_off_topic_sample_strategy(
        self, audio: torch.Tensor, strategy: str
    ) -> torch.Tensor:
        """Helper to create off-topic sample with specific strategy"""
        if strategy == "external" and self.contamination_dataset is not None:
            try:
                ext_idx = np.random.randint(0, len(self.contamination_dataset))
                ext_data = self.contamination_dataset[ext_idx]
                ext_audio = ext_data[0]
                ext_audio = self._match_audio_shape(ext_audio, audio.shape)
                return ext_audio
            except Exception as e:
                logger.error(f"Failed to get item from cont. dataset: {e}")

        if strategy == "corrupted":
            corrupted = audio.clone()
            noise = torch.normal(0, self.noise_level, size=corrupted.shape)
            corrupted = corrupted + noise
            return torch.clamp(corrupted, -1.0, 1.0)

        # Default to noise
        return self._create_noise_sample(audio)

    def _create_noise_sample(self, reference_audio: torch.Tensor) -> torch.Tensor:
        """Create pure noise with same shape as reference"""
        if np.random.random() < 0.5:
            # Gaussian noise
            return torch.normal(0, self.noise_level, size=reference_audio.shape)
        else:
            # Uniform noise
            return (torch.rand_like(reference_audio) - 0.5) * 2 * self.noise_level

    def _match_audio_shape(
        self, source_audio: torch.Tensor, target_shape: torch.Size
    ) -> torch.Tensor:
        """Resize/pad/crop source audio to match target shape"""
        if source_audio.shape == target_shape:
            return source_audio

        # Handle different tensor dimensions
        if len(target_shape) == 1:  # [length]
            target_length = target_shape[0]
            if len(source_audio.shape) == 2:  # [channels, length]
                source_audio = source_audio.mean(dim=0)  # Convert to mono

            current_length = source_audio.shape[0]
            if current_length > target_length:
                # Crop
                start = np.random.randint(0, current_length - target_length)
                return source_audio[start : start + target_length]
            elif current_length < target_length:
                # Pad with repetition
                repetitions = (target_length // current_length) + 1
                repeated = source_audio.repeat(repetitions)
                return repeated[:target_length]
            else:
                return source_audio

        elif len(target_shape) == 2:  # [channels, length]
            target_channels, target_length = target_shape

            # Ensure correct channel dimension
            if len(source_audio.shape) == 1:  # [length] -> [channels, length]
                source_audio = source_audio.unsqueeze(0).repeat(target_channels, 1)
            elif source_audio.shape[0] != target_channels:
                if source_audio.shape[0] > target_channels:
                    source_audio = source_audio[:target_channels]
                else:
                    source_audio = source_audio.repeat(
                        target_channels // source_audio.shape[0] + 1, 1
                    )
                    source_audio = source_audio[:target_channels]

            # Handle length dimension
            current_length = source_audio.shape[1]
            if current_length > target_length:
                start = np.random.randint(0, current_length - target_length)
                return source_audio[:, start : start + target_length]
            elif current_length < target_length:
                repetitions = (target_length // current_length) + 1
                repeated = source_audio.repeat(1, repetitions)
                return repeated[:, :target_length]
            else:
                return source_audio

        return source_audio  # Fallback

    def __len__(self) -> int:
        """Same size as original dataset"""
        return len(self.original_dataset)

    def __getitem__(self, idx: int):
        """Get item - either original or contaminated version"""
        if idx in self.contaminated_indices:
            return self.contaminated_samples[idx]
        audio, path, label, _ = extract_sample(self.original_dataset[idx])
        return audio, path, label, torch.tensor(0)

    def get_errors(self) -> list[int]:
        """
        Return ground truth off-topic indicators.
        This matches the interface from the image domain.

        Returns:
            List of 0/1 indicating whether each sample is off-topic
        """
        return [
            1 if idx in self.contaminated_indices else 0 for idx in range(len(self))
        ]

    def info(self):
        """Print dataset information"""
        print(f"Name: {self.name}")
        print(f"Dataset size: {len(self)}")
        print(f"Number of off-topic samples: {len(self.contaminated_indices)}")
        print(f"Contamination strategy: {self.contamination_strategy}")
        print(f"Has external dataset: {self.contamination_dataset is not None}")
        print(f"Noise level: {self.noise_level}")
