# Copyright (c) Lucerne University of Applied Sciences and Arts.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math

import numpy as np
import torch
from torch.utils.data import Dataset

from selfclean_audio.utils.sample import extract_sample


class LabelErrorDataset(Dataset):
    """
    Dataset that creates label errors by changing labels of selected samples.
    This follows the same approach as the image domain implementation.
    """

    def __init__(
        self,
        dataset: Dataset,
        frac_error: float = 0.1,
        n_errors: int | None = None,
        change_for_every_label: bool = False,
        random_state: int = 42,
        name: str | None = None,
    ):
        """
        Args:
            dataset: Original clean dataset
            frac_error: Fraction of samples with label errors
            n_errors: Exact number of errors (overrides frac_error)
            change_for_every_label: If True, change labels for each class separately
            random_state: Random seed for reproducibility
            name: Dataset name for logging
        """
        self.name = name if name is not None else f"LabelErrorDataset_{frac_error}"
        self.original_dataset = dataset
        self.frac_error = frac_error
        self.change_for_every_label = change_for_every_label

        # Forward sample_rate attribute for downstream components (e.g., LoRA adaptation)
        if hasattr(self.original_dataset, "sample_rate"):
            try:
                self.sample_rate = int(getattr(self.original_dataset, "sample_rate"))
            except Exception:  # pragma: no cover
                self.sample_rate = 16000
        else:
            self.sample_rate = 16000

        # Set random seed for reproducibility
        np.random.seed(random_state)
        torch.manual_seed(random_state)

        # Get number of classes from dataset
        if hasattr(dataset, "n_classes"):
            self.n_classes = dataset.n_classes
        elif hasattr(dataset, "classes"):
            self.n_classes = len(dataset.classes)
        else:
            # Estimate from dataset by looking at a few samples
            labels = []
            for i in range(min(100, len(dataset))):
                _, _p, label, _n = extract_sample(dataset[i])
                labels.append(int(label))
            self.n_classes = len(set(labels))

        # Determine which samples to corrupt
        if change_for_every_label:
            self.error_indices = self._select_indices_per_class(n_errors)
        else:
            if n_errors is not None:
                n_changes = n_errors
            else:
                n_changes = math.ceil(frac_error * len(self.original_dataset))

            idx_range = np.arange(0, len(self.original_dataset))
            self.error_indices = set(
                np.random.choice(idx_range, size=n_changes, replace=False)
            )

        # Pre-generate corrupted samples for consistency
        self.corrupted_samples = {}
        for idx in self.error_indices:
            audio, path, original_label, _ = extract_sample(self.original_dataset[idx])

            # Create wrong label
            corrupted_label = self._create_wrong_label(original_label)

            # Store corrupted version (same audio, same path, wrong label, noisy_label=1)
            self.corrupted_samples[idx] = (
                audio,
                path,
                corrupted_label,
                torch.tensor(1),
            )

    def _select_indices_per_class(self, n_errors_per_class: int | None) -> set[int]:
        """Select error indices separately for each class"""
        error_indices = set()

        # Group samples by class
        class_indices = {}
        for idx in range(len(self.original_dataset)):
            _, _p, label, _n = extract_sample(self.original_dataset[idx])

            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

        # Select errors for each class
        for class_label, indices in class_indices.items():
            if n_errors_per_class is not None:
                n_changes = n_errors_per_class
            else:
                n_changes = math.ceil(self.frac_error * len(indices))

            if n_changes > 0:
                selected = np.random.choice(
                    indices, size=min(n_changes, len(indices)), replace=False
                )
                error_indices.update(selected)

        return error_indices

    def _create_wrong_label(self, original_label):
        """Create a wrong label for the given original label"""
        if isinstance(original_label, torch.Tensor):
            orig_label_val = original_label.item()
        else:
            orig_label_val = original_label

        # Get all possible labels except the original
        possible_labels = list(range(self.n_classes))
        if orig_label_val in possible_labels:
            possible_labels.remove(orig_label_val)

        # Randomly select a wrong label
        wrong_label_val = np.random.choice(possible_labels)

        # Return in same format as original
        if isinstance(original_label, torch.Tensor):
            return torch.tensor(wrong_label_val, dtype=original_label.dtype)
        else:
            return wrong_label_val

    def __len__(self) -> int:
        """Same size as original dataset"""
        return len(self.original_dataset)

    def __getitem__(self, idx: int):
        """Get item - either original or corrupted version"""
        if idx in self.error_indices:
            # Return corrupted sample
            return self.corrupted_samples[idx]
        else:
            # Return original sample with noisy_label=0 (clean)
            audio, path, label, _n = extract_sample(self.original_dataset[idx])
            return audio, path, label, torch.tensor(0)

    def get_errors(self) -> list[int]:
        """
        Return ground truth label error indicators.
        This matches the interface from the image domain.

        Returns:
            List of 0/1 indicating whether each sample has a label error
        """
        return [1 if idx in self.error_indices else 0 for idx in range(len(self))]

    def info(self):
        """Print dataset information"""
        print(f"Name: {self.name}")
        print(f"Dataset size: {len(self)}")
        print(f"Fraction of label errors: {self.frac_error}")
        print(f"Number of label errors: {len(self.error_indices)}")
        print(f"Change for every label separately: {self.change_for_every_label}")
        print(f"Number of classes: {self.n_classes}")
