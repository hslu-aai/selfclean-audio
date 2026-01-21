from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from selfclean_audio.datasets import FolderAudioDataset

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def generate_audio(batch_size: int, num_samples: int) -> torch.Tensor:
    return torch.randn(batch_size, num_samples)


@pytest.fixture
def clean_audio():
    return torch.randn(1, 16000)


@pytest.fixture
def noise_audio():
    return torch.randn(1, 16000)


@pytest.fixture
def wav_dataset():
    # Mock the FolderAudioDataset
    dataset = MagicMock(FolderAudioDataset)

    # Set up the attributes
    dataset.extensions = [".wav", ".mp3"]  # Example extensions
    dataset.samples = {
        "class1": ["path/to/audio_0.wav", "path/to/audio_1.wav"],
        "class2": ["path/to/audio_2.wav", "path/to/audio_3.wav"],
    }

    # Class to index mapping
    dataset.class_to_idx = {
        cls: idx for idx, cls in enumerate(sorted(dataset.samples.keys()))
    }
    # Index to class mapping
    dataset.idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    # Mock __len__ and __getitem__
    dataset.__len__.return_value = 10
    dataset.__getitem__.side_effect = lambda idx: (
        torch.randn(1, 16000),  # Mock audio tensor
        f"path/to/audio_{idx}.wav",  # Path as string
        torch.tensor(
            [dataset.class_to_idx.get(f"class{idx % 2 + 1}", 0)]
        ),  # Mock label based on class
    )

    return dataset
