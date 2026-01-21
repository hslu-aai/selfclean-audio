import numpy as np
import torch

from selfclean_audio.datasets.off_topic_dataset import OffTopicDataset


class _Base:
    def __init__(self, n=4, sr=16000):
        self.n = n
        self.sample_rate = sr

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.randn(1, 32), f"p{idx}.wav", torch.tensor(idx % 2)


class _BadExternal:
    def __len__(self):
        return 1

    def __getitem__(self, idx):  # always fail
        raise RuntimeError("broken external dataset")


def test_offtopic_external_fallbacks_to_noise():
    """Goal: Ensure OffTopicDataset falls back to noise generation when external contamination is unavailable."""
    base = _Base(3)
    ds = OffTopicDataset(
        dataset=base,
        frac_error=1.0,
        contamination_strategy="external",
        contamination_dataset=None,
        random_state=0,
    )
    # All samples contaminated and marked noisy
    _, _, _, noisy = ds[0]
    assert noisy.item() == 1


def test_offtopic_combined_strategies():
    """Goal: Exercise combined off-topic strategy and robustness to failing external contamination dataset."""
    np.random.seed(0)
    base = _Base(2)
    ext = _BadExternal()
    ds = OffTopicDataset(
        dataset=base,
        frac_error=1.0,
        contamination_strategy="combined",
        contamination_dataset=ext,
        random_state=0,
    )
    # Sampling returns a 4-tuple and should not crash despite failing external
    item = ds[1]
    assert len(item) == 4
