import math
from pathlib import Path

import pytest
import torch
import torchaudio

from selfclean_audio.datasets.duplicate_dataset import DuplicateDataset
from selfclean_audio.datasets.label_error_dataset import LabelErrorDataset
from selfclean_audio.datasets.off_topic_dataset import OffTopicDataset


class TinyWavDataset:
    """Minimal dataset that writes tiny WAV files and returns (audio, path, label)."""

    def __init__(self, root: Path, n: int = 8, sample_rate: int = 16000):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.n = n
        self._make_files()

    def _make_files(self):
        self.files = []
        self.labels = []
        t = torch.arange(0, 0.25, 1 / self.sample_rate)  # 0.25s
        base_freqs = [220, 440]  # two classes
        for i in range(self.n):
            cls = i % 2
            f = base_freqs[cls]
            wave = torch.sin(2 * math.pi * f * t).unsqueeze(0).to(torch.float32)
            path = self.root / f"sample_{i}.wav"
            torchaudio.save(str(path), wave, self.sample_rate)
            self.files.append(str(path))
            self.labels.append(cls)

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        # Return waveform, path, label
        wav, _ = torchaudio.load(self.files[idx])
        return wav, self.files[idx], torch.tensor(self.labels[idx], dtype=torch.long)


@pytest.mark.parametrize(
    "strategy",
    [
        "duplicates",
        "noisy_duplicates",
        "cropped_duplicates",
        "mixed_duplicates",
        "combined_duplicates",
    ],
)
def test_duplicate_dataset_get_errors(tmp_path, strategy):
    base = TinyWavDataset(tmp_path / "base", n=6)
    frac = 0.5
    # Map template-level names to DuplicateDataset strategies
    ds_strategy = (
        "exact" if strategy == "duplicates" else strategy.replace("_duplicates", "")
    )
    ds = DuplicateDataset(
        dataset=base,
        frac_error=frac,
        duplicate_strategy=ds_strategy,
        random_state=42,
        save_to_temp=True,
    )
    # Total length equals originals + duplicates
    assert len(ds) == len(base) + math.ceil(frac * len(base))
    # Ground truth pairs exist
    pairs, names = ds.get_errors()
    assert len(pairs) == math.ceil(frac * len(base))
    assert set(names) == {"original", "duplicate"}


def test_label_error_dataset_get_errors(tmp_path):
    base = TinyWavDataset(tmp_path / "base", n=10)
    frac = 0.3
    ds = LabelErrorDataset(dataset=base, frac_error=frac, random_state=42)
    errors = ds.get_errors()
    assert len(errors) == len(base)
    assert sum(errors) == math.ceil(frac * len(base))


@pytest.mark.parametrize("strategy", ["noise", "external", "corrupted", "combined"])
def test_off_topic_dataset_get_errors(tmp_path, strategy):
    base = TinyWavDataset(tmp_path / "base", n=10)
    external = TinyWavDataset(tmp_path / "ext", n=5)
    frac = 0.4
    ds = OffTopicDataset(
        dataset=base,
        contamination_dataset=(
            external if strategy in ["external", "combined"] else None
        ),
        frac_error=frac,
        contamination_strategy=strategy,
        random_state=42,
    )
    errors = ds.get_errors()
    assert len(errors) == len(base)
    assert sum(errors) == math.ceil(frac * len(base))
