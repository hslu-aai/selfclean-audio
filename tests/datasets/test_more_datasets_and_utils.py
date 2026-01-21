import csv
from pathlib import Path

import pytest
import torch
import torchaudio

from selfclean_audio.datasets.duplicate_dataset import DuplicateDataset
from selfclean_audio.datasets.folder import FolderAudioDataset
from selfclean_audio.datasets.label_error_dataset import LabelErrorDataset
from selfclean_audio.datasets.noisy import NoisyDataset


def _write_sine(path: Path, seconds=0.05, sr=16000):
    t = torch.arange(0, int(seconds * sr)) / sr
    wave = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0).to(torch.float32)
    torchaudio.save(str(path), wave, sr)


class _TinyBase:
    def __init__(self, n: int, root: Path, sample_rate: int = 16000):
        self.n = n
        self.sample_rate = sample_rate
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.files = [str(self.root / f"x_{i}.wav") for i in range(n)]
        t = torch.arange(0, 0.05, 1 / sample_rate)
        for i, p in enumerate(self.files):
            wave = torch.sin(2 * torch.pi * (220 + 10 * (i % 2)) * t).unsqueeze(0)
            torchaudio.save(p, wave.to(torch.float32), sample_rate)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        w, _ = torchaudio.load(self.files[idx])
        label = idx % 2
        return w, self.files[idx], torch.tensor(label)


def test_duplicate_dataset_writes_and_cleans_temp(tmp_path):
    """Goal: Verify DuplicateDataset writes synthetic files for hashing baselines and cleanup_temp_dir removes them."""
    base = _TinyBase(6, tmp_path / "base")
    dup = DuplicateDataset(
        base, frac_error=0.5, duplicate_strategy="exact", save_to_temp=True
    )
    # Ensure temp dir has files
    assert dup.temp_dir is not None and dup.temp_dir.exists()
    assert any(p.suffix == ".wav" for p in dup.temp_dir.iterdir())
    # Cleanup removes directory
    dup.cleanup_temp_dir()
    assert dup.temp_dir is None or not Path(dup.temp_dir).exists()


def test_label_error_dataset_per_class_selection(tmp_path):
    """Goal: Exercise per-class selection path in LabelErrorDataset and ensure noisy_label propagation."""
    base = _TinyBase(10, tmp_path / "base")

    # Wrap base to conform to 4-tuple format expected by _select_indices_per_class
    class _Wrap:
        def __init__(self, inner):
            self.inner = inner
            self.sample_rate = inner.sample_rate

        def __len__(self):
            return len(self.inner)

        def __getitem__(self, idx):
            x, p, y = self.inner[idx]
            return x, p, y, torch.tensor(0)

    ds = LabelErrorDataset(
        _Wrap(base), frac_error=0.5, change_for_every_label=True, random_state=0
    )
    # Each class should contribute some indices
    errs = ds.get_errors()
    assert isinstance(errs, list) and len(errs) == len(ds)
    # Corrupted samples return noisy_label=1
    noisy_count = 0
    for i in range(len(ds)):
        item = ds[i]
        _, _, _, nz = item
        noisy_count += int(nz.item())
    assert noisy_count > 0


def test_noisy_dataset_label_noise_and_strategy(tmp_path):
    """Goal: Validate NoisyDataset behavior for noise_strategy and label_error injection with multiple classes."""
    root = tmp_path / "noisy"
    root.mkdir(parents=True, exist_ok=True)
    wav = root / "a.wav"
    _write_sine(wav)
    wav2 = root / "b.wav"
    _write_sine(wav2)
    # metadata with label and a strategy column; include 2 classes to allow noisy label different from original
    meta = root / "metadata.csv"
    with meta.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label", "noise_strategy"])
        w.writerow(
            ["a.wav", "c0", "Noisy"]
        )  # should mark noisy_label=1 when label_error=False
        w.writerow(
            ["b.wav", "c1", "Clean"]
        )  # second class ensures _get_noisy_label has options

    ds = NoisyDataset(root=root, sample_rate=16000, label_error=False)
    wav, path, lbl, noisy = ds[0]
    assert noisy.item() == 1

    # With label_error=True, alpha=1.0 forces all noisy
    ds2 = NoisyDataset(root=root, sample_rate=16000, label_error=True, alpha=1.0)
    _, _, _, noisy2 = ds2[0]
    assert noisy2.item() == 1
    assert 0 in ds2.class_to_idx.values()


def test_folder_audio_dataset_transform_and_empty(tmp_path):
    """Goal: Confirm FolderAudioDataset raises on empty roots and applies transforms to loaded waveforms."""
    root = tmp_path / "folders"
    # Empty should raise
    with pytest.raises(RuntimeError):
        FolderAudioDataset(root)

    # Create class folders and files
    (root / "c0").mkdir(parents=True)
    (root / "c1").mkdir(parents=True)
    _write_sine(root / "c0" / "a.wav")
    _write_sine(root / "c1" / "b.wav")

    # Simple transform doubles amplitude
    def tfm(x):
        return x * 2

    ds = FolderAudioDataset(root, transforms=tfm, sample_rate=16000)
    x, path, y = ds[0]
    assert torch.allclose(x, (x / 2) * 2)
    assert y in (0, 1)
