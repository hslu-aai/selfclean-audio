import csv
from pathlib import Path

import pytest
import torch
import torchaudio
from omegaconf import OmegaConf

from selfclean_audio.config import LazyFactory
from selfclean_audio.datasets.duplicate_dataset import DuplicateDataset
from selfclean_audio.datasets.folder import FolderAudioDataset
from selfclean_audio.datasets.off_topic_dataset import OffTopicDataset


def _write_sine(path: Path, seconds=0.05, sr=16000):
    t = torch.arange(0, int(seconds * sr)) / sr
    wave = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0).to(torch.float32)
    torchaudio.save(str(path), wave, sr)


def _tiny_esc50(root: Path, meta_path: Path, n: int = 6):
    root.mkdir(parents=True, exist_ok=True)
    files = []
    for i in range(n):
        f = f"f_{i}.wav"
        files.append(f)
        _write_sine(root / f)
    with meta_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "filename",
                "fold",
                "target",
                "category",
                "esc10",
                "esc50",
                "src_file",
                "take",
            ]
        )
        for i, f in enumerate(files):
            w.writerow([f, 1, i % 2, f"c{i%2}", 0, 1, "x", 1])


@pytest.mark.parametrize(
    "issue_type,expected_strategy",
    [
        ("duplicates", "exact"),
        ("noisy_duplicates", "noisy"),
        ("cropped_duplicates", "cropped"),
        ("mixed_duplicates", "mixed"),
        ("combined_duplicates", "combined"),
    ],
)
def test_build_dataloader_duplicates_variants(tmp_path, issue_type, expected_strategy):
    """Goal: Ensure duplicates ISSUE_TYPE variants map to expected DuplicateDataset strategies in LazyFactory."""
    esc_root = tmp_path / "esc"
    esc_meta = tmp_path / "esc.csv"
    _tiny_esc50(esc_root, esc_meta, n=8)

    cfg = OmegaConf.create(
        {
            "ISSUE_TYPE": issue_type,
            "FRAC_ERROR": 0.25,
            "ESC50_ROOT": str(esc_root),
            "ESC50_META": str(esc_meta),
            "NOISE_ROOT": str(esc_root),
            "SEED": 1,
            "dataloader": {
                "num_workers": 0,
                "batch_size": 2,
                "drop_last": False,
                "pin_memory": False,
            },
        }
    )
    dl = LazyFactory.build_dataloader(cfg)
    assert isinstance(dl.dataset, DuplicateDataset)
    assert dl.dataset.duplicate_strategy == expected_strategy


@pytest.mark.parametrize(
    "issue_type,expected_strategy",
    [
        ("off_topic_noise", "noise"),
        ("off_topic_external", "external"),
        ("off_topic_corrupted", "corrupted"),
        ("off_topic_combined", "combined"),
    ],
)
def test_build_dataloader_offtopic_variants(tmp_path, issue_type, expected_strategy):
    """Goal: Ensure off-topic ISSUE_TYPE variants map to expected strategies and external datasets are constructed."""
    esc_root = tmp_path / "esc"
    esc_meta = tmp_path / "esc.csv"
    noise_root = tmp_path / "noise_root"
    _tiny_esc50(esc_root, esc_meta, n=6)

    # Build a small folder dataset for external contamination
    noise_root.mkdir(parents=True, exist_ok=True)
    (noise_root / "extA").mkdir(parents=True)
    _write_sine(noise_root / "extA" / "n0.wav")

    cfg = OmegaConf.create(
        {
            "ISSUE_TYPE": issue_type,
            "FRAC_ERROR": 0.25,
            "ESC50_ROOT": str(esc_root),
            "ESC50_META": str(esc_meta),
            "NOISE_ROOT": str(noise_root),
            "SEED": 1,
            "dataloader": {
                "num_workers": 0,
                "batch_size": 2,
                "drop_last": False,
                "pin_memory": False,
            },
        }
    )

    dl = LazyFactory.build_dataloader(cfg)
    assert isinstance(dl.dataset, OffTopicDataset)
    assert dl.dataset.contamination_strategy == expected_strategy
    if issue_type in ("off_topic_external", "off_topic_combined"):
        assert isinstance(dl.dataset.contamination_dataset, FolderAudioDataset)
