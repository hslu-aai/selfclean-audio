import math
from pathlib import Path

import torch
import torchaudio

from selfclean_audio.datasets.base import BaseAudioDataset
from selfclean_audio.datasets.off_topic_dataset import OffTopicDataset


class _TestBaseDataset(BaseAudioDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.files = []

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        raise NotImplementedError


def _write_sine(path: Path, freq=440, seconds=0.4, sr=22050, channels=1):
    t = torch.arange(0, int(seconds * sr)) / sr
    wave = torch.sin(2 * math.pi * freq * t)
    if channels == 2:
        wave = torch.stack([wave, wave], dim=0)
    else:
        wave = wave.unsqueeze(0)
    torchaudio.save(str(path), wave.to(torch.float32), sr)
    return wave, sr


def test_base_audio_preprocess_resample_and_pad_trim(tmp_path):
    """Goal: Verify BaseAudioDataset preprocessing: mono conversion, resampling, and pad/trim to target duration."""
    # Create a stereo 22.05kHz file of 0.4s
    p = tmp_path / "a.wav"
    _write_sine(p, seconds=0.4, sr=22050, channels=2)

    # Dataset configured to mono, 16kHz, and target 0.5s
    ds = _TestBaseDataset(
        root=tmp_path, convert_mono=True, sample_rate=16000, target_duration_sec=0.5
    )
    wav, sr = ds._load_and_preprocess_audio(p)
    # Resampled to 16kHz
    assert sr == 16000
    # Mono
    assert wav.shape[0] == 1
    # Padded to 0.5s
    assert wav.shape[1] == int(0.5 * 16000)

    # Now a longer file 0.8s -> should trim to 0.5s
    p2 = tmp_path / "b.wav"
    _write_sine(p2, seconds=0.8, sr=16000, channels=1)
    wav2, sr2 = ds._load_and_preprocess_audio(p2)
    assert sr2 == 16000
    assert wav2.shape == wav.shape


def test_offtopic_match_audio_shape_variants(tmp_path):
    """Goal: Ensure OffTopicDataset._match_audio_shape handles 1D/2D tensors with pad/crop and channel changes."""

    # Build minimal OffTopicDataset to access helper
    class _B:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return torch.randn(1, 100), str(tmp_path / "x.wav"), torch.tensor(0)

    ds = OffTopicDataset(dataset=_B(), frac_error=0.0, contamination_strategy="noise")

    # 1D target shape case
    src = torch.randn(100)
    out = ds._match_audio_shape(src, torch.Size([150]))
    assert out.shape == torch.Size([150])
    out2 = ds._match_audio_shape(src, torch.Size([60]))
    assert out2.shape == torch.Size([60])

    # 2D target shape case (channels, length)
    src2 = torch.randn(1, 80)
    out3 = ds._match_audio_shape(src2, torch.Size([2, 120]))
    assert out3.shape == torch.Size([2, 120])
    out4 = ds._match_audio_shape(torch.randn(3, 50), torch.Size([1, 30]))
    assert out4.shape == torch.Size([1, 30])
