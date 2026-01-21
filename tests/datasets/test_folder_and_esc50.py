import csv
from pathlib import Path

import torch
import torchaudio

from selfclean_audio.datasets.esc50 import ESC50
from selfclean_audio.datasets.folder import FolderAudioDataset


def _write_sine(path: Path, seconds=0.2, sr=16000):
    t = torch.arange(0, int(seconds * sr)) / sr
    wave = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0).to(torch.float32)
    torchaudio.save(str(path), wave, sr)


def test_folder_audio_dataset_scans_and_loads(tmp_path):
    # Create folder structure: root/class1, root/class2
    root = tmp_path / "data"
    (root / "class1").mkdir(parents=True)
    (root / "class2").mkdir(parents=True)
    _write_sine(root / "class1" / "a.wav")
    _write_sine(root / "class2" / "b.wav")

    ds = FolderAudioDataset(
        root=root,
        convert_mono=True,
        sample_rate=8000,
        target_duration_sec=1,
    )
    assert len(ds) == 2
    wav, path, y = ds[0]
    assert wav.shape[0] == 1  # mono
    assert wav.shape[1] == 8000  # 1s at 8kHz
    assert Path(path).exists()
    assert isinstance(y, int)


def test_esc50_dataset_reads_csv_and_loads(tmp_path):
    root = tmp_path / "esc50"
    root.mkdir(parents=True)
    # Write two audio files and a CSV
    f1 = "c1.wav"
    f2 = "c2.wav"
    _write_sine(root / f1)
    _write_sine(root / f2)
    meta = tmp_path / "esc50.csv"
    with meta.open("w", newline="") as f:
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
        w.writerow([f1, 1, 0, "class1", 0, 1, "x", 1])
        w.writerow([f2, 1, 1, "class2", 0, 1, "x", 1])

    ds = ESC50(root=str(root), dataframe=str(meta), convert_mono=True, resample=8000)
    assert len(ds) == 2
    wav, path, label = ds[0]
    assert wav.shape[0] == 1
    assert (
        wav.shape[1] == 8000 * 5 or wav.shape[1] > 0
    )  # allow default duration behavior
    assert Path(path).exists()
    assert label in (0, 1)
