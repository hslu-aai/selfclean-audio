import csv
import shutil
from pathlib import Path

import torch
import torchaudio

from selfclean_audio.datasets.csem import CSEMMembranePumps


def _write_sine(path: Path, seconds=0.05, sr=16000):
    t = torch.arange(0, int(seconds * sr)) / sr
    wave = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0).to(torch.float32)
    torchaudio.save(str(path), wave, sr)


def test_csem_dataset_basic_read(tmp_path: Path):
    root = tmp_path / "dataset_membranepumps"
    files = root / "files"
    files.mkdir(parents=True, exist_ok=True)

    # Create two wav files named by GUIDs
    _write_sine(files / "g1.wav")
    _write_sine(files / "g2.wav")

    # Write index.csv with id, filename (GUID w/o extension), label
    with (root / "index.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "filename", "label"])  # header
        w.writerow([0, "g1", 1])
        w.writerow([1, "g2", 0])

    ds = CSEMMembranePumps(root=root, sample_rate=16000, target_duration_sec=0.03)

    try:
        assert len(ds) == 2

        x0 = ds[0]
        assert isinstance(x0, tuple) and len(x0) == 4
        wav0, path0, label0, noisy0 = x0
        assert isinstance(wav0, torch.Tensor)
        assert Path(path0).name == "g1.wav"
        assert int(label0) in (0, 1)
        assert int(noisy0.item()) == 0  # dataset returns clean noisy label by default

        # Check length matches target_duration_sec * sample_rate
        assert wav0.shape[1] == int(0.03 * 16000)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_csem_dataset_missing_file_raises(tmp_path: Path):
    root = tmp_path / "dataset_membranepumps"
    (root / "files").mkdir(parents=True, exist_ok=True)
    with (root / "index.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "filename", "label"])  # header
        w.writerow([0, "no_such_guid", 1])

    ds = CSEMMembranePumps(root=root)
    try:
        _ = ds[0]
        assert False, "Expected FileNotFoundError for missing audio file"
    except FileNotFoundError:
        pass
    finally:
        shutil.rmtree(root, ignore_errors=True)
