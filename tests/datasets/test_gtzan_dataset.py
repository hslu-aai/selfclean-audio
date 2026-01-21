from pathlib import Path

import torch
import torchaudio

from selfclean_audio.datasets.gtzan import GTZANKnownIssuesDataset


def _write_sine(path: Path, seconds=0.05, sr=16000):
    t = torch.arange(0, int(seconds * sr)) / sr
    wave = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0).to(torch.float32)
    torchaudio.save(str(path), wave, sr)


def _make_gtzan_layout(root: Path):
    # Create minimal GTZAN-like layout with 3 files
    (root / "pop").mkdir(parents=True, exist_ok=True)
    (root / "rock").mkdir(parents=True, exist_ok=True)
    _write_sine(root / "pop" / "pop.00001.wav")
    _write_sine(root / "pop" / "pop.00002.wav")
    _write_sine(root / "rock" / "rock.00003.wav")


def test_gtzan_duplicates_get_errors(tmp_path):
    genres = tmp_path / "genres"
    _make_gtzan_layout(genres)

    # Build a tiny duplicates ground-truth file with one group (Ex1)
    gt_file = tmp_path / "gtzan_ground_truth_full_exact_rep.txt"
    with gt_file.open("w", encoding="utf-8", newline="") as f:
        f.write("Id;Issue;Value\n")
        f.write("pop.00001;Exact Repetition;Ex1\n")
        f.write("pop.00002;Exact Repetition;Ex1\n")

    ds = GTZANKnownIssuesDataset(
        root=str(genres),
        issue_type="duplicates",
        gt_duplicates_file=str(gt_file),
        convert_mono=True,
        sample_rate=16000,
    )

    assert len(ds) == 3

    # Ensure one duplicate pair is returned and it matches the ids
    pairs, names = ds.get_errors()
    assert isinstance(pairs, set)
    assert len(pairs) == 1
    id_to_idx = ds.id_to_index
    expected = tuple(sorted([id_to_idx["pop.00001"], id_to_idx["pop.00002"]]))
    got = tuple(sorted(list(pairs)[0]))
    assert got == expected

    # __getitem__ loads audio and returns (waveform, path, label, noisy)
    wav, path, label, noisy = ds[0]
    assert wav.ndim == 2 and wav.shape[0] == 1
    assert isinstance(path, str)
    assert isinstance(label, int)
    assert noisy.item() in (0, 1)


def test_gtzan_label_errors_get_errors_and_noisy_flag(tmp_path):
    genres = tmp_path / "genres"
    _make_gtzan_layout(genres)

    # Build a tiny mislabels file that marks rock.00003 as mislabeled
    prep_file = tmp_path / "gtzan_ground_truth_prep.txt"
    with prep_file.open("w", encoding="utf-8", newline="") as f:
        f.write(
            "Id;Artist Repetition;Distortions;Exact Repetition;Exact Rep Value;Mislabelings;Recording Repetition;Version Repetitions\n"
        )
        f.write("pop.00001;No;No;No;;No;No;No\n")
        f.write("pop.00002;No;No;No;;No;No;No\n")
        f.write("rock.00003;No;No;No;;Yes;No;No\n")

    ds = GTZANKnownIssuesDataset(
        root=str(genres),
        issue_type="label_errors",
        gt_prep_file=str(prep_file),
        convert_mono=True,
        sample_rate=16000,
    )

    indicators = ds.get_errors()
    assert isinstance(indicators, list)
    assert len(indicators) == len(ds)

    # Only one mislabeled item
    assert sum(indicators) == 1
    idx_mis = ds.id_to_index["rock.00003"]
    assert indicators[idx_mis] == 1

    # __getitem__ returns noisy=1 for mislabeled id
    _, _, _, noisy = ds[idx_mis]
    assert noisy.item() == 1
