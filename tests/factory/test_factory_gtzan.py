from pathlib import Path

import torch
import torchaudio
from omegaconf import OmegaConf

from selfclean_audio.config import LazyFactory
from selfclean_audio.datasets.gtzan import GTZANKnownIssuesDataset


def _write_sine(path: Path, seconds=0.05, sr=16000):
    t = torch.arange(0, int(seconds * sr)) / sr
    wave = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0).to(torch.float32)
    torchaudio.save(str(path), wave, sr)


def _make_gtzan_layout(root: Path):
    (root / "pop").mkdir(parents=True, exist_ok=True)
    (root / "rock").mkdir(parents=True, exist_ok=True)
    _write_sine(root / "pop" / "pop.00001.wav")
    _write_sine(root / "pop" / "pop.00002.wav")
    _write_sine(root / "rock" / "rock.00003.wav")


def test_factory_build_dataloader_gtzan_duplicates(tmp_path):
    genres = tmp_path / "genres"
    _make_gtzan_layout(genres)

    gt_file = tmp_path / "gtzan_ground_truth_full_exact_rep.txt"
    with gt_file.open("w", encoding="utf-8") as f:
        f.write("Id;Issue;Value\n")
        f.write("pop.00001;Exact Repetition;Ex1\n")
        f.write("pop.00002;Exact Repetition;Ex1\n")

    cfg = OmegaConf.create(
        {
            "EVAL_DATASET": "gtzan",
            "ISSUE_TYPE": "duplicates",
            "GTZAN_ROOT": str(genres),
            "GTZAN_GT_FILE": str(gt_file),
            "dataloader": {
                "num_workers": 0,
                "batch_size": 2,
                "drop_last": False,
                "pin_memory": False,
            },
        }
    )

    dl = LazyFactory.build_dataloader(cfg)
    assert isinstance(dl.dataset, GTZANKnownIssuesDataset)
    errors = dl.dataset.get_errors()
    assert isinstance(errors, tuple)
    pairs, _ = errors
    assert isinstance(pairs, set)
    assert len(pairs) == 1


def test_factory_build_dataloader_gtzan_label_errors(tmp_path):
    genres = tmp_path / "genres"
    _make_gtzan_layout(genres)

    prep = tmp_path / "gtzan_ground_truth_prep.txt"
    with prep.open("w", encoding="utf-8") as f:
        f.write(
            "Id;Artist Repetition;Distortions;Exact Repetition;Exact Rep Value;Mislabelings;Recording Repetition;Version Repetitions\n"
        )
        f.write("rock.00003;No;No;No;;Yes;No;No\n")

    cfg = OmegaConf.create(
        {
            "EVAL_DATASET": "gtzan",
            "ISSUE_TYPE": "label_errors",
            "GTZAN_ROOT": str(genres),
            "GTZAN_PREP_FILE": str(prep),
            "dataloader": {
                "num_workers": 0,
                "batch_size": 2,
                "drop_last": False,
                "pin_memory": False,
            },
        }
    )

    dl = LazyFactory.build_dataloader(cfg)
    assert isinstance(dl.dataset, GTZANKnownIssuesDataset)
    errors = dl.dataset.get_errors()
    assert isinstance(errors, list)
    indicators: list[int] = errors
    assert sum(indicators) == 1
