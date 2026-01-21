import csv
import shutil
from pathlib import Path

import torch
import torchaudio
from omegaconf import OmegaConf

from selfclean_audio.config import LazyFactory


def _write_sine(path: Path, seconds=0.05, sr=16000):
    t = torch.arange(0, int(seconds * sr)) / sr
    wave = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0).to(torch.float32)
    torchaudio.save(str(path), wave, sr)


def _tiny_csem(root: Path):
    (root / "files").mkdir(parents=True, exist_ok=True)
    _write_sine(root / "files" / "a.wav")
    _write_sine(root / "files" / "b.wav")
    with (root / "index.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "filename", "label"])  # header
        w.writerow([0, "a", 1])
        w.writerow([1, "b", 0])


def test_factory_build_dataloader_csem(tmp_path: Path):
    csem_root = tmp_path / "dataset_membranepumps"
    _tiny_csem(csem_root)

    cfg = OmegaConf.create(
        {
            "EVAL_DATASET": "csem",
            "CSEM_ROOT": str(csem_root),
            "dataloader": {
                "num_workers": 0,
                "batch_size": 2,
                "drop_last": False,
                "pin_memory": False,
            },
            # Minimal selfclean block to allow instantiation in follow-up test
            "selfclean_audio": {
                "_target_": "selfclean_audio.selfclean_audio.SelfCleanAudio",
                "model_path": "dummy",
                "pretraining_ssl": "beats",
            },
        }
    )

    dl = LazyFactory.build_dataloader(cfg)
    ds = dl.dataset
    from selfclean_audio.datasets.csem import CSEMMembranePumps

    try:
        assert isinstance(ds, CSEMMembranePumps)
        assert len(ds) == 2
    finally:
        shutil.rmtree(csem_root, ignore_errors=True)


def test_factory_build_selfclean_audio_does_not_override_issues_for_csem(
    monkeypatch, tmp_path: Path
):
    # Monkeypatch Embedder.load_pretrained to avoid heavy model
    import SelfClean.selfclean.core.src.pkg.embedder as embedder

    class _Stub(torch.nn.Module):  # pragma: no cover - trivial
        def extract_features(self, batch):
            if batch.ndim == 1:
                batch = batch.unsqueeze(0)
            return torch.randn(1, 8), torch.randn(1, 2, 8)

    monkeypatch.setattr(embedder.Embedder, "load_pretrained", lambda *a, **k: _Stub())

    cfg = OmegaConf.create(
        {
            "EVAL_DATASET": "CSEM",
            "CSEM_ROOT": str(tmp_path),
            "dataloader": {
                "num_workers": 0,
                "batch_size": 1,
                "drop_last": False,
                "pin_memory": False,
            },
            # Even if ISSUE_TYPE is set for other templates, CSEM path should not override
            "ISSUE_TYPE": "duplicates",
            "selfclean_audio": {
                "_target_": "selfclean_audio.selfclean_audio.SelfCleanAudio",
                "model_path": "dummy",
                "pretraining_ssl": "beats",
                # Leave issues_to_detect unset -> should remain None
            },
        }
    )

    sc = LazyFactory.build_selfclean_audio(cfg)
    # For CSEM, we do not force a single-issue list; leave None so SelfClean runs all
    assert getattr(sc, "issues_to_detect", None) is None
