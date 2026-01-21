import csv
from argparse import Namespace
from pathlib import Path

import torch
import torchaudio
from omegaconf import OmegaConf

from selfclean_audio.config import LazyFactory


class StubModel(torch.nn.Module):
    def __init__(self, emb_dim=12, t_steps=4):
        super().__init__()
        self.emb_dim = emb_dim
        self.t_steps = t_steps

    def extract_features(self, batch: torch.Tensor):  # type: ignore
        if batch.ndim == 1:
            batch = batch.unsqueeze(0)
        b = 1
        emb = torch.randn(b, self.emb_dim, device=batch.device)
        emb_t = torch.randn(b, self.t_steps, self.emb_dim, device=batch.device)
        return emb, emb_t


def _write_sine(path: Path, seconds=0.2, sr=16000):
    t = torch.arange(0, int(seconds * sr)) / sr
    wave = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0).to(torch.float32)
    torchaudio.save(str(path), wave, sr)


def _tiny_esc50(root: Path, meta_path: Path, n: int = 25):
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


def test_lazy_factory_builds_selfclean_audio_with_stub(monkeypatch, tmp_path):
    """Goal: Verify LazyFactory builds SelfCleanAudio with a stub model and runs end-to-end to produce scores."""
    # Monkeypatch Embedder.load_pretrained to return stub model
    import SelfClean.selfclean.core.src.pkg.embedder as embedder

    monkeypatch.setattr(
        embedder.Embedder, "load_pretrained", lambda *a, **k: StubModel()
    )

    # Minimal config with required keys
    from selfclean_audio.selfclean_audio import PretrainingSSL

    cfg = OmegaConf.create(
        {
            "ISSUE_TYPE": "duplicates",
            "FRAC_ERROR": 0.5,
            "ESC50_ROOT": str(tmp_path / "esc50"),
            "ESC50_META": str(tmp_path / "esc50.csv"),
            "NOISE_ROOT": str(tmp_path / "noise"),
            "dataloader": {
                "num_workers": 0,
                "batch_size": 1,
                "drop_last": False,
                "pin_memory": False,
            },
            "selfclean_audio": {
                "_target_": "selfclean_audio.selfclean_audio.SelfCleanAudio",
                "distance_function_path": "sklearn.metrics.pairwise.",
                "distance_function_name": "cosine_similarity",
                "chunk_size": 10,
                "memmap": False,
                "plot_distribution": False,
                "plot_top_N": None,
                "output_path": None,
                "figsize": [10, 8],
                "pretraining_ssl": PretrainingSSL.BEATS,
                "model_path": "dummy",
                "near_duplicate_method": "embedding_distance",
                "off_topic_method": "lad",
                "label_error_method": "intra_extra_distance",
                "random_seed": 42,
                "device": "cpu",
            },
        }
    )

    _tiny_esc50(Path(cfg.ESC50_ROOT), Path(cfg.ESC50_META), n=30)
    fac = LazyFactory()
    sc = fac.build_selfclean_audio(cfg)
    # Ensure any artifacts (plots) are written into a temp directory
    sc.workdir = tmp_path
    dl = fac.build_dataloader(cfg)
    # Quick sanity: run and get issue manager
    im = sc.run_on_dataloader(dl)
    # Ensure score files were computed in-memory
    for itype in sc.issues_to_detect:
        assert f"Scores-{itype.value}" in im.issue_dict


def test_main_end_to_end_with_stub(monkeypatch, tmp_path):
    """Goal: Exercise __main__.py entrypoint with a tiny config and stub model, ensuring Score-*.csv files are written."""
    # Monkeypatch Embedder.load_pretrained to return stub model
    import SelfClean.selfclean.core.src.pkg.embedder as embedder

    monkeypatch.setattr(
        embedder.Embedder, "load_pretrained", lambda *a, **k: StubModel()
    )

    # Write a tiny config .py compatible with LazyConfig
    cfg_dir = tmp_path
    cfg_py = cfg_dir / "tiny_config.py"
    esc_root = tmp_path / "esc50"
    esc_meta = tmp_path / "esc50.csv"
    _tiny_esc50(esc_root, esc_meta, n=30)

    cfg_py.write_text(
        f"""
from selfclean_audio.config import LazyCall as L
from selfclean_audio.selfclean_audio import SelfCleanAudio, PretrainingSSL

ISSUE_TYPE = "duplicates"
FRAC_ERROR = 0.5
SEED = 42

ESC50_ROOT = r"{esc_root}"
ESC50_META = r"{esc_meta}"
NOISE_ROOT = r"{esc_root}"

MODEL_TYPE = "beats"
MODEL_PATHS = {{"beats": "dummy"}}
MODEL_ENUMS = {{"beats": PretrainingSSL.BEATS}}

dataloader = dict(num_workers=0, batch_size=1, drop_last=False, pin_memory=False)

params = dict(
    seed=SEED,
    cudnn_benchmark=False,
    cudnn_deterministic=True,
)

selfclean_audio = L(SelfCleanAudio)(
    distance_function_path="sklearn.metrics.pairwise.",
    distance_function_name="cosine_similarity",
    chunk_size=10,
    memmap=False,
    plot_distribution=False,
    plot_top_N=None,
    output_path=None,
    figsize=(10, 8),
    pretraining_ssl=MODEL_ENUMS[MODEL_TYPE],
    model_path=MODEL_PATHS[MODEL_TYPE],
    near_duplicate_method="embedding_distance",
    near_duplicate_params={{}},
    off_topic_method="lad",
    off_topic_params={{}},
    label_error_method="intra_extra_distance",
    label_error_params={{}},
    random_seed=SEED,
    device="cpu",
)
"""
    )

    # Call __main__.main programmatically
    from selfclean_audio.__main__ import main

    out_dir = tmp_path / "out"
    ns = Namespace(config=str(cfg_py), overrides=[], output_dir=str(out_dir))
    main(ns)

    # Expect score CSVs for selected issues
    assert any(p.name.startswith("Score-") for p in out_dir.iterdir())
