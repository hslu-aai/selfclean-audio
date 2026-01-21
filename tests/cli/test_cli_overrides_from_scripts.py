import csv
from argparse import Namespace
from pathlib import Path

import pytest


class _DummyIssueManager:
    def __init__(self):
        # minimal structure expected by __main__ for writing score CSVs
        self.issue_dict = {"Scores-near_duplicates": {"evaluation/AUROC": 1.0}}


class _DummySelfClean:
    def __init__(self, cfg):
        # capture key fields that should have been propagated to initialization
        self._cfg = cfg
        self.cleaner = type(
            "C",
            (),
            {
                "near_duplicate_method": cfg.selfclean_audio.get(
                    "near_duplicate_method", None
                ),
                "near_duplicate_params": cfg.selfclean_audio.get(
                    "near_duplicate_params", None
                ),
                "off_topic_method": cfg.selfclean_audio.get("off_topic_method", None),
                "off_topic_params": cfg.selfclean_audio.get("off_topic_params", None),
                "label_error_method": cfg.selfclean_audio.get(
                    "label_error_method", None
                ),
                "label_error_params": cfg.selfclean_audio.get(
                    "label_error_params", None
                ),
            },
        )()
        # issues set by LazyFactory in our code (not used here)
        self.issues_to_detect = []

    def run_on_dataloader(self, dataloader):  # noqa: ARG002
        return _DummyIssueManager()


@pytest.fixture()
def tiny_cfg_file(tmp_path: Path) -> Path:
    # Create a tiny valid config mirroring config/templates/file_template.py
    # but using local paths and CPU.
    root = tmp_path / "esc"
    root.mkdir(parents=True, exist_ok=True)
    (root / "a.wav").write_bytes(b"\x00\x00")  # not used
    meta = tmp_path / "esc.csv"
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
        w.writerow(["a.wav", 1, 0, "c0", 0, 1, "x", 1])

    cfg_py = tmp_path / "cfg.py"
    cfg_py.write_text(
        f"""
from selfclean_audio.config import LazyCall as L
from selfclean_audio.selfclean_audio import SelfCleanAudio, PretrainingSSL

ISSUE_TYPE = "duplicates"
FRAC_ERROR = 0.1
SEED = 1

ESC50_ROOT = r"{root}"
ESC50_META = r"{meta}"
NOISE_ROOT = r"{root}"

MODEL_TYPE = "beats"
MODEL_PATHS = {{
  "beats": "dummy_beats",
  "eat": "dummy_eat",
}}
MODEL_ENUMS = {{
  "beats": PretrainingSSL.BEATS,
  "eat": PretrainingSSL.EAT_BASE_PRETRAIN,
}}

dataloader = dict(num_workers=0, batch_size=1, drop_last=False, pin_memory=False)

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

params = dict(
  seed=SEED,
  cudnn_benchmark=False,
  cudnn_deterministic=True,
)
"""
    )
    return cfg_py


def test_get_file_results_style_overrides_propagate(
    monkeypatch, tiny_cfg_file, tmp_path
):
    """Goal: Verify CLI overrides propagate through LazyConfig to selfclean_audio, including model mapping and baseline methods."""
    # Patch LazyFactory methods to capture the effective config and avoid heavy work
    import selfclean_audio.__main__ as entry
    import selfclean_audio.config as cfgmod

    captured = {}

    def fake_build_selfclean_audio(cfg):
        captured["cfg"] = cfg
        return _DummySelfClean(cfg)

    def fake_build_dataloader(cfg):  # noqa: ARG001
        class _D:
            batch_size = 1

            def __len__(self):
                return 1

            @property
            def dataset(self):  # pragma: no cover - not used
                return []

        return _D()

    monkeypatch.setattr(
        cfgmod.LazyFactory,
        "build_selfclean_audio",
        staticmethod(fake_build_selfclean_audio),
    )
    monkeypatch.setattr(
        cfgmod.LazyFactory, "build_dataloader", staticmethod(fake_build_dataloader)
    )

    # Simulate ./scripts/get_file_results.sh invocation
    ns = Namespace(
        config=str(tiny_cfg_file),
        overrides=[
            "MODEL_TYPE=eat",
            "ISSUE_TYPE=label_errors",
            "FRAC_ERROR=0.2",
            # Baseline method overrides at top-level (script users may pass these)
            "near_duplicate_method=audio_hash",
            "off_topic_method=quantile",
            "label_error_method=cleanlab",
        ],
        output_dir=str(tmp_path / "out1"),
    )
    entry.main(ns)

    # Effective config passed to builder reflects overrides and propagation logic
    cfg = captured["cfg"]
    assert cfg.MODEL_TYPE == "eat"
    # Mapped into the selfclean_audio block (enum and path)
    assert cfg.selfclean_audio.pretraining_ssl == cfg.MODEL_ENUMS["eat"]
    assert cfg.selfclean_audio.model_path == cfg.MODEL_PATHS["eat"]
    assert cfg.ISSUE_TYPE == "label_errors"
    assert pytest.approx(cfg.FRAC_ERROR, rel=0, abs=1e-9) == 0.2
    # Baseline method overrides were propagated into selfclean_audio
    assert cfg.selfclean_audio.near_duplicate_method == "audio_hash"
    assert cfg.selfclean_audio.off_topic_method == "quantile"
    assert cfg.selfclean_audio.label_error_method == "cleanlab"


def test_finetune_grid_overrides_propagate(monkeypatch, tiny_cfg_file, tmp_path):
    """Goal: Ensure LoRA and adaptation overrides from finetuning grid reach selfclean_audio fields correctly."""
    # Patch LazyFactory as before
    import selfclean_audio.__main__ as entry
    import selfclean_audio.config as cfgmod

    captured = {}

    def fake_build_selfclean_audio(cfg):
        captured["cfg"] = cfg
        return _DummySelfClean(cfg)

    def fake_build_dataloader(cfg):  # noqa: ARG001
        class _D:
            batch_size = 1

            def __len__(self):
                return 1

            @property
            def dataset(self):  # pragma: no cover - not used
                return []

        return _D()

    monkeypatch.setattr(
        cfgmod.LazyFactory,
        "build_selfclean_audio",
        staticmethod(fake_build_selfclean_audio),
    )
    monkeypatch.setattr(
        cfgmod.LazyFactory, "build_dataloader", staticmethod(fake_build_dataloader)
    )

    # Simulate one OV set assembled by scripts/finetuning/run_lora_grid.sh
    overrides = [
        "MODEL_TYPE=beats",
        "ISSUE_TYPE=duplicates",
        "FRAC_ERROR=0.1",
        # Enable LoRA and key hyperparameters
        "selfclean_audio.lora_enable=true",
        "selfclean_audio.lora_r=8",
        "selfclean_audio.lora_alpha=16",
        "selfclean_audio.lora_dropout=0.05",
        "selfclean_audio.adapt_epochs=1",
        "selfclean_audio.adapt_lr=1e-4",
        "selfclean_audio.adapt_projection_dim=256",
        "selfclean_audio.adapt_objective=infonce",
        "selfclean_audio.adapt_max_steps=200",
        "selfclean_audio.adapt_temperature=0.2",
        # Extra overrides example
        "selfclean_audio.adapt_strong_aug=true",
    ]

    ns = Namespace(
        config=str(tiny_cfg_file),
        overrides=overrides,
        output_dir=str(tmp_path / "out2"),
    )
    entry.main(ns)

    cfg = captured["cfg"]
    sc = cfg.selfclean_audio
    assert cfg.MODEL_TYPE == "beats"
    assert sc.get("lora_enable") is True
    assert sc.get("lora_r") == 8
    assert sc.get("lora_alpha") == 16
    assert float(sc.get("adapt_lr")) == pytest.approx(1e-4)
    assert sc.get("adapt_epochs") == 1
    assert sc.get("adapt_objective") == "infonce"
    assert sc.get("adapt_max_steps") == 200
    assert sc.get("adapt_temperature") == pytest.approx(0.2)
    assert sc.get("adapt_strong_aug") is True
