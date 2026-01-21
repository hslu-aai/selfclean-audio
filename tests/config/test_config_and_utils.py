import pytest
from omegaconf import OmegaConf

from selfclean_audio.config import (
    LazyConfig,
    LazyFactory,
    TemplateType,
    _detect_template_type,
)
from selfclean_audio.datasets import utils as ds_utils


def test_lazyconfig_load_save_and_overrides(tmp_path):
    """Goal: Verify LazyConfig loading, apply_overrides propagation, template detection, and YAML save."""
    cfg_py = tmp_path / "cfg.py"
    cfg_py.write_text(
        """
ISSUE_TYPE = "label_errors"
FRAC_ERROR = 0.2
SEED = 123
dataloader = dict(num_workers=0, batch_size=1, drop_last=False, pin_memory=False)
selfclean_audio = {"foo": 1}
"""
    )
    cfg = LazyConfig.load(str(cfg_py))
    assert hasattr(cfg, "_config_path")
    assert _detect_template_type(cfg) == TemplateType.LABEL_ERRORS

    # Apply overrides and ensure _config_path persists
    cfg2 = LazyConfig.apply_overrides(
        cfg, ["FRAC_ERROR=0.5", "ISSUE_TYPE=off_topic_noise"]
    )
    assert getattr(cfg2, "_config_path", None) == str(cfg_py)
    assert cfg2.FRAC_ERROR == 0.5
    assert _detect_template_type(cfg2) == TemplateType.OFF_TOPIC

    # Save to YAML
    out_yaml = tmp_path / "saved.yaml"
    LazyConfig.save(cfg2, str(out_yaml))
    assert out_yaml.exists()


def test_build_dataloader_missing_required_keys(tmp_path):
    """Goal: Ensure LazyFactory.build_dataloader validates required keys and raises informative errors."""
    fac = LazyFactory()
    # Missing ISSUE_TYPE
    with pytest.raises(ValueError):
        fac.build_dataloader(OmegaConf.create({}))
    # Missing FRAC_ERROR
    with pytest.raises(ValueError):
        fac.build_dataloader(OmegaConf.create({"ISSUE_TYPE": "duplicates"}))
    # Missing dataset paths
    with pytest.raises(ValueError):
        fac.build_dataloader(
            OmegaConf.create({"ISSUE_TYPE": "duplicates", "FRAC_ERROR": 0.1})
        )
    # Missing dataloader section
    with pytest.raises(ValueError):
        fac.build_dataloader(
            OmegaConf.create(
                {
                    "ISSUE_TYPE": "duplicates",
                    "FRAC_ERROR": 0.1,
                    "ESC50_ROOT": "/tmp",
                    "ESC50_META": "/tmp/a.csv",
                    "NOISE_ROOT": "/tmp",
                }
            )
        )
    # Missing dataloader keys
    # Prepare minimal valid dataset files
    esc_root = tmp_path / "esc50"
    esc_root.mkdir(parents=True, exist_ok=True)
    audio_path = esc_root / "a.wav"
    # Write a tiny silent wav
    import torch
    import torchaudio

    torchaudio.save(str(audio_path), torch.zeros(1, 1600), 16000)
    esc_meta = tmp_path / "a.csv"
    with esc_meta.open("w") as f:
        f.write("filename,fold,target,category,esc10,esc50,src_file,take\n")
        f.write(f"{audio_path.name},1,0,class,0,1,src,1\n")

    base = {
        "ISSUE_TYPE": "duplicates",
        "FRAC_ERROR": 0.1,
        "ESC50_ROOT": str(esc_root),
        "ESC50_META": str(esc_meta),
        "NOISE_ROOT": str(esc_root),
        "dataloader": {},
        "SEED": 1,
    }
    with pytest.raises(ValueError):
        fac.build_dataloader(OmegaConf.create(base))
    for key in ["num_workers", "batch_size", "drop_last", "pin_memory"]:
        base["dataloader"][key] = 0 if key in ("num_workers", "batch_size") else False
        # fill one by one to trigger next missing
        if key != "pin_memory":
            with pytest.raises(ValueError):
                fac.build_dataloader(OmegaConf.create(base))


def test_datasets_utils_resize_and_scan(tmp_path):
    """Goal: Validate dataset utility functions: resize_waveform and fast_scandir with recursion."""
    # resize_waveform
    x = ds_utils.resize_waveform
    import torch

    wav = torch.ones(1, 10)
    out = x(wav, 15)
    assert out.shape[1] == 15
    out2 = x(wav, 5)
    assert out2.shape[1] == 5

    # fast_scandir
    (tmp_path / "a").mkdir(parents=True)
    (tmp_path / "a" / "b").mkdir(parents=True)
    f1 = tmp_path / "a" / "x.wav"
    f1.write_text("hi")
    f2 = tmp_path / "a" / "b" / "y.mp3"
    f2.write_text("hi")
    sf, files = ds_utils.fast_scandir(
        str(tmp_path / "a"), [".wav", ".mp3"], recursive=True
    )
    assert any(str(f1) == p for p in files)
    assert any(str(f2) == p for p in files)
