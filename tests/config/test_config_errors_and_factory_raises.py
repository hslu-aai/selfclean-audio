import pytest
from omegaconf import OmegaConf

from selfclean_audio.config import LazyConfig, LazyFactory


def test_lazyconfig_bad_relative_import_raises(tmp_path):
    """Goal: Confirm LazyConfig.load raises ImportError for missing relative imports in config files."""
    d = tmp_path / "cfg"
    d.mkdir(parents=True, exist_ok=True)
    (d / "cfg.py").write_text("from .missing import CFG\n")
    with pytest.raises(ImportError):
        LazyConfig.load(str(d / "cfg.py"))


@pytest.mark.parametrize("issue_type", ["foo_bar", "off_topic_xyz"])
def test_build_dataloader_unknown_issue_type_raises(tmp_path, issue_type):
    """Goal: Check LazyFactory.build_dataloader guards against unknown ISSUE_TYPE values and variants."""
    esc_root = tmp_path / "esc"
    esc_root.mkdir(parents=True)
    esc_meta = tmp_path / "esc.csv"
    esc_meta.write_text("filename,fold,target,category,esc10,esc50,src_file,take\n")
    cfg = OmegaConf.create(
        {
            "ISSUE_TYPE": issue_type,
            "FRAC_ERROR": 0.1,
            "ESC50_ROOT": str(esc_root),
            "ESC50_META": str(esc_meta),
            "NOISE_ROOT": str(esc_root),
            "dataloader": {
                "num_workers": 0,
                "batch_size": 1,
                "drop_last": False,
                "pin_memory": False,
            },
            "SEED": 1,
        }
    )
    with pytest.raises(ValueError):
        LazyFactory.build_dataloader(cfg)
