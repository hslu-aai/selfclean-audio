from pathlib import Path

from omegaconf import DictConfig

from selfclean_audio.config import LazyCall, LazyConfig, callable_to_str


def test_lazycall_inserts_target_and_types():
    """Goal: Verify LazyCall inserts _target_ and callable_to_str works for callables."""

    class T:
        def __call__(self):
            return 1

    lc = LazyCall(T)
    cfg = lc(foo=1, bar="x")
    assert isinstance(cfg, DictConfig)
    assert cfg._target_.endswith("T")
    assert cfg.foo == 1 and cfg.bar == "x"
    assert callable_to_str(len).endswith("len")


def test_lazyconfig_relative_import_and_wrapping(tmp_path: Path):
    """Goal: Exercise LazyConfig.load with relative imports and wrapping dicts and module-level configs."""
    d = tmp_path / "cfg"
    d.mkdir(parents=True, exist_ok=True)
    (d / "helper.py").write_text("CFG={'a':1,'b':2}\nL=[1,2,3]\nX=5\n")
    # Import only CFG (dict) to align with loader wrapping behavior
    (d / "cfg.py").write_text(
        "from .helper import CFG\n"
        "dataloader={'num_workers':0,'batch_size':1,'drop_last':False,'pin_memory':False}\n"
    )
    cfg = LazyConfig.load(str(d / "cfg.py"))
    assert isinstance(cfg.CFG, DictConfig)
    # dataloader declared in the module body should be wrapped at top-level
    assert isinstance(cfg.dataloader, DictConfig)
