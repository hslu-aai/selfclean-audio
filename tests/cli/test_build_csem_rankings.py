import runpy
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


class _DummyIssueManager:
    def __init__(self) -> None:
        self._dfs = {
            "near_duplicates": pd.DataFrame(
                {
                    "indices_1": [0, 1],
                    "indices_2": [1, 0],
                    "scores": [0.9, 0.8],
                    "path_indices_1": ["/a.wav", "/b.wav"],
                    "path_indices_2": ["/b.wav", "/a.wav"],
                }
            ),
            "off_topic_samples": pd.DataFrame(
                {
                    "indices": [2, 3],
                    "scores": [0.7, 0.6],
                    "path": ["/c.wav", "/d.wav"],
                }
            ),
            "label_errors": pd.DataFrame(
                {
                    "indices": [4, 5],
                    "scores": [0.55, 0.5],
                    "path": ["/e.wav", "/f.wav"],
                }
            ),
        }

    def get_issues(self, issue_type, return_as_df=False):  # noqa: ARG002
        return self._dfs.get(issue_type)


class _DummySelfClean:
    def __init__(self, methods: Dict[str, str]):
        self.cleaner = type("C", (), methods)()

    def run_on_dataloader(self, dataloader):  # noqa: ARG002
        return _DummyIssueManager()


_MINIMAL_CFG = """
from selfclean_audio.config import LazyCall as L
from selfclean_audio.selfclean_audio import SelfCleanAudio

EVAL_DATASET = "CSEM"
MODEL_TYPE = "BEATS"
MODEL_PATHS = {"BEATS": "dummy"}

dataloader = dict(num_workers=0, batch_size=1, drop_last=False, pin_memory=False)

selfclean_audio = L(SelfCleanAudio)(model_path="dummy", pretraining_ssl="beats")
"""


def _run_script(
    monkeypatch,
    tmp_path: Path,
    *,
    cleaner_methods: Dict[str, str],
    issues: Optional[List[str]] = None,
    overrides: Optional[List[str]] = None,
) -> Tuple[Path, Dict[str, str]]:
    import sys

    import selfclean_audio.config as cfgmod

    captured_cfg: Dict[str, str] = {}

    def fake_build_selfclean_audio(cfg):
        captured_cfg.update(
            {
                "near_duplicates": getattr(
                    cfg.selfclean_audio, "near_duplicate_method", None
                ),
                "off_topic_samples": getattr(
                    cfg.selfclean_audio, "off_topic_method", None
                ),
                "label_errors": getattr(
                    cfg.selfclean_audio, "label_error_method", None
                ),
            }
        )
        return _DummySelfClean(cleaner_methods)

    def fake_build_dataloader(cfg):  # noqa: ARG001
        class _D:
            def __len__(self):  # pragma: no cover - not used by script
                return 1

        return _D()

    monkeypatch.setattr(
        cfgmod.LazyFactory,
        "build_selfclean_audio",
        staticmethod(fake_build_selfclean_audio),
    )
    monkeypatch.setattr(
        cfgmod.LazyFactory,
        "build_dataloader",
        staticmethod(fake_build_dataloader),
    )

    cfg_py = tmp_path / "c.py"
    cfg_py.write_text(_MINIMAL_CFG)
    outdir = tmp_path / "out"

    argv = [
        "build_csem_rankings.py",
        "--config",
        str(cfg_py),
        "--output-dir",
        str(outdir),
    ]
    if issues:
        argv.extend(["--issues", *issues])

    if overrides:
        argv.append("--")
        argv.extend(overrides)

    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(str(Path("scripts") / "build_csem_rankings.py"))["main"]()
    return outdir, captured_cfg


def test_build_csem_rankings_defaults(monkeypatch, tmp_path: Path):
    cleaner_methods = {
        "near_duplicate_method": "embedding_distance",
        "off_topic_method": "lad",
        "label_error_method": "intra_extra_distance",
    }

    outdir, captured = _run_script(
        monkeypatch,
        tmp_path,
        cleaner_methods=cleaner_methods,
    )

    try:
        names = {p.name for p in outdir.iterdir()}
        assert names == {
            "config-used.yaml",
            "near_duplicates-BEATS_default.csv",
            "off_topic_samples-BEATS_default.csv",
            "label_errors-BEATS_default.csv",
        }

        assert captured == {
            "near_duplicates": "embedding_distance",
            "off_topic_samples": "lad",
            "label_errors": "intra_extra_distance",
        }
    finally:
        shutil.rmtree(outdir, ignore_errors=True)


def test_build_csem_rankings_override_subset(monkeypatch, tmp_path: Path):
    cleaner_methods = {
        "near_duplicate_method": "embedding_distance",
        "off_topic_method": "isolation_forest",
        "label_error_method": "intra_extra_distance",
    }

    outdir, captured = _run_script(
        monkeypatch,
        tmp_path,
        cleaner_methods=cleaner_methods,
        issues=["off_topic_samples"],
        overrides=["off_topic_method=isolation_forest"],
    )

    try:
        names = {p.name for p in outdir.iterdir()}
        assert names == {
            "config-used.yaml",
            "off_topic_samples-BEATS_isolation_forest.csv",
        }

        assert captured == {
            "near_duplicates": "embedding_distance",
            "off_topic_samples": "isolation_forest",
            "label_errors": "intra_extra_distance",
        }
    finally:
        shutil.rmtree(outdir, ignore_errors=True)
