import csv
import subprocess
import sys
from pathlib import Path


def _write_config(run_dir: Path, issue_type: str, frac: float, model: str = "beats"):
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = f"""
ISSUE_TYPE: {issue_type}
FRAC_ERROR: {frac}
MODEL_TYPE: {model}
near_duplicate_method: embedding_distance
off_topic_method: lad
label_error_method: intra_extra_distance
selfclean_audio:
  _target_: selfclean_audio.selfclean_audio.SelfCleanAudio
  pretraining_ssl: BEATS
  model_path: BEATS
  near_duplicate_method: embedding_distance
  off_topic_method: lad
  label_error_method: intra_extra_distance
"""
    (run_dir / "config.yaml").write_text(cfg)


def _write_ranking(run_dir: Path, issue_key: str, ranking):
    p = run_dir / f"Ranking-{issue_key}.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target"])
        for r in ranking:
            w.writerow([r])


def _make_minimal_group(base: Path, frac: float = 0.05):
    # Near duplicates
    run_nd = base / f"beats_esc50_duplicates_frac{frac}"
    _write_config(run_nd, issue_type="duplicates", frac=frac, model="beats")
    _write_ranking(run_nd, "near_duplicates", [1, 0, 1, 0, 0, 1, 0, 0, 0, 1])

    # Off-topic (keep both combined and noise; combined should take priority)
    run_ot_comb = base / f"beats_esc50_off_topic_combined_frac{frac}"
    _write_config(
        run_ot_comb, issue_type="off_topic_combined", frac=frac, model="beats"
    )
    _write_ranking(run_ot_comb, "off_topic_samples", [0, 1, 0, 0, 1, 0, 0, 1, 0, 0])

    run_ot_noise = base / f"beats_esc50_off_topic_noise_frac{frac}"
    _write_config(run_ot_noise, issue_type="off_topic_noise", frac=frac, model="beats")
    _write_ranking(run_ot_noise, "off_topic_samples", [0, 0, 1, 0, 0, 1, 0, 0, 1, 0])

    # Label errors
    run_le = base / f"beats_esc50_label_errors_frac{frac}"
    _write_config(run_le, issue_type="label_errors", frac=frac, model="beats")
    _write_ranking(run_le, "label_errors", [0, 0, 0, 1, 0, 0, 1, 0, 1, 0])


def test_visualize_full_curves_smoke(tmp_path: Path):
    base = tmp_path / "outputs"
    _make_minimal_group(base, frac=0.05)

    # Run the per-combo script
    cmd = [
        sys.executable,
        "scripts/visualize_full_curves.py",
        "--base-dir",
        str(base),
        "--alpha",
        "0.05",
        "--model",
        "beats",
    ]
    subprocess.check_call(cmd)

    viz_dir = base / "aggregates" / "viz"
    assert viz_dir.exists()
    # Expect at least a combined curves figure and an effort-only figure
    assert any(p.name.startswith("curves_") for p in viz_dir.iterdir())
    assert any(
        p.name.startswith("Annotation_Effort_Saving_") for p in viz_dir.iterdir()
    )


def test_visualize_all_curves_smoke(tmp_path: Path):
    base = tmp_path / "outputs"
    _make_minimal_group(base, frac=0.05)

    cmd = [
        sys.executable,
        "scripts/visualize_all_curves.py",
        "--base-dir",
        str(base),
    ]
    subprocess.check_call(cmd)

    viz_dir = base / "aggregates" / "viz"
    assert viz_dir.exists()
    # At least some artifacts should be present
    assert any(viz_dir.iterdir())
