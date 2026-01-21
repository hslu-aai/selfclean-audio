import csv
import subprocess
import sys
from pathlib import Path


def _write_config(
    run_dir: Path,
    issue_type: str = "duplicates",
    frac: float = 0.1,
    model: str = "beats",
):
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = f"""
ISSUE_TYPE: {issue_type}
FRAC_ERROR: {frac}
MODEL_TYPE: {model}
selfclean_audio:
  near_duplicate_method: embedding_distance
  off_topic_method: lad
  label_error_method: intra_extra_distance
"""
    (run_dir / "config.yaml").write_text(cfg)


def _write_score(run_dir: Path, issue: str, metrics: dict):
    p = run_dir / f"Score-{issue}.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        for k, v in metrics.items():
            w.writerow([k, v])


def test_aggregate_results_smoke(tmp_path):
    """Goal: Validate aggregator script reads run configs/scores and writes all_results.csv."""
    base = tmp_path / "outputs"
    run = base / "beats_esc50_duplicates_frac0.1"
    _write_config(run, issue_type="duplicates", frac=0.1, model="beats")
    _write_score(
        run, "near_duplicates", {"evaluation/AUROC": 0.8, "evaluation/AP": 0.75}
    )
    _write_score(
        run, "off_topic_samples", {"evaluation/AUROC": 0.5, "evaluation/AP": 0.4}
    )
    _write_score(run, "label_errors", {"evaluation/AUROC": 0.6, "evaluation/AP": 0.5})

    out_dir = base / "aggregates"
    cmd = [
        sys.executable,
        "scripts/collect_results.py",
        "--base-dir",
        str(base),
        "--out-dir",
        str(out_dir),
    ]
    subprocess.check_call(cmd)

    # Only all_results.csv is written
    assert (out_dir / "all_results.csv").exists()
