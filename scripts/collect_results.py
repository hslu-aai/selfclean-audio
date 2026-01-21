"""
Collect all run results into a single flat CSV.

Scans <base-dir> (default: ./outputs) for run folders with a config.yaml and
Score-*.csv files. For each run, it selects the Score-<issue>.csv that matches
the configured ISSUE_TYPE and appends its metrics along with key config fields
into one DataFrame, written to <base>/aggregates/all_results.csv.

Usage:
  python scripts/collect_results.py --base-dir outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml


def find_run_dirs(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []
    return sorted(p.parent for p in base_dir.rglob("config.yaml"))


def load_config(p: Path) -> dict:
    try:
        return yaml.safe_load(p.read_text())
    except yaml.constructor.ConstructorError:
        # Try with unsafe_load for files with Python-specific YAML tags
        try:
            return yaml.unsafe_load(p.read_text())
        except Exception:
            return {}
    except Exception:
        return {}


def relevant_issue_for_noise(issue_type: Optional[str]) -> Optional[str]:
    """Map config ISSUE_TYPE variants to canonical score/ranking file keys.

    Accepts broader aliases so aggregation works with different naming styles
    used across scripts/configs.
    """
    if issue_type is None:
        return None
    it = str(issue_type).strip().lower()

    # Duplicates family (map to near_duplicates)
    if it in {
        "duplicates",
        "near_duplicates",
        "cropped_duplicates",
        "noisy_duplicates",
        "mixed_duplicates",
        "combined_duplicates",
    }:
        return "near_duplicates"

    # Label errors
    if it in {"label_errors", "label_error"}:
        return "label_errors"

    # Off-topic family (map to off_topic_samples)
    if (
        it
        in {
            "off_topic",
            "offtopic",
            "off-topic",
            "off_topic_samples",
            "off_topic_noise",
            "off_topic_external",
            "off_topic_corrupted",
            "off_topic_combined",
        }
        or "off_topic" in it
    ):
        return "off_topic_samples"

    return None


def load_score_row(run_dir: Path, issue_key: Optional[str]) -> dict | None:
    if not issue_key:
        return None
    score_path = run_dir / f"Score-{issue_key}.csv"
    if not score_path.exists():
        return None
    try:
        df = pd.read_csv(score_path)
        if set(df.columns) >= {"Metric", "Value"}:
            return df.set_index("Metric")["Value"].to_dict()
    except Exception:
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="outputs")
    ap.add_argument("--out-dir", default=None, help="Defaults to <base-dir>/aggregates")
    ap.add_argument(
        "--filter",
        choices=["all", "synthetic", "gtzan"],
        default="all",
        help="Filter results by dataset type",
    )
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir) if args.out_dir else base_dir / "aggregates"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for run in find_run_dirs(base_dir):
        cfg = load_config(run / "config.yaml")
        issue_key = relevant_issue_for_noise(cfg.get("ISSUE_TYPE"))
        metrics = load_score_row(run, issue_key)
        if metrics is None:
            continue

        # Apply dataset filter
        run_name = run.name
        is_gtzan = run_name.startswith("gtzan_")
        is_synthetic = "_esc50_" in run_name or (
            not is_gtzan
            and (
                "_combined_duplicates_" in run_name
                or "_label_errors_" in run_name
                or "_off_topic_" in run_name
                or "__baseline" in run_name
                or "__obj-" in run_name
            )
        )

        if args.filter == "synthetic" and not is_synthetic:
            continue
        elif args.filter == "gtzan" and not is_gtzan:
            continue

        row = {
            "run_dir": str(run),
            "run_name": run_name,
            "model_type": cfg.get("MODEL_TYPE"),
            "noise_type": cfg.get("ISSUE_TYPE"),
            "frac_error": cfg.get("FRAC_ERROR"),
            "near_duplicate_method": cfg.get("near_duplicate_method"),
            "off_topic_method": cfg.get("off_topic_method"),
            "label_error_method": cfg.get("label_error_method"),
        }
        sc = cfg.get("selfclean_audio")
        if isinstance(sc, dict):
            row.update(
                {
                    k: sc.get(k)
                    for k in [
                        "near_duplicate_method",
                        "off_topic_method",
                        "label_error_method",
                    ]
                    if row.get(k) is None
                }
            )
        row.update(metrics)
        rows.append(row)

    if not rows:
        print(f"No results found under {base_dir}")
        return

    df = pd.DataFrame(rows)

    # Choose output filename based on filter
    if args.filter == "synthetic":
        out_filename = "synthetic_results.csv"
    elif args.filter == "gtzan":
        out_filename = "gtzan_results.csv"
    else:
        out_filename = "all_results.csv"

    out_path = out_dir / out_filename
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
