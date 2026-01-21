"""
Create aggregated visualization figures with full ROC/PR/PRG and
annotation-effort-saved curves for SelfClean-Audio runs.

The script scans a base directory (default: ./outputs) for run folders
that contain:
  - a saved config.yaml
  - a saved ranking CSV per issue produced by SelfCleanAudio
    (Ranking-<issue>.csv with a single column 'target' of 0/1 at each rank)

It groups runs by the tuple:
  (MODEL_TYPE, near_duplicate_method, off_topic_method, label_error_method, FRAC_ERROR)

and only keeps groups that have results for all three issues:
  near_duplicates, off_topic_samples, label_errors

For each qualifying group, it overlays the three issue types on each subplot:
  - ROC (TPR vs FPR)
  - PR (Precision vs Recall)
  - PRG (Precision-Gain vs Recall-Gain)
  - Annotation Effort Saved (1 - time ratio vs fraction of positives annotated)

Outputs are saved under <base-dir>/aggregates/viz as PNG and PDF.

Example:
  python scripts/visualize_full_curves.py \
      --base-dir outputs \
      --alpha 0.05 \
      --model BEATS
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
import yaml

ISSUE_TO_FILE = {
    # Near Duplicates
    "duplicates": "near_duplicates",
    "near_duplicates": "near_duplicates",
    "cropped_duplicates": "near_duplicates",
    "noisy_duplicates": "near_duplicates",
    "mixed_duplicates": "near_duplicates",
    "combined_duplicates": "near_duplicates",
    # Label Errors
    "label_errors": "label_errors",
    # Off-topic Samples
    "off_topic": "off_topic_samples",
    "off_topic_samples": "off_topic_samples",
    "off_topic_noise": "off_topic_samples",
    "off_topic_external": "off_topic_samples",
    "off_topic_corrupted": "off_topic_samples",
    "off_topic_combined": "off_topic_samples",
}

# Priority when multiple variants exist for a canonical issue type.
# Lower index = higher priority.
VARIANT_PRIORITY = {
    "near_duplicates": [
        "combined_duplicates",
        "mixed_duplicates",
        "cropped_duplicates",
        "noisy_duplicates",
        "duplicates",
    ],
    "off_topic_samples": [
        "off_topic_combined",
        "off_topic_noise",
        "off_topic_external",
        "off_topic_corrupted",
    ],
    "label_errors": [
        "label_errors",
    ],
}


def load_yaml(p: Path) -> dict:
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


def calc_frac_time_needed(ranking: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute annotation effort statistics from a 0/1 ranking vector.

    Returns:
      fraction_annotated_random: cumulative fraction of positives covered
      ratio: time ratio vs. random at each step
      average_annotation_time_fraction: scalar average time fraction vs. random
    """
    N = len(ranking)
    N_T = np.sum(ranking == 1)
    if N == 0 or N_T == 0:
        return np.array([]), np.array([]), np.nan

    fraction_annotated_random = np.cumsum(ranking) / N_T
    fraction_annotated_selfclean = (np.arange(N) + 1) / N
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = fraction_annotated_selfclean / fraction_annotated_random
        avg_fraction = np.nansum(np.diff(fraction_annotated_random, prepend=0) * ratio)
    return fraction_annotated_random, ratio, float(avg_fraction)


def compute_curves_from_ranking(ranking: np.ndarray) -> Dict[str, np.ndarray]:
    """Reproduce the stepwise ROC/PR/PRG arrays used in core plotting.

    ranking is a 0/1 array where 1 indicates a true positive at that rank.
    """
    target = np.asarray(ranking).astype(int)
    n_true = np.sum(target == 1)
    n_false = np.sum(target == 0)
    if n_true == 0 or n_false == 0:
        # Degenerate case
        return {
            "fpr": np.array([0.0, 1.0]),
            "tpr": np.array([0.0, 1.0]),
            "precision": np.array([1.0, 1.0]),
            "recall": np.array([0.0, 1.0]),
            "precision_gain": np.array([0.0, 1.0]),
            "recall_gain": np.array([0.0, 1.0]),
        }

    n_t = np.cumsum(target)
    n_f = np.cumsum(1 - target)
    tpr = n_t / n_true
    fpr = n_f / n_false
    precision = n_t / (n_t + n_f)

    # precision/recall gain
    pi = n_true / len(target)
    with np.errstate(all="ignore"):
        precision_gain = (precision - pi) / ((1 - pi) * precision)
        precision_gain = np.clip(precision_gain, 0, 1)
        recall_gain = (tpr - pi) / ((1 - pi) * tpr)
        recall_gain = np.clip(recall_gain, 0, 1)

    return {
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": tpr,  # recall == tpr in this ranking-based setup
        "precision_gain": precision_gain,
        "recall_gain": recall_gain,
    }


@dataclass(frozen=True)
class GroupKey:
    model_type: str
    near_duplicate_method: str
    off_topic_method: str
    label_error_method: str
    frac_error: float


def group_key_from_cfg(cfg: dict) -> GroupKey:
    sc = (
        cfg.get("selfclean_audio", {})
        if isinstance(cfg.get("selfclean_audio"), dict)
        else {}
    )
    return GroupKey(
        model_type=str(cfg.get("MODEL_TYPE", "")).strip(),
        near_duplicate_method=str(
            cfg.get("near_duplicate_method", sc.get("near_duplicate_method", ""))
        ).strip(),
        off_topic_method=str(
            cfg.get("off_topic_method", sc.get("off_topic_method", ""))
        ).strip(),
        label_error_method=str(
            cfg.get("label_error_method", sc.get("label_error_method", ""))
        ).strip(),
        frac_error=float(cfg.get("FRAC_ERROR", np.nan))
        if cfg.get("FRAC_ERROR") is not None
        else np.nan,
    )


def find_runs(base_dir: Path) -> List[Path]:
    return sorted(p.parent for p in base_dir.rglob("config.yaml"))


def load_ranking(run_dir: Path, issue_key: str) -> Optional[np.ndarray]:
    p = run_dir / f"Ranking-{issue_key}.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        if "target" in df.columns:
            return df["target"].astype(int).to_numpy()
    except Exception:
        return None
    return None


def render_full_curves_for_combo(
    base_dir: str | Path,
    out_dir: str | Path | None = None,
    alpha: float | None = None,
    model: str | None = None,
    image_format: str = "png",
):
    base_dir = Path(base_dir)
    out_dir = Path(out_dir) if out_dir else base_dir / "aggregates" / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect per-group issues
    # For each group, keep the best run per canonical issue according to priority
    groups: Dict[GroupKey, Dict[str, Tuple[int, Path]]] = {}
    for run in find_runs(base_dir):
        cfg = load_yaml(run / "config.yaml")
        key = group_key_from_cfg(cfg)
        if alpha is not None and not np.isclose(key.frac_error, float(alpha)):
            continue
        if model is not None and key.model_type.lower() != str(model).lower():
            continue
        issue_type = str(cfg.get("ISSUE_TYPE"))
        issue_key = ISSUE_TO_FILE.get(issue_type, None)
        if issue_key is None:
            continue
        # compute priority rank
        priority_list = VARIANT_PRIORITY.get(issue_key, [])
        try:
            rank = priority_list.index(issue_type)
        except ValueError:
            # unknown variant -> lowest priority
            rank = 999
        cur = groups.setdefault(key, {}).get(issue_key)
        if cur is None or rank < cur[0]:
            groups[key][issue_key] = (rank, run)

    # Process only groups that have all 3 issues
    wanted = ["near_duplicates", "off_topic_samples", "label_errors"]
    generated: list[Path] = []
    for key, m in groups.items():
        if not all(k in m for k in wanted):
            continue

        curves: Dict[str, Dict[str, np.ndarray]] = {}
        eff: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
        for issue in wanted:
            ranking = load_ranking(m[issue][1], issue)
            if ranking is None:
                # Skip figure if any ranking missing
                curves = {}
                break
            curves[issue] = compute_curves_from_ranking(ranking)
            eff[issue] = calc_frac_time_needed(ranking)

        if not curves:
            continue

        # Plot with science style (no LaTeX to avoid dependency issues)
        with plt.style.context(["science", "no-latex", "std-colors", "grid"]):
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            colors = {
                "near_duplicates": "tab:blue",
                "off_topic_samples": "tab:orange",
                "label_errors": "tab:green",
            }
            labels = {
                "near_duplicates": "Near Duplicates",
                "off_topic_samples": "Off-topic Samples",
                "label_errors": "Label Errors",
            }

            for issue in wanted:
                c = colors[issue]
                data = curves[issue]
                # ROC
                axes[0].plot(data["fpr"], data["tpr"], label=labels[issue], color=c)
                # PR
                axes[1].plot(data["recall"], data["precision"], color=c)
                # PRG
                axes[2].plot(data["recall_gain"], data["precision_gain"], color=c)
                # Annotation Effort Saved (1 - time ratio)
                frac, ratio, avg = eff[issue]
                if len(frac) > 0:
                    axes[3].plot(frac, ratio, color=c)

            axes[0].set_title("ROC")
            axes[0].set_xlabel("FPR", fontsize=14)
            axes[0].set_ylabel("TPR", fontsize=14)
            axes[0].set_xlim([-0.05, 1.05])
            axes[0].set_ylim([-0.05, 1.05])
            axes[0].legend()

            axes[1].set_title("Precision-Recall")
            axes[1].set_xlabel("Recall", fontsize=14)
            axes[1].set_ylabel("Precision", fontsize=14)
            axes[1].set_xlim([-0.05, 1.05])
            axes[1].set_ylim([-0.05, 1.05])

            axes[2].set_title("PR-Gain")
            axes[2].set_xlabel("Recall-Gain", fontsize=14)
            axes[2].set_ylabel("Precision-Gain", fontsize=14)
            axes[2].set_xlim([-0.05, 1.05])
            axes[2].set_ylim([-0.05, 1.05])

            axes[3].set_xlabel("Recall", fontsize=14)
            axes[3].set_ylabel("FoE", fontsize=14)
            axes[3].set_xlim([-0.05, 1.05])

        title = (
            f"{key.model_type} | ND={key.near_duplicate_method} "
            f"OT={key.off_topic_method} LE={key.label_error_method} | "
            f"alpha={key.frac_error}"
        )
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        stem = (
            f"{key.model_type}_alpha{key.frac_error}_ND-{key.near_duplicate_method}_"
            f"OT-{key.off_topic_method}_LE-{key.label_error_method}"
        )
        img_path = out_dir / f"curves_{stem}.{image_format}"
        fig.savefig(img_path, dpi=200)
        # Always also save PDF for vector quality
        fig.savefig(out_dir / f"curves_{stem}.pdf")
        plt.close(fig)
        generated.append(img_path)
        generated.append(out_dir / f"curves_{stem}.pdf")

        # Save average FoE values to txt file
        foe_txt_path = out_dir / f"curves_{stem}_avgFoE.txt"
        with open(foe_txt_path, "w") as f:
            f.write(f"Average Time Fraction and Effort Saved for {title}\n")
            f.write("=" * 60 + "\n\n")
            for issue in wanted:
                frac, ratio, avg_time_fraction = eff[issue]
                effort_saved = 1.0 - avg_time_fraction
                f.write(f"{labels[issue]}:\n")
                f.write(f"  Average Time Fraction: {avg_time_fraction:.4f}\n")
                f.write(f"  Average Effort Saved (1-FoE): {effort_saved:.4f}\n\n")
        generated.append(foe_txt_path)

        # Additionally, export the Annotation Effort Saved panel alone
        with plt.style.context(["science", "no-latex", "std-colors", "grid"]):
            fig2, ax2 = plt.subplots(1, 1, figsize=(4, 3))
            for issue in wanted:
                frac, ratio, _ = eff[issue]
                if len(frac) > 0:
                    ax2.plot(
                        frac,
                        ratio,
                        label=labels[issue],
                        color=colors[issue],
                        linewidth=2,
                    )
            ax2.set_xlabel("Recall", fontsize=14)
            ax2.set_ylabel("FoE", fontsize=14)
            ax2.set_xlim([-0.05, 1.05])
            ax2.legend()
            fig2.tight_layout()
            fig2.savefig(out_dir / f"Annotation_Effort_Saving_{stem}.pdf")
            fig2.savefig(out_dir / f"Annotation_Effort_Saving_{stem}.png", dpi=200)
            plt.close(fig2)
        generated.append(out_dir / f"Annotation_Effort_Saving_{stem}.pdf")
        generated.append(out_dir / f"Annotation_Effort_Saving_{stem}.png")

        # Save average FoE values for the standalone plot
        standalone_foe_txt_path = (
            out_dir / f"Annotation_Effort_Saving_{stem}_avgFoE.txt"
        )
        with open(standalone_foe_txt_path, "w") as f:
            f.write(
                "Average Time Fraction and Effort Saved for Annotation Effort Saving\n"
            )
            f.write(f"Model: {key.model_type}, Alpha: {key.frac_error}\n")
            f.write("=" * 70 + "\n\n")
            for issue in wanted:
                frac, ratio, avg_time_fraction = eff[issue]
                effort_saved = 1.0 - avg_time_fraction
                f.write(f"{labels[issue]}:\n")
                f.write(f"  Average Time Fraction: {avg_time_fraction:.4f}\n")
                f.write(f"  Average Effort Saved (1-FoE): {effort_saved:.4f}\n\n")
        generated.append(standalone_foe_txt_path)

    return generated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="outputs")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--alpha", type=float, default=None, help="Select FRAC_ERROR")
    ap.add_argument(
        "--model", type=str, default=None, help="Select MODEL_TYPE (e.g., BEATS)"
    )
    ap.add_argument(
        "--format", default="png", choices=["png", "pdf"], help="Primary image format"
    )
    args = ap.parse_args()

    render_full_curves_for_combo(
        base_dir=args.base_dir,
        out_dir=args.out_dir,
        alpha=args.alpha,
        model=args.model,
        image_format=args.format,
    )


if __name__ == "__main__":
    main()
