#!/usr/bin/env python3
"""
Build rankings for CSEM Membrane Pumps by applying SelfClean models.

Usage examples:

  python scripts/build_csem_rankings.py \
      --config config/templates/csem_template.py \
      --output-dir data/CSEM/membranepumps_rankings

  # Off-topic only with isolation forest
  python scripts/build_csem_rankings.py \
      --config config/templates/csem_template.py \
      --issues off_topic_samples \
      off_topic_method=isolation_forest \
      --output-dir data/CSEM/membranepumps_rankings
"""

import argparse
from pathlib import Path
from typing import List

from loguru import logger

from selfclean_audio.config import LazyConfig, LazyFactory

DEFAULT_OUT = "./data/CSEM/membranepumps_rankings"


def _method_variant(
    issue_key: str, cleaner=None, cfg=None, map_default: bool = True
) -> str:
    """Return a stable method tag per issue.

    Prefer the configured method from `cfg.selfclean_audio` when available
    (robust even if the cleaner does not expose attributes). Fallback to
    the cleaner attribute if present. Finally, map canonical defaults to
    the string 'default' and otherwise return a lowercase method name.
    """
    defaults = {
        "near_duplicates": "embedding_distance",
        "off_topic_samples": "lad",
        "label_errors": "intra_extra_distance",
    }
    attr = {
        "near_duplicates": "near_duplicate_method",
        "off_topic_samples": "off_topic_method",
        "label_errors": "label_error_method",
    }[issue_key]

    method = None
    cfg_method = None
    # 1) Try to read from config (most reliable for labeling)
    try:
        if cfg is not None and hasattr(cfg, "selfclean_audio"):
            cfg_method = getattr(cfg.selfclean_audio, attr)
            method = cfg_method
    except Exception:
        cfg_method = None
        method = None

    # 2) Fallback to cleaner attribute when config is unset or simply mirrors the default
    cleaner_method = None
    if cleaner is not None:
        cleaner_method = getattr(cleaner, attr, None)
        if method is None:
            method = cleaner_method
        else:
            try:
                m_norm = str(method).strip().lower()
            except Exception:
                m_norm = None
            try:
                c_norm = (
                    str(cleaner_method).strip().lower()
                    if cleaner_method is not None
                    else None
                )
            except Exception:
                c_norm = None
            if m_norm == defaults[issue_key] and c_norm not in (None, m_norm):
                method = cleaner_method

    if method is None:
        return defaults[issue_key] if not map_default else "default"

    m = str(method).strip().lower()
    if not map_default:
        # Always return the literal method name
        return m
    return "default" if m == defaults[issue_key] else m


def _issue_keys_from_cfg(cfg, requested: List[str] | None) -> List[str]:
    if requested:
        return requested
    # Default: run all three
    return ["near_duplicates", "off_topic_samples", "label_errors"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to a config template")
    p.add_argument(
        "--issues",
        nargs="*",
        choices=["near_duplicates", "off_topic_samples", "label_errors"],
        help="Subset of issue types to export; default = all",
    )
    p.add_argument(
        "--output-dir",
        default=DEFAULT_OUT,
        help="Directory to write rankings CSV files",
    )
    # Use parse_known_args so that any KEY=VALUE pairs after '--' are treated as overrides
    args, overrides = p.parse_known_args()

    cfg = LazyConfig.load(args.config)
    cfg = LazyConfig.apply_overrides(cfg, overrides)

    # If methods are unset (None or "default"), fall back to canonical SelfClean defaults.
    # Otherwise honour whatever was provided in the config/overrides.
    enforced_methods: set[str] = set()
    if hasattr(cfg, "selfclean_audio"):
        override_keys = {
            "near_duplicate_method": "embedding_distance",
            "off_topic_method": "lad",
            "label_error_method": "intra_extra_distance",
        }

        # Support both fully-qualified (selfclean_audio.X) and shorthand overrides (X).
        alias_overrides: dict[str, str] = {}
        for raw in overrides:
            if "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key.startswith("selfclean_audio."):
                key = key.split(".", 1)[1]
            if key in override_keys:
                alias_overrides[key] = value

        for attr, default in override_keys.items():
            if attr in alias_overrides:
                try:
                    setattr(cfg.selfclean_audio, attr, alias_overrides[attr])
                    enforced_methods.add(attr)
                except Exception:
                    pass
                continue

            try:
                current = getattr(cfg.selfclean_audio, attr)
            except Exception:
                current = None

            if current in (None, "", "default"):
                try:
                    setattr(cfg.selfclean_audio, attr, default)
                except Exception:
                    pass
            else:
                try:
                    if str(current).strip().lower() != default:
                        enforced_methods.add(attr)
                except Exception:
                    enforced_methods.add(attr)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Persist the effective config for reproducibility
    LazyConfig.save(cfg, str(outdir / "config-used.yaml"))

    # Build components
    factory = LazyFactory()
    sc = factory.build_selfclean_audio(cfg)
    dataloader = factory.build_dataloader(cfg)

    logger.info("Running SelfClean on CSEM dataset...")
    issue_manager = sc.run_on_dataloader(dataloader=dataloader)

    # Determine model key for filenames
    model_key = str(getattr(cfg, "MODEL_TYPE", "MODEL")).upper()
    if not model_key or model_key == "NONE":
        try:
            model_key = sc.model.__class__.__name__
        except Exception:
            model_key = "MODEL"

    # For each requested issue type, export a DataFrame with mapped filenames
    issue_keys = _issue_keys_from_cfg(cfg, args.issues)
    # Validate that the cleaner is using the expected method per issue
    expected_default = {
        "near_duplicates": "embedding_distance",
        "off_topic_samples": "lad",
        "label_errors": "intra_extra_distance",
    }
    cleaner = getattr(sc, "cleaner", None)
    for issue_key in issue_keys:
        if cleaner is None:
            break
        attr = {
            "near_duplicates": "near_duplicate_method",
            "off_topic_samples": "off_topic_method",
            "label_errors": "label_error_method",
        }[issue_key]
        # Determine expected from cfg (override if present)
        expected = expected_default[issue_key]
        try:
            if hasattr(cfg, "selfclean_audio") and hasattr(cfg.selfclean_audio, attr):
                expected = str(getattr(cfg.selfclean_audio, attr)).strip().lower()
        except Exception:
            pass
        actual = getattr(cleaner, attr, None)
        should_enforce = attr in enforced_methods
        actual_norm = str(actual).strip().lower() if actual is not None else None
        if should_enforce and actual_norm != expected:
            raise AssertionError(
                f"Method mismatch for {issue_key}: expected '{expected}', got '{actual}'"
            )
        if not should_enforce and actual_norm is not None and actual_norm != expected:
            logger.warning(
                f"Cleaner method for {issue_key} differs from expected default: "
                f"'{actual_norm}' vs '{expected}'. Using cleaner value."
            )
    for issue_key in issue_keys:
        df = issue_manager.get_issues(issue_key, return_as_df=True)
        if df is None or len(df) == 0:
            logger.warning(f"No results for {issue_key}; skipping")
            continue

        # Add explicit rank (1-based)
        df = df.reset_index(drop=True)
        df.insert(0, "rank", df.index + 1)

        # Tag with model and variant
        cleaner = getattr(sc, "cleaner", None)
        variant = _method_variant(issue_key, cleaner=cleaner, cfg=cfg, map_default=True)

        # Check if LoRA adaptation is enabled
        lora_enabled = False
        try:
            if hasattr(cfg, "selfclean_audio") and hasattr(
                cfg.selfclean_audio, "lora_enable"
            ):
                lora_enabled = cfg.selfclean_audio.lora_enable
        except Exception:
            pass

        # Add _adapted suffix for LoRA-enabled runs
        if lora_enabled:
            variant = f"{variant}_adapted"

        fname = f"{issue_key}-{model_key}_{variant}.csv"

        out_path = outdir / fname
        df.to_csv(out_path, index=False)
        logger.info(f"Wrote {fname} with {len(df)} rows")


if __name__ == "__main__":
    main()
