"""
Batch visualization runner that generates full curves for all models and all
noise fractions (alphas) found under a base outputs directory.

It discovers combinations by scanning for `config.yaml` files and reading
`MODEL_TYPE` and `FRAC_ERROR`. For each (model, alpha), it invokes the
`generate_for` function from `visualize_full_curves.py`, which handles
issue-type prioritization (combined variants first) and plotting logic.

Usage:
  python scripts/visualize_all_curves.py --base-dir outputs
  python scripts/visualize_all_curves.py --base-dir outputs --format png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Set

import yaml


def find_models_and_alphas(base_dir: Path) -> tuple[Set[str], Set[float]]:
    models: set[str] = set()
    alphas: set[float] = set()
    for cfg_path in base_dir.rglob("config.yaml"):
        try:
            cfg = yaml.safe_load(cfg_path.read_text())
        except yaml.constructor.ConstructorError:
            # Try with unsafe_load for files with Python-specific YAML tags
            try:
                cfg = yaml.unsafe_load(cfg_path.read_text())
            except Exception:
                continue
        except Exception:
            continue
        model = cfg.get("MODEL_TYPE")
        if model is not None:
            models.add(str(model))
        frac = cfg.get("FRAC_ERROR")
        if frac is not None:
            try:
                alphas.add(float(frac))
            except Exception:
                pass
    return models, alphas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="outputs")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--format", default="png", choices=["png", "pdf"])
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir) if args.out_dir else base_dir / "aggregates" / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import with path tweak so we can import from scripts without package init
    import sys

    sys.path.append(str(Path(__file__).parent))
    from visualize_full_curves import render_full_curves_for_combo  # type: ignore

    models, alphas = find_models_and_alphas(base_dir)
    if not models or not alphas:
        print("No models or alphas discovered under", base_dir)
        return

    print(f"Discovered models: {sorted(models)}")
    print(f"Discovered alphas: {sorted(alphas)}")

    for model in sorted(models):
        for alpha in sorted(alphas):
            print(f"Generating for model={model}, alpha={alpha}...")
            try:
                render_full_curves_for_combo(
                    base_dir=base_dir,
                    out_dir=out_dir,
                    alpha=alpha,
                    model=model,
                    image_format=args.format,
                )
            except Exception as e:
                print(f"Failed for model={model}, alpha={alpha}: {e}")


if __name__ == "__main__":
    main()
