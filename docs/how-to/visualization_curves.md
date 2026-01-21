# Visualization: ROC/PR/PRG + Annotation Effort Saved

## Inputs

- Script: `scripts/visualize_full_curves.py`
- Input: run folders under `outputs/` that contain `config.yaml`, `Score-*.csv`,
  and `Ranking-<issue>.csv` files.

## Generating Rankings

- Rankings are saved automatically when running the CLI after 2025-09-14.
- Each run writes `Ranking-near_duplicates.csv`, `Ranking-off_topic_samples.csv`, and `Ranking-label_errors.csv` containing a `target` column of 0/1 values ordered by the method's ranking.

## Create Figures

- Example (BEATS at alpha 0.05):
  `python scripts/visualize_full_curves.py --base-dir outputs --alpha 0.05 --model BEATS`

## Batch for All Models and Alphas

- Generate every figure the repo has results for:
  `python scripts/visualize_all_curves.py --base-dir outputs`
  - Optional: `--format pdf` to only save PDFs (PNGs are always created by the per-combo script alongside the PDF of the effort plot).

## What It Produces

- For every combination that has results for all three issue types, it saves:
  - `curves_<MODEL>_alpha<ALPHA>_ND-<...>_OT-<...>_LE-<...>.png/.pdf` (ROC, PR, PRG, Effort Saved)
  - `Annotation_Effort_Saving_<MODEL>_alpha<ALPHA>_... .pdf/.png` (single panel)

## Notes

- Only groups with all three issues are visualized.
- If older runs are missing ranking CSVs, re-run the experiments to emit them.
- If multiple variants exist per issue, the script prioritizes combined variants
  when present (e.g., `off_topic_combined` over `off_topic_noise/external/corrupted`,
  `combined_duplicates` over other duplicate corruptions).
