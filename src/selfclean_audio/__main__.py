# Copyright (c) Lucerne University of Applied Sciences and Arts.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""Interface for ``python -m selfclean_audio``."""

import argparse
import os
import warnings
from argparse import ArgumentParser
from pathlib import Path

from loguru import logger

from . import __version__
from .constants import DEFAULT_OUTPUT_DIR, LOG_FORMAT, LOG_SEPARATOR

__all__ = ["main"]

parser = ArgumentParser()
parser.add_argument(
    "--config",
    help="Path to the .py config file to generate the synthetic noise.",
)
parser.add_argument(
    "overrides",
    nargs="*",
    default=[],
    help="Config overrides as a dotlist (e.g. a.b=1 c.d=true)",
)
parser.add_argument(
    "--output-dir",
    default=DEFAULT_OUTPUT_DIR,
    help="Path to a directory to save checkpoints and job logs.",
)
parser.add_argument(
    "-v",
    "--version",
    action="version",
    version=__version__,
)


def main(_A: argparse.Namespace) -> None:
    # Defer heavy imports until runtime
    from .config import LazyConfig, LazyFactory
    from .utils.types import ensure_dictconfig, set_seeds

    # Suppress third-party deprecation warnings that we cannot address locally
    # timm moved layers from `timm.models.layers` to `timm.layers`; some
    # upstream modules still import the old path and emit a FutureWarning.
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"timm\.models\.layers",
    )
    _C = LazyConfig.load(_A.config)

    # Ensure that the type is DictConfig.
    # Otherwise the pyright complains.
    _C = ensure_dictconfig(_C)
    _C = LazyConfig.apply_overrides(_C, _A.overrides)
    _C = ensure_dictconfig(_C)

    # Apply overridden baseline method parameters to selfclean_audio config
    baseline_params = [
        "near_duplicate_method",
        "near_duplicate_params",
        "off_topic_method",
        "off_topic_params",
        "label_error_method",
        "label_error_params",
    ]

    for param in baseline_params:
        if hasattr(_C, param):
            setattr(_C.selfclean_audio, param, getattr(_C, param))

    # Ensure changes to MODEL_TYPE propagate to the embedding model selection
    try:
        if (
            hasattr(_C, "MODEL_TYPE")
            and hasattr(_C, "MODEL_ENUMS")
            and hasattr(_C, "MODEL_PATHS")
        ):
            model_key = _C.MODEL_TYPE
            if model_key in _C.MODEL_ENUMS:
                _C.selfclean_audio.pretraining_ssl = _C.MODEL_ENUMS[model_key]
            if model_key in _C.MODEL_PATHS:
                _C.selfclean_audio.model_path = _C.MODEL_PATHS[model_key]
    except Exception:
        # Do not break if template does not define these
        pass

    set_seeds(_C)

    output_dir = Path(_A.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LazyConfig.save(_C, str(output_dir / "config.yaml"))

    # Defer third-party logging import to here to keep module import light
    from SelfClean.selfclean.core.src.utils.logging import set_log_level

    set_log_level(min_log_level="INFO")
    logger.add(output_dir / "logger.txt", format=LOG_FORMAT, level="INFO")

    logger.info(LOG_SEPARATOR)
    logger.info("Command line args:")
    for arg in vars(_A):
        logger.info(f"{arg:<20}: {getattr(_A, arg)}")

    # Log requested (intended) initialization parameters from config
    try:
        logger.info("Requested init parameters (from config):")
        logger.info(f"  MODEL_TYPE:            {_C.get('MODEL_TYPE', None)}")
        logger.info(
            f"  pretraining_ssl:       {_C.selfclean_audio.get('pretraining_ssl', None)}"
        )
        logger.info(
            f"  model_path:            {_C.selfclean_audio.get('model_path', None)}"
        )
        logger.info(f"  ISSUE_TYPE:            {_C.get('ISSUE_TYPE', None)}")
        logger.info(f"  FRAC_ERROR:            {_C.get('FRAC_ERROR', None)}")
        logger.info(
            f"  near_duplicate_method: {_C.selfclean_audio.get('near_duplicate_method', None)}"
        )
        logger.info(
            f"  near_duplicate_params: {_C.selfclean_audio.get('near_duplicate_params', None)}"
        )
        logger.info(
            f"  off_topic_method:      {_C.selfclean_audio.get('off_topic_method', None)}"
        )
        logger.info(
            f"  off_topic_params:      {_C.selfclean_audio.get('off_topic_params', None)}"
        )
        logger.info(
            f"  label_error_method:    {_C.selfclean_audio.get('label_error_method', None)}"
        )
        logger.info(
            f"  label_error_params:    {_C.selfclean_audio.get('label_error_params', None)}"
        )
        logger.info(
            f"  device:                {_C.selfclean_audio.get('device', None)}"
        )
        logger.info(
            f"  memmap:                {_C.selfclean_audio.get('memmap', None)}"
        )
        logger.info(
            f"  distance_function:     {_C.selfclean_audio.get('distance_function_path', None)}{_C.selfclean_audio.get('distance_function_name', '')}"
        )
    except Exception:
        pass

    factory = LazyFactory()
    selfclean_audio = factory.build_selfclean_audio(_C)
    selfclean_audio.workdir = output_dir

    # Log what we actually initialized
    try:
        logger.info("Actual initialized components:")
        model_cls = (
            type(selfclean_audio.model).__name__
            if hasattr(selfclean_audio, "model")
            else None
        )
        logger.info(f"  model_class:           {model_cls}")
        logger.info(
            f"  device:                {getattr(selfclean_audio, 'device', None)}"
        )
        cleaner = getattr(selfclean_audio, "cleaner", None)
        if cleaner is not None:
            logger.info(
                f"  near_duplicate_method: {getattr(cleaner, 'near_duplicate_method', None)} params={getattr(cleaner, 'near_duplicate_params', None)}"
            )
            logger.info(
                f"  off_topic_method:      {getattr(cleaner, 'off_topic_method', None)} params={getattr(cleaner, 'off_topic_params', None)}"
            )
            logger.info(
                f"  label_error_method:    {getattr(cleaner, 'label_error_method', None)} params={getattr(cleaner, 'label_error_params', None)}"
            )
            logger.info(
                f"  memmap:                {getattr(cleaner, 'memmap', None)} path={getattr(cleaner, 'memmap_path', None)}"
            )
            logger.info(
                f"  distance_function:     {getattr(cleaner, 'distance_function_name', None)}"
            )
            logger.info(
                f"  chunk_size:            {getattr(cleaner, 'chunk_size', None)}  approximate_nn={getattr(cleaner, 'approximate_nn', None)}"
            )
    except Exception:
        pass

    dataloader = factory.build_dataloader(_C)
    # Log dataset/dataloader actuals
    try:
        ds = dataloader.dataset
        ds_name = type(ds).__name__
        ds_len = len(ds) if hasattr(ds, "__len__") else None
        sample_rate = getattr(ds, "sample_rate", None)
        logger.info("Dataset/Dataloader initialized:")
        logger.info(f"  dataset_class:         {ds_name}  len={ds_len}")
        logger.info(f"  sample_rate:           {sample_rate}")
        logger.info(
            f"  batch_size:            {getattr(dataloader, 'batch_size', None)}  num_workers={getattr(dataloader, 'num_workers', None)}"
        )
        logger.info(
            f"  issues_to_detect:      {getattr(selfclean_audio, 'issues_to_detect', None)}"
        )
    except Exception:
        pass

    issue_manager = selfclean_audio.run_on_dataloader(dataloader=dataloader)

    # Defer heavy imports to avoid side-effects for --version, etc.
    import pandas as pd

    for issue in selfclean_audio.issues_to_detect:
        # Issue Manager
        # df_issue = issue_manager.get_issues(issue.value, return_as_df=True)
        # df_issue.to_csv(os.path.join(output_dir, issue.value + ".csv"), index=False)

        # Ranking scores
        df_score = pd.DataFrame(
            list(issue_manager.issue_dict[f"Scores-{issue.value}"].items()),
            columns=["Metric", "Value"],
        )
        df_score.to_csv(
            os.path.join(output_dir, f"Score-{issue.value}.csv"), index=False
        )


if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)
