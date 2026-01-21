# Copyright (c) Lucerne University of Applied Sciences and Arts.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import builtins
import importlib.machinery
import importlib.util
import os
from collections.abc import Callable
from contextlib import contextmanager
from enum import Enum
from pathlib import Path

import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader

from .constants import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_TARGET_DURATION_SEC,
)
from .validation import (
    get_seed_from_config,
    validate_duplicate_strategy,
    validate_full_config,
    validate_gtzan_config,
    validate_off_topic_strategy,
)

__all__ = ["callable_to_str", "LazyCall", "LazyConfig", "LazyFactory", "TemplateType"]
_CFG_PACKAGE_NAME = "selfclean_audio._cfg_loader"


class TemplateType(Enum):
    """Enum for different experiment template types."""

    DUPLICATES = "duplicates"
    LABEL_ERRORS = "label_errors"
    OFF_TOPIC = "off_topic"


def _detect_template_type(cfg: DictConfig) -> TemplateType:
    """
    Detect the template type from config parameters.

    Args:
        cfg: Configuration object

    Returns:
        TemplateType: The detected template type
    """
    issue_type = getattr(cfg, "ISSUE_TYPE", "duplicates")

    # Check for duplicates (any type)
    if "duplicates" in issue_type:
        return TemplateType.DUPLICATES

    # Check for label errors
    if "label_errors" in issue_type:
        return TemplateType.LABEL_ERRORS

    # Check for off-topic
    if "off_topic" in issue_type:
        return TemplateType.OFF_TOPIC

    return TemplateType.DUPLICATES


@contextmanager
def _patch_import():
    """
    Enhance relative import statements in config files, so that they:
    1. locate files purely based on relative location, regardless of packages.
       e.g. you can import file without having __init__
    2. do not cache modules globally; modifications of module states has no side effect
    3. imported dict are turned into omegaconf.DictConfig automatically
    """
    old_import = builtins.__import__

    def find_relative_file(original_file, relative_import_path, level):
        cur_file = os.path.dirname(original_file)
        for _ in range(level - 1):
            cur_file = os.path.dirname(cur_file)
        cur_name = relative_import_path.lstrip(".")
        for part in cur_name.split("."):
            cur_file = os.path.join(cur_file, part)
        # NOTE: directory import is not handled. Because then it's unclear
        # if such import should produce python module or DictConfig. This can
        # be discussed further if needed.
        if not cur_file.endswith(".py"):
            cur_file += ".py"
        if not os.path.isfile(cur_file):
            raise ImportError(
                f"Cannot import name {relative_import_path} from "
                f"{original_file}: {cur_file} has to exist."
            )
        return cur_file

    def new_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            # Only deal with relative imports inside config files
            level != 0
            and globals is not None
            and (globals.get("__package__", "") or "").startswith(_CFG_PACKAGE_NAME)
        ):
            cur_file = find_relative_file(globals["__file__"], name, level)
            spec = importlib.machinery.ModuleSpec(
                _CFG_PACKAGE_NAME + "." + os.path.basename(cur_file),
                None,
                origin=cur_file,
            )
            module = importlib.util.module_from_spec(spec)
            module.__file__ = cur_file
            with open(cur_file) as f:
                content = f.read()
            exec(compile(content, cur_file, "exec"), module.__dict__)
            for name in fromlist:  # turn imported dict into DictConfig automatically
                val = DictConfig(module.__dict__[name], flags={"allow_objects": True})
                module.__dict__[name] = val
            return module
        return old_import(name, globals, locals, fromlist=fromlist, level=level)

    builtins.__import__ = new_import
    yield new_import
    builtins.__import__ = old_import


def callable_to_str(some_callable: Callable) -> str:
    # Return module and name of a callable (function or class) for OmegaConf.
    return f"{some_callable.__module__}.{some_callable.__qualname__}"


class LazyCall:
    """
    Wrap a callable so that when it's called, the call will not be executed, but
    returns a dict that describes the call. Only supports keyword arguments.
    """

    def __init__(self, target: Callable):
        self.T = target

    def target_str(self):
        return

    def __call__(self, **kwargs):
        # Pop `_target_` if it already exists in kwargs. This happens when the
        # callable target is changed while keeping everything else same.
        _ = kwargs.pop("_target_", None)

        # Put current target first; it reads better in printed/saved output.
        kwargs = {"_target_": callable_to_str(self.T), **kwargs}
        return DictConfig(content=kwargs, flags={"allow_objects": True})


class LazyConfig:
    """
    Provide methods to save, load, and overrides an omegaconf config object
    which may contain definition of lazily-constructed objects.
    """

    @staticmethod
    def load(filename: str | Path) -> DictConfig | ListConfig:
        """
        Load a config file (either Python or YAML).

        Args:
            filename: absolute path or relative path w.r.t. current directory.
        """
        filename = str(filename).replace("/./", "/")
        if os.path.splitext(filename)[1] not in [".py", ".yaml", ".yml"]:
            raise ValueError(f"Config file {filename} has to be a python or yaml file.")

        if filename.endswith(".py"):
            with _patch_import():
                # Record the filename
                module_namespace = {
                    "__file__": filename,
                    "__package__": _CFG_PACKAGE_NAME + "." + os.path.basename(filename),
                }
                with open(filename) as f:
                    content = f.read()
                # Compile first with filename to make filename appear in stacktrace
                exec(compile(content, filename, "exec"), module_namespace)

            # Collect final objects in config:
            ret = OmegaConf.create(flags={"allow_objects": True})
            # Store config path for template detection
            ret._config_path = filename

            for name, value in module_namespace.items():
                # Ignore "private" variables (starting with underscores).
                if name.startswith("_"):
                    continue

                if isinstance(value, (DictConfig | dict)):
                    value = DictConfig(value, flags={"allow_objects": True})
                    ret[name] = value
                elif isinstance(value, (ListConfig | list)):
                    value = ListConfig(value, flags={"allow_objects": True})
                    ret[name] = value
                elif isinstance(value, (str, int, float, bool)):
                    # Also save simple scalar values
                    ret[name] = value
        else:
            with open(filename) as f:
                obj = yaml.unsafe_load(f)
            ret = OmegaConf.create(obj, flags={"allow_objects": True})
            # Store config path for template detection
            ret._config_path = filename

        return ret

    @staticmethod
    def save(cfg: DictConfig, filename: str) -> None:
        """Save a config object as YAML file. (same as :meth:`OmegaConf.save`)."""
        OmegaConf.save(cfg, filename, resolve=False)

    @staticmethod
    def apply_overrides(
        cfg: DictConfig, overrides: list[str]
    ) -> DictConfig | ListConfig:
        """
        Return a new config by applying overrides (provided as dotlist). See
        https://hydra.cc/docs/advanced/override_grammar/basic/ for dotlist syntax.
        """
        # Preserve _config_path during merge
        config_path = getattr(cfg, "_config_path", None)
        result = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        if config_path is not None:
            result._config_path = config_path
        return result


class LazyFactory:
    """
    Provides a clean interface to easily construct essential objects from input
    lazy configs (omegaconf): dataloader, model.
    """

    @staticmethod
    def build_selfclean_audio(cfg: DictConfig):
        # For special evaluation datasets (like CSEM), do not override issues_to_detect.
        # Allow SelfCleanAudio default to include all three issue types when None.
        from SelfClean.selfclean.cleaner.issue_manager import IssueTypes

        if not (
            hasattr(cfg, "EVAL_DATASET") and str(cfg.EVAL_DATASET).lower() == "csem"
        ):
            template_type = _detect_template_type(cfg)

            # Map template types to issue types
            issue_mapping = {
                TemplateType.DUPLICATES: [IssueTypes.NEAR_DUPLICATES],
                TemplateType.LABEL_ERRORS: [IssueTypes.LABEL_ERRORS],
                TemplateType.OFF_TOPIC: [IssueTypes.OFF_TOPIC_SAMPLES],
            }

            cfg.selfclean_audio.issues_to_detect = issue_mapping[template_type]
        return instantiate(cfg.selfclean_audio)

    @staticmethod
    def build_dataloader(cfg: DictConfig):
        # Special-case: evaluation on CSEM Membrane Pumps
        if hasattr(cfg, "EVAL_DATASET") and str(cfg.EVAL_DATASET).lower() == "csem":
            # Expect CSEM_ROOT and dataloader params
            if not hasattr(cfg, "CSEM_ROOT"):
                raise ValueError("Config must define CSEM_ROOT for CSEM dataset")

            dataset_cfg = OmegaConf.create(
                {
                    "_target_": "selfclean_audio.datasets.csem.CSEMMembranePumps",
                    "root": cfg.CSEM_ROOT,
                    "convert_mono": True,
                    "sample_rate": DEFAULT_SAMPLE_RATE,
                    "target_duration_sec": DEFAULT_TARGET_DURATION_SEC,
                }
            )

            dataset = instantiate(dataset_cfg)

            # Validate dataloader config
            if not hasattr(cfg, "dataloader"):
                raise ValueError("Config must define dataloader parameters")
            for k in ["num_workers", "batch_size", "drop_last", "pin_memory"]:
                if not hasattr(cfg.dataloader, k):
                    raise ValueError(f"Config dataloader must define {k}")

            return DataLoader(
                dataset,
                num_workers=cfg.dataloader.num_workers,
                batch_size=cfg.dataloader.batch_size,
                drop_last=cfg.dataloader.drop_last,
                pin_memory=cfg.dataloader.pin_memory,
            )

        # Special-case: evaluation on GTZAN with known issues
        if hasattr(cfg, "EVAL_DATASET") and str(cfg.EVAL_DATASET).lower() == "gtzan":
            validate_gtzan_config(cfg)

            # Optional ground-truth files (duplicates and prep with mislabels)
            gt_file = getattr(cfg, "GTZAN_GT_FILE", None)
            gt_prep = getattr(cfg, "GTZAN_PREP_FILE", None)

            # Build GTZAN dataset
            dataset_cfg = OmegaConf.create(
                {
                    "_target_": "selfclean_audio.datasets.gtzan.GTZANKnownIssuesDataset",
                    "root": cfg.GTZAN_ROOT,
                    "issue_type": cfg.ISSUE_TYPE,
                    "gt_duplicates_file": gt_file,
                    "gt_prep_file": gt_prep,
                    "convert_mono": True,
                    "sample_rate": DEFAULT_SAMPLE_RATE,
                    "target_duration_sec": DEFAULT_TARGET_DURATION_SEC,
                }
            )

            dataset = instantiate(dataset_cfg)

            # Validate dataloader config
            if not hasattr(cfg, "dataloader"):
                raise ValueError("Config must define dataloader parameters")
            for k in ["num_workers", "batch_size", "drop_last", "pin_memory"]:
                if not hasattr(cfg.dataloader, k):
                    raise ValueError(f"Config dataloader must define {k}")

            return DataLoader(
                dataset,
                num_workers=cfg.dataloader.num_workers,
                batch_size=cfg.dataloader.batch_size,
                drop_last=cfg.dataloader.drop_last,
                pin_memory=cfg.dataloader.pin_memory,
            )

        # Comprehensive validation
        validate_full_config(cfg)

        # Get seed from config
        seed = get_seed_from_config(cfg)

        esc50_root = cfg.ESC50_ROOT
        esc50_meta = cfg.ESC50_META
        noise_root = cfg.NOISE_ROOT

        issue_type = cfg.ISSUE_TYPE
        frac_error = cfg.FRAC_ERROR

        # Base dataset configuration (ESC50)
        base_dataset_cfg = OmegaConf.create(
            {
                "_target_": "selfclean_audio.datasets.esc50.ESC50",
                "root": esc50_root,
                "dataframe": esc50_meta,
                "convert_mono": True,
                "resample": DEFAULT_SAMPLE_RATE,
            }
        )

        # Build appropriate dataset config based on issue type
        if "duplicates" in issue_type:
            duplicate_strategy = validate_duplicate_strategy(issue_type)

            dataset_cfg = OmegaConf.create(
                {
                    "_target_": "selfclean_audio.datasets.duplicate_dataset.DuplicateDataset",
                    "dataset": base_dataset_cfg,
                    "frac_error": frac_error,
                    "duplicate_strategy": duplicate_strategy,
                    "noise_level": 0.02,
                    "crop_ratio_range": [0.1, 0.25],
                    "random_state": seed,
                    "name": f"DuplicateDataset_{issue_type}",
                    "save_to_temp": True,
                }
            )

        elif issue_type == "label_errors":
            dataset_cfg = OmegaConf.create(
                {
                    "_target_": "selfclean_audio.datasets.label_error_dataset.LabelErrorDataset",
                    "dataset": base_dataset_cfg,
                    "frac_error": frac_error,
                    "change_for_every_label": False,
                    "random_state": seed,
                    "name": "LabelErrorDataset",
                }
            )

        elif "off_topic" in issue_type:
            contamination_strategy = validate_off_topic_strategy(issue_type)

            # External contamination dataset for off_topic_external and off_topic_combined
            contamination_dataset_cfg = None
            if issue_type in ["off_topic_external", "off_topic_combined"]:
                contamination_dataset_cfg = OmegaConf.create(
                    {
                        "_target_": "selfclean_audio.datasets.folder.FolderAudioDataset",
                        "root": noise_root,
                        "convert_mono": True,
                        "sample_rate": DEFAULT_SAMPLE_RATE,
                    }
                )

            dataset_cfg = OmegaConf.create(
                {
                    "_target_": "selfclean_audio.datasets.off_topic_dataset.OffTopicDataset",
                    "dataset": base_dataset_cfg,
                    "contamination_dataset": contamination_dataset_cfg,
                    "frac_error": frac_error,
                    "contamination_strategy": contamination_strategy,
                    "noise_level": 0.3,
                    "random_state": seed,
                    "name": f"OffTopicDataset_{issue_type}",
                }
            )

        else:
            from .validation import validate_issue_type

            validate_issue_type(
                issue_type
            )  # This will raise ValidationError with proper message

        dataset = instantiate(dataset_cfg)

        return DataLoader(
            dataset,
            num_workers=cfg.dataloader.num_workers,
            batch_size=cfg.dataloader.batch_size,
            drop_last=cfg.dataloader.drop_last,
            pin_memory=cfg.dataloader.pin_memory,
        )
