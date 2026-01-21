# Copyright (c) Lucerne University of Applied Sciences and Arts.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""Centralized validation functions for configuration and parameters."""

from omegaconf import DictConfig

from .constants import (
    DUPLICATE_STRATEGY_MAP,
    OFF_TOPIC_STRATEGY_MAP,
    REQUIRED_BASE_PARAMS,
    REQUIRED_DATALOADER_PARAMS,
    REQUIRED_DATASET_PATHS,
    REQUIRED_GTZAN_PARAMS,
)


class ValidationError(ValueError):
    """Custom exception for validation errors."""

    pass


def validate_required_attributes(
    obj, required_attrs: list[str], context: str = ""
) -> None:
    """
    Validate that an object has all required attributes.

    Args:
        obj: Object to validate
        required_attrs: List of required attribute names
        context: Context string for better error messages

    Raises:
        ValidationError: If any required attribute is missing
    """
    missing_attrs = [attr for attr in required_attrs if not hasattr(obj, attr)]
    if missing_attrs:
        context_str = f" for {context}" if context else ""
        raise ValidationError(
            f"Missing required attributes{context_str}: {', '.join(missing_attrs)}"
        )


def validate_base_config(cfg: DictConfig) -> None:
    """Validate basic configuration parameters."""
    validate_required_attributes(cfg, REQUIRED_BASE_PARAMS, "base configuration")


def validate_dataset_paths(cfg: DictConfig) -> None:
    """Validate dataset path parameters."""
    validate_required_attributes(cfg, REQUIRED_DATASET_PATHS, "dataset paths")


def validate_gtzan_config(cfg: DictConfig) -> None:
    """Validate GTZAN-specific configuration parameters."""
    validate_required_attributes(cfg, REQUIRED_GTZAN_PARAMS, "GTZAN configuration")


def validate_dataloader_config(cfg: DictConfig) -> None:
    """Validate dataloader configuration parameters."""
    if not hasattr(cfg, "dataloader"):
        raise ValidationError("Config must define dataloader parameters")

    validate_required_attributes(
        cfg.dataloader, REQUIRED_DATALOADER_PARAMS, "dataloader configuration"
    )


def validate_issue_type(issue_type: str) -> None:
    """
    Validate that the issue type is supported.

    Args:
        issue_type: Issue type to validate

    Raises:
        ValidationError: If issue type is not supported
    """
    all_supported_types = (
        list(DUPLICATE_STRATEGY_MAP.keys())
        + ["label_errors"]
        + list(OFF_TOPIC_STRATEGY_MAP.keys())
    )

    if issue_type not in all_supported_types:
        raise ValidationError(
            f"Unknown ISSUE_TYPE: {issue_type}. Must be one of: {all_supported_types}"
        )


def validate_duplicate_strategy(issue_type: str) -> str:
    """
    Validate and get duplicate strategy from issue type.

    Args:
        issue_type: Issue type for duplicate detection

    Returns:
        str: Corresponding duplicate strategy

    Raises:
        ValidationError: If issue type is not a valid duplicate type
    """
    if issue_type not in DUPLICATE_STRATEGY_MAP:
        raise ValidationError(
            f"Unknown duplicate ISSUE_TYPE: {issue_type}. "
            f"Must be one of: {list(DUPLICATE_STRATEGY_MAP.keys())}"
        )
    return DUPLICATE_STRATEGY_MAP[issue_type]


def validate_off_topic_strategy(issue_type: str) -> str:
    """
    Validate and get off-topic strategy from issue type.

    Args:
        issue_type: Issue type for off-topic detection

    Returns:
        str: Corresponding off-topic strategy

    Raises:
        ValidationError: If issue type is not a valid off-topic type
    """
    if issue_type not in OFF_TOPIC_STRATEGY_MAP:
        raise ValidationError(
            f"Unknown off-topic ISSUE_TYPE: {issue_type}. "
            f"Must be one of: {list(OFF_TOPIC_STRATEGY_MAP.keys())}"
        )
    return OFF_TOPIC_STRATEGY_MAP[issue_type]


def get_seed_from_config(cfg: DictConfig) -> int:
    """
    Extract seed from various possible locations in config.

    Args:
        cfg: Configuration object

    Returns:
        int: Seed value

    Raises:
        ValidationError: If no seed is found in any expected location
    """
    seed_locations = [
        ("SEED", lambda c: c.SEED),
        ("params.seed", lambda c: c.params.seed),
        ("selfclean_audio.random_seed", lambda c: c.selfclean_audio.random_seed),
    ]

    for location_name, extractor in seed_locations:
        try:
            if hasattr(cfg, location_name.split(".")[0]):
                seed = extractor(cfg)
                return int(seed)
        except (AttributeError, TypeError):
            continue

    raise ValidationError(
        "Config must define seed parameter as SEED, params.seed, or selfclean_audio.random_seed"
    )


def validate_full_config(cfg: DictConfig) -> None:
    """
    Perform comprehensive configuration validation.

    Args:
        cfg: Configuration object to validate

    Raises:
        ValidationError: If any validation fails
    """
    # Basic validation
    validate_base_config(cfg)

    # Issue type validation
    validate_issue_type(cfg.ISSUE_TYPE)

    # Seed validation
    get_seed_from_config(cfg)

    # Path validation (only for non-GTZAN datasets)
    eval_dataset = getattr(cfg, "EVAL_DATASET", "").lower()
    if eval_dataset != "gtzan":
        validate_dataset_paths(cfg)
    else:
        validate_gtzan_config(cfg)

    # Dataloader validation
    validate_dataloader_config(cfg)
