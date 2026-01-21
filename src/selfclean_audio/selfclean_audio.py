# Copyright (c) Lucerne University of Applied Sciences and Arts.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import tempfile
from collections.abc import Sized
from enum import Enum
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from selfclean_audio.datasets import FolderAudioDataset, NoisyDataset
from selfclean_audio.ssl_adapt import LoraAdaptConfig, adapt_model_with_lora

from .constants import (
    DEFAULT_FIGURE_SIZE,
    DEFAULT_HISTOGRAM_ALPHA,
    DEFAULT_HISTOGRAM_BINS,
    DEFAULT_TSNE_PERPLEXITY_MAX,
    DEFAULT_TSNE_PERPLEXITY_MIN,
    TEMPORAL_FEATURE_NAMES,
)

ARR_TYPE = Union[np.ndarray, np.memmap]

__all__ = [
    "PretrainingSSL",
    "SelfCleanAudio",
    "create_memmap",
    "create_memmap_path",
    "embed_dataset",
    "extract_temporal_stats_batch",
]


class PretrainingSSL(Enum):
    """Enum of supported self-supervised learning pretraining models."""

    EAT_BASE_PRETRAIN = "eat-base_pretrain"
    EAT_BASE_FINETUNE = "eat-base_finetune"
    EAT_LARGE_PRETRAIN = "eat-large_pretrain"
    EAT_LARGE_FINETUNE = "eat-large_finetune"
    BEATS = "beats"
    CAVMAE = "cav-mae"
    M2D = "m2d"
    CLMR = "clmr"


class SelfCleanAudio:
    """
    Main class to clean audio datasets using pretrained SSL models and
    distance-based cleaner.
    """

    def __init__(
        self,
        distance_function_path: str = "sklearn.metrics.pairwise.",
        distance_function_name: str = "cosine_similarity",
        chunk_size: int = 100,
        precision_type_distance: type = np.float32,
        memmap: bool = True,
        memmap_path: Path | str | None = None,
        plot_distribution: bool = False,
        plot_top_N: int | None = None,
        output_path: str | None = None,
        figsize: tuple = DEFAULT_FIGURE_SIZE,
        pretraining_ssl: PretrainingSSL = PretrainingSSL.BEATS,
        model_path: str | None = None,
        off_topic_method: str = "lad",
        off_topic_params: dict | None = None,
        near_duplicate_method: str = "embedding_distance",
        near_duplicate_params: dict | None = None,
        label_error_method: str = "intra_extra_distance",
        label_error_params: dict | None = None,
        issues_to_detect: list | None = None,
        random_seed: int = 42,
        device: torch.device | str = "cuda",
        # Optional lightweight dataset adaptation (LoRA + SimCLR)
        lora_enable: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        adapt_epochs: int = 0,
        adapt_lr: float = 1e-4,
        adapt_weight_decay: float = 0.0,
        adapt_temperature: float = 0.2,
        adapt_projection_dim: int = 256,
        adapt_max_steps: int | None = None,
        adapt_objective: str = "infonce",
        vicreg_sim_coeff: float = 25.0,
        vicreg_var_coeff: float = 25.0,
        vicreg_cov_coeff: float = 1.0,
        # Augmentation controls
        adapt_sample_rate: int | None = None,
        adapt_strong_aug: bool = True,
        adapt_time_shift_max: float = 0.1,
        adapt_add_noise_snr_db: float = 15.0,
        adapt_tempo_min: float = 0.9,
        adapt_tempo_max: float = 1.1,
        adapt_pitch_semitones: float = 2.0,
        adapt_reverb_prob: float = 0.3,
        adapt_eq_prob: float = 0.4,
        adapt_time_mask_prob: float = 0.5,
        adapt_time_mask_max_ratio: float = 0.2,
        # Gradient accumulation for memory efficiency
        gradient_accumulation_steps: int = 1,
        **kwargs,
    ):
        """
        Initialize SelfCleanAudio with model and cleaning parameters.

        Args:
            distance_function_path (str): Module path for distance function.
            distance_function_name (str): Distance function name.
            chunk_size (int): Size of chunks to process.
            precision_type_distance (type): Precision for distance calculation.
            memmap (bool): Use memory-mapped arrays for embeddings.
            memmap_path (Path|str|None): Path for memmap storage.
            plot_distribution (bool): Whether to plot distance distribution.
            plot_top_N (int|None): Top N to plot.
            output_path (str|None): Path for outputs.
            figsize (tuple): Figure size for plots.
            pretraining_ssl (PretrainingSSL): SSL pretraining model enum.
            model_path (str|None): Path to pretrained SSL model. If None, will
                try environment variable ``SELFCLEAN_AUDIO_MODEL_PATH``.
            off_topic_method (str): Off-topic detection method ("lad", "quantile", "isolation_forest", "cleanlab").
            off_topic_params (dict|None): Parameters for the off-topic detection method.
            near_duplicate_method (str): Near duplicate detection method ("embedding_distance", "cleanlab", "dejavu").
            near_duplicate_params (dict|None): Parameters for the near duplicate detection method.
            label_error_method (str): Label error detection method ("intra_extra_distance", "cleanlab").
            label_error_params (dict|None): Parameters for the label error detection method.
            random_seed (int): Random seed for reproducibility.
            device (torch.device|str): Device for model inference.
            **kwargs: Additional arguments.
        """
        # Accept both enum and string for pretraining_ssl for robustness in configs/tests
        if isinstance(pretraining_ssl, str):
            # Try to match by enum value first (e.g., "beats"), then by name (case-insensitive)
            matched = None
            for e in PretrainingSSL:
                if (
                    pretraining_ssl == e.value
                    or pretraining_ssl.lower() == e.name.lower()
                ):
                    matched = e
                    break
            if matched is None:
                raise ValueError(
                    f"Unknown pretraining_ssl '{pretraining_ssl}'. Must be one of: "
                    + ", ".join([f"{e.name} ({e.value})" for e in PretrainingSSL])
                )
            pretraining_ssl = matched
        # Defer heavy SelfClean imports to runtime to keep module import light
        from SelfClean.selfclean.cleaner.selfclean_cleaner import SelfCleanCleaner
        from SelfClean.selfclean.core.src.pkg import Embedder
        from SelfClean.selfclean.core.src.utils.utils import fix_random_seeds

        fix_random_seeds(seed=random_seed)
        self.memmap = memmap
        self.memmap_path = memmap_path
        self.cleaner = SelfCleanCleaner(
            distance_function_path=distance_function_path,
            distance_function_name=distance_function_name,
            chunk_size=chunk_size,
            precision_type_distance=precision_type_distance,
            memmap=memmap,
            memmap_path=memmap_path,
            plot_distribution=plot_distribution,
            plot_top_N=plot_top_N,
            output_path=output_path,
            figsize=figsize,
        )
        # Preserve requested methods/params for logging and downstream inspection.
        self.cleaner.near_duplicate_method = near_duplicate_method
        self.cleaner.near_duplicate_params = near_duplicate_params
        self.cleaner.off_topic_method = off_topic_method
        self.cleaner.off_topic_params = off_topic_params
        self.cleaner.label_error_method = label_error_method
        self.cleaner.label_error_params = label_error_params
        # Require explicit model path - no fallbacks
        if model_path is None:
            raise ValueError(
                "model_path must be explicitly provided in config. "
                "Set `selfclean_audio.model_path` in the config template."
            )

        self.model = Embedder.load_pretrained(
            ssl=pretraining_ssl.value, model_path=model_path, **kwargs
        ).to(device)

        self.device = device
        self.workdir = kwargs.get("workdir", "./outputs/")
        self.issues_to_detect = issues_to_detect
        self._lora_cfg = LoraAdaptConfig(
            enable=lora_enable,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            epochs=int(adapt_epochs) if adapt_epochs is not None else 0,
            lr=adapt_lr,
            weight_decay=adapt_weight_decay,
            temperature=adapt_temperature,
            projection_dim=adapt_projection_dim,
            max_steps=adapt_max_steps,
            objective=adapt_objective,
            vicreg_sim_coeff=vicreg_sim_coeff,
            vicreg_var_coeff=vicreg_var_coeff,
            vicreg_cov_coeff=vicreg_cov_coeff,
            sample_rate=(adapt_sample_rate if adapt_sample_rate else 16000),
            strong_aug=adapt_strong_aug,
            time_shift_max=adapt_time_shift_max,
            add_noise_snr_db=adapt_add_noise_snr_db,
            tempo_min=adapt_tempo_min,
            tempo_max=adapt_tempo_max,
            pitch_semitones=adapt_pitch_semitones,
            reverb_prob=adapt_reverb_prob,
            eq_prob=adapt_eq_prob,
            time_mask_prob=adapt_time_mask_prob,
            time_mask_max_ratio=adapt_time_mask_max_ratio,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

    def run_on_dataloader(
        self,
        dataloader: torch.utils.data.DataLoader,
        issues_to_detect: list | None = None,
        apply_l2_norm: bool = False,
    ):
        """
        Detect issues in dataset by running the cleaner on a dataloader.

        Args:
            dataloader (DataLoader): PyTorch DataLoader with audio data.
            issues_to_detect (list[IssueTypes]|None): Issues to detect.
            apply_l2_norm (bool): Whether to L2 normalize embeddings.

        Returns:
            np.ndarray: Predicted issues mask or results.
        """
        # Local import to avoid module import side-effects when not used
        from SelfClean.selfclean.cleaner.issue_manager import IssueTypes

        if issues_to_detect is None:
            # Use instance variable if set, otherwise default to all issues
            issues_to_detect = self.issues_to_detect or [
                IssueTypes.NEAR_DUPLICATES,
                IssueTypes.OFF_TOPIC_SAMPLES,
                IssueTypes.LABEL_ERRORS,
            ]
        self.issues_to_detect = issues_to_detect
        return self._run(
            dataloader=dataloader,
            apply_l2_norm=apply_l2_norm,
        )

    def calculate_scores(self, issue_manager, noisy_labels, dataset=None):
        # Local import to keep top-level import light
        from SelfClean.selfclean.cleaner.issue_manager import IssueTypes
        from SelfClean.selfclean.core.src.utils.plotting import (
            calculate_scores_from_ranking,
        )

        logger.info("Calculating Scores for Issues")

        if self.issues_to_detect is None:
            logger.warning("No issues to detect specified")
            return issue_manager

        for issue_type in self.issues_to_detect:
            logger.info(
                f"Calculating {issue_type.name.replace('_', ' ').title()} Errors"
            )

            if dataset is None or not hasattr(dataset, "get_errors"):
                raise ValueError(
                    "Cannot calculate scores without dataset.get_errors() providing ground truth"
                )

            if issue_type == IssueTypes.NEAR_DUPLICATES:
                # Handle near duplicates with pair comparison
                pred_duplicate_pairs = issue_manager.issue_dict[issue_type.value][
                    "indices"
                ]

                # Use get_errors method to get ground truth duplicate pairs
                true_duplicate_pairs, _ = dataset.get_errors()

                # Create ranking based on whether predicted pairs are true duplicates
                ranking_target = [
                    1 if (int(x[0]), int(x[1])) in true_duplicate_pairs else 0
                    for x in pred_duplicate_pairs
                ]
            else:
                # Handle off-topic samples and label errors with simple index comparison
                pred_indices = issue_manager.issue_dict[issue_type.value]["indices"]

                error_indicators = dataset.get_errors()
                true_issues = {
                    i for i, is_error in enumerate(error_indicators) if is_error
                }

                ranking_target = [
                    1 if idx in true_issues else 0 for idx in pred_indices
                ]

            # Calculate scores using the correct ground truth ranking
            scores = calculate_scores_from_ranking(ranking_target, show_plots=False)
            issue_manager.issue_dict[f"Scores-{issue_type.value}"] = scores.copy()

            # Persist the full ranking vector for downstream visualizations
            try:
                # Only write if we appear to be inside a run directory
                if hasattr(self, "workdir") and self.workdir:
                    workdir = Path(self.workdir)
                    if (workdir / "config.yaml").exists() or (
                        workdir / "logger.txt"
                    ).exists():
                        workdir.mkdir(parents=True, exist_ok=True)
                        out_path = workdir / f"Ranking-{issue_type.value}.csv"
                        # Save as a simple CSV with one column: target (1=TP at rank, 0=FP)
                        pd.DataFrame({"target": list(ranking_target)}).to_csv(
                            out_path, index=False
                        )
            except Exception:
                # Do not fail the run if persisting rankings is not possible
                pass

        return issue_manager

    def _run(
        self,
        dataloader: torch.utils.data.DataLoader,
        apply_l2_norm: bool = False,
    ):
        """
        Internal method to embed dataset and apply cleaning algorithm.

        Args:
            dataloader (DataLoader): PyTorch DataLoader with audio data.
            apply_l2_norm (bool): Whether to L2 normalize embeddings.

        Returns:
            np.ndarray: Predicted issues mask or results.
        """
        if not self.cleaner.is_fitted:
            dataset = dataloader.dataset
            if isinstance(dataset, (FolderAudioDataset, NoisyDataset)):
                class_labels = [
                    dataset.idx_to_class[i] for i in sorted(dataset.idx_to_class)
                ]
            else:
                class_labels = None
            if self._lora_cfg.enable and self._lora_cfg.epochs > 0:
                if not hasattr(dataset, "sample_rate"):
                    raise ValueError(
                        f"Dataset {type(dataset).__name__} must define sample_rate attribute when LoRA adaptation is enabled"
                    )
                if not isinstance(dataset.sample_rate, int) or dataset.sample_rate <= 0:
                    raise ValueError(
                        f"Dataset sample_rate must be a positive integer, got {dataset.sample_rate}"
                    )
                self._lora_cfg.sample_rate = dataset.sample_rate
                self.model = adapt_model_with_lora(
                    model=self.model,
                    dataloader=dataloader,
                    device=self.device,
                    cfg=self._lora_cfg,
                )
            emb_space, paths, labels, noisy_labels = embed_dataset(
                dataloader=dataloader,
                model=self.model,
                normalize=apply_l2_norm,
                memmap=self.memmap,
                memmap_path=self.memmap_path,
                tqdm_desc="Creating dataset representation",
                device=self.device,
                workdir=self.workdir,
                save_plots=False,
            )

            emb_space = np.asarray(emb_space)
            self.cleaner.fit(
                emb_space=emb_space,
                labels=np.asarray(labels),
                paths=np.asarray(paths),
                dataset=dataset,
                class_labels=class_labels,
            )
            # Ensure issues_to_detect is not None
            from SelfClean.selfclean.cleaner.issue_manager import IssueTypes

            issues_to_detect = self.issues_to_detect or [
                IssueTypes.NEAR_DUPLICATES,
                IssueTypes.OFF_TOPIC_SAMPLES,
                IssueTypes.LABEL_ERRORS,
            ]
            issue_manager = self.cleaner.predict(issues_to_detect=issues_to_detect)
            issue_manager = self.calculate_scores(
                issue_manager=issue_manager,
                noisy_labels=noisy_labels,
                dataset=dataset,
            )

            # Cleanup any temporary resources (e.g., synthetic duplicate files)
            try:
                if hasattr(dataset, "cleanup_temp_dir"):
                    dataset.cleanup_temp_dir()
            except Exception:
                pass

            return issue_manager


def extract_temporal_stats_batch(embeddings: torch.Tensor):
    """
    Extracts statistical features from a [N, T, D] tensor of audio embeddings.

    Args:
        embeddings (torch.Tensor): shape [N, T, D]

    Returns:
        torch.Tensor: shape [N, 8], each row is the feature vector for one sample.
    """
    assert embeddings.ndim == 3, "Input must be shape [N, T, D]"
    N, T, D = embeddings.shape

    # Frame-wise L2 norm: [N, T]
    frame_norms = torch.norm(embeddings, dim=-1)

    # Temporal deltas: [N, T-1, D]
    deltas = embeddings[:, 1:, :] - embeddings[:, :-1, :]
    frame_deltas = torch.norm(deltas, dim=-1)  # [N, T-1]

    # Temporal mean and var over features: [N, D]
    mean_over_time = embeddings.mean(dim=1)  # [N, D]
    var_over_time = embeddings.var(dim=1)  # [N, D]

    # Aggregate stats
    mean_norm = frame_norms.mean(dim=1)  # [N]
    std_norm = frame_norms.std(dim=1)  # [N]

    mean_delta = frame_deltas.mean(dim=1)  # [N]
    std_delta = frame_deltas.std(dim=1)  # [N]

    mean_feat_mean = mean_over_time.mean(dim=1)  # [N]
    mean_feat_std = mean_over_time.std(dim=1)  # [N]

    var_feat_mean = var_over_time.mean(dim=1)  # [N]
    var_feat_std = var_over_time.std(dim=1)  # [N]

    # Stack all features into [N, 8]
    features = torch.stack(
        [
            mean_norm,
            std_norm,
            mean_delta,
            std_delta,
            mean_feat_mean,
            mean_feat_std,
            var_feat_mean,
            var_feat_std,
        ],
        dim=1,
    )

    return features  # shape: [N, 8]


def embed_dataset(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    normalize: bool = False,
    memmap: bool = True,
    memmap_path: Path | str | None = None,
    tqdm_desc: str | None = None,
    device: torch.device | str = "cpu",
    workdir: str = "./outputs",
    save_plots: bool = False,
) -> tuple[ARR_TYPE, np.ndarray, torch.Tensor, torch.Tensor]:
    """
    Compute embeddings for all samples in a dataloader using a model.

    Args:
        dataloader (DataLoader): Dataset loader.
        model (nn.Module): Pretrained model for embeddings.
        normalize (bool): Normalize embeddings if True.
        memmap (bool): Use memory-mapped storage.
        memmap_path (Path|str|None): Path for memory map.
        tqdm_desc (str|None): Description for progress bar.
        device (torch.device|str): Device for computation.

    Returns:
        tuple: (embedding array, array of file paths, tensor of labels, tensor of noisy_labels)
    """
    if isinstance(dataloader.dataset, Sized):
        dataset_len = len(dataloader.dataset)
    else:
        logger.error("The dataset does not have attribute __len__")
        dataset_len = 0

    labels = []
    noisy_labels = []
    paths = []
    temporal_list = [] if save_plots else None

    _, emb_dim = _infer_shapes(dataloader, model, device)

    emb_space = _initialize_storage(
        memmap=memmap,
        memmap_path=memmap_path,
        dataset_len=dataset_len,
        emb_dim=emb_dim,
    )

    # Ensure workdir exists for saving plots
    try:
        os.makedirs(workdir, exist_ok=True)
    except Exception:
        pass

    iterator = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=tqdm_desc,
        position=0,
        leave=True,
    )

    write_ptr = 0
    with torch.no_grad():
        for _, batch_tup in iterator:
            batch, path, label, noisy_label = batch_tup

            emb_batch = []
            emb_batch_temporal = [] if save_plots else None
            for waveform in batch:
                emb, emb_t = _compute_embedding(
                    model=model,
                    batch=waveform,
                    normalize=normalize,
                    device=device,
                )
                emb_batch.append(emb)
                if save_plots and emb_batch_temporal is not None:
                    emb_batch_temporal.append(emb_t)

            emb_batch = torch.concat(emb_batch, dim=0)
            # Append temporal embeddings only when the accumulator exists
            if temporal_list is not None and emb_batch_temporal is not None:
                # Keep per-sample temporal embeddings as [T, D]
                for emb_t in emb_batch_temporal:
                    temporal_list.append(emb_t.squeeze(0).cpu())
            n = emb_batch.shape[0]
            emb_np = emb_batch.detach().cpu().numpy()
            emb_space[write_ptr : write_ptr + n] = emb_np
            write_ptr += n

            if isinstance(emb_space, np.memmap):
                emb_space.flush()

            labels.append(label.cpu())
            noisy_labels.append(noisy_label.cpu())

            paths.extend(path)

    labels = torch.concat(labels).cpu()
    noisy_labels = torch.concat(noisy_labels).cpu()

    if save_plots:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.manifold import TSNE

        # Column names for the 8 stats
        feature_names = TEMPORAL_FEATURE_NAMES

        emb_temporal = None
        if temporal_list is not None and len(temporal_list) > 0:
            try:
                emb_temporal = torch.stack(temporal_list, dim=0)
                emb_temporal = extract_temporal_stats_batch(emb_temporal)
            except Exception:
                emb_temporal = None

        if emb_temporal is not None:
            # Create DataFrame
            df = pd.DataFrame(emb_temporal, columns=feature_names)
            df["label"] = ["clean" if lbl == 0 else "noisy" for lbl in noisy_labels]
        else:
            df = None

        plt.figure(figsize=(16, 10))
        bins = DEFAULT_HISTOGRAM_BINS
        alpha = DEFAULT_HISTOGRAM_ALPHA

        if df is not None:
            for i, feat in enumerate(feature_names):
                plt.subplot(2, 4, i + 1)
                sns.histplot(
                    df[df["label"] == "clean"][feat],
                    bins=bins,
                    color="blue",
                    label="clean",
                    stat="density",
                    kde=False,
                    alpha=alpha,
                )
                sns.histplot(
                    df[df["label"] == "noisy"][feat],
                    bins=bins,
                    color="red",
                    label="noisy",
                    stat="density",
                    kde=False,
                    alpha=alpha,
                )
                plt.title(feat)
                plt.legend()
                plt.tight_layout()

            plt.suptitle(
                "Histograms of Temporal Stats (Clean vs Noisy)", fontsize=16, y=1.02
            )
            plt.savefig(os.path.join(workdir, "Figure1.png"))

        # TSNE plot
        if len(emb_space) >= 2:
            perplexity = float(
                min(
                    max(len(emb_space) - 1, DEFAULT_TSNE_PERPLEXITY_MIN),
                    DEFAULT_TSNE_PERPLEXITY_MAX,
                )
            )
            X_embedded = TSNE(n_components=2, perplexity=perplexity).fit_transform(
                emb_space
            )

            x_embed = X_embedded[:, 0]
            y_embed = X_embedded[:, 1]

            plt.figure(figsize=(10, 10))
            scatter = plt.scatter(x_embed, y_embed, c=noisy_labels, alpha=0.8)
            plt.colorbar(scatter, label="Class Label")
            plt.title("t-SNE of Embeddings")
            plt.xlabel("TSNE-1")
            plt.ylabel("TSNE-2")
            plt.grid(True)
            plt.savefig(os.path.join(workdir, "Figure2.png"))

    paths = np.array(paths)

    return emb_space, paths, labels, noisy_labels


def _infer_shapes(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device | str,
):
    """
    Infer sample and embedding dimensions from a sample batch.

    Args:
        dataloader (DataLoader): Dataset loader.
        model (nn.Module): Model to generate embeddings.
        device (torch.device|str): Device for computation.

    Returns:
        tuple: (sample batch tensor, embedding dimension)
    """

    sample_batch = dataloader.dataset[0][0]  # [C, T] Mono=1
    if model is not None:
        sample_batch = sample_batch.to(device)

    if model is None:
        emb = sample_batch
    else:
        emb, _ = model.extract_features(sample_batch)  # type:ignore

    emb_dim = emb.squeeze().shape[-1]  # [Seq_len, Embed_dim]

    return sample_batch, emb_dim


def _initialize_storage(memmap, memmap_path, dataset_len, emb_dim):
    """
    Initialize numpy array or memmap to store embeddings.

    Args:
        memmap (bool): Use memmap if True.
        memmap_path (Path): Path to store memmap.
        dataset_len (int): Number of dataset samples.
        emb_dim (int): Embedding dimension.

    Returns:
        np.ndarray or np.memmap: Storage for embeddings.
    """
    if memmap:
        memmap_path = create_memmap_path(memmap_path)
        emb_space = create_memmap(
            memmap_path, "embedding_space.dat", dataset_len, emb_dim
        )
    else:
        emb_space = np.zeros((dataset_len, emb_dim), dtype=np.float32)

    return emb_space


def _compute_embedding(
    model: torch.nn.Module,
    batch: torch.Tensor,
    normalize: bool,
    device: torch.device | str,
):
    """
    Compute embedding for a single batch with optional normalization.

    Args:
        model (nn.Module): Model to extract features.
        batch (Tensor): Input batch tensor.
        normalize (bool): Apply L2 normalization if True.
        device (torch.device|str): Device for computation.

    Returns:
        tuple: (main_embedding, temporal_embedding) where main_embedding is used
               for cleaning and temporal_embedding contains temporal features.
    """
    if model is not None:
        batch = batch.to(device)
        emb, emb_t = model.extract_features(batch)  # type:ignore
    else:
        # If no model provided, use the batch itself as embedding
        emb = batch
        emb_t = batch

    if normalize:
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)

    return emb, emb_t


def create_memmap(memmap_path: Path, memmap_file_name: str, len_dataset: int, *dims):
    """
    Create a memory-mapped numpy array for storing embeddings.

    Args:
        memmap_path (Path): Directory to store memmap file.
        memmap_file_name (str): Filename for memmap.
        len_dataset (int): Number of samples.
        *dims: Dimensions of each embedding.

    Returns:
        np.memmap: Memory-mapped numpy array.
    """
    memmap_file = memmap_path / memmap_file_name
    if memmap_file.exists():
        memmap_file.unlink()
    memmap = np.memmap(
        str(memmap_file),
        dtype=np.float32,
        mode="w+",
        shape=(len_dataset, *dims),
    )
    return memmap


def create_memmap_path(memmap_path: str | Path | None) -> Path:
    """
    Ensure memmap directory exists or create a temporary directory.

    Args:
        memmap_path (str|Path|None): Desired memmap directory or None.

    Returns:
        Path: Path to memmap directory.
    """
    if memmap_path is None:
        memmap_path = Path(tempfile.mkdtemp())
    else:
        memmap_path = Path(memmap_path)
        memmap_path.mkdir(parents=True, exist_ok=True)
    return memmap_path
