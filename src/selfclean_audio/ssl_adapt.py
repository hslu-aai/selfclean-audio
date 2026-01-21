# Copyright (c) Lucerne University of Applied Sciences and Arts.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torchaudio
from loguru import logger


@dataclass
class LoraAdaptConfig:
    enable: bool = False
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "fc1",
        "fc2",
    )
    epochs: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    temperature: float = 0.2
    projection_dim: int = 256
    max_steps: int | None = None  # limit updates for quick runs
    # objective: "infonce" or "vicreg"
    objective: str = "infonce"
    # VICReg coefficients
    vicreg_sim_coeff: float = 25.0
    vicreg_var_coeff: float = 25.0
    vicreg_cov_coeff: float = 1.0
    # Augmentation controls
    sample_rate: int = 16000
    strong_aug: bool = True
    time_shift_max: float = 0.1
    add_noise_snr_db: float = 15.0
    tempo_min: float = 0.9
    tempo_max: float = 1.1
    pitch_semitones: float = 2.0
    reverb_prob: float = 0.3
    eq_prob: float = 0.4
    time_mask_prob: float = 0.5
    time_mask_max_ratio: float = 0.2
    # Gradient accumulation for memory efficiency
    gradient_accumulation_steps: int = 1


def _unwrap_quant_noise_modules(model: nn.Module) -> None:
    """Best-effort unwrap quant_noise wrappers so PEFT can see nn.Linear modules.

    Some backbones (e.g., BEATs) wrap linear projections with a quant_noise
    module that stores the actual nn.Linear under attribute `module`. PEFT's
    LoRA injection targets nn.Linear; unwrapping ensures adapters attach.
    """
    attrs = ("q_proj", "k_proj", "v_proj", "out_proj")
    for module in model.modules():
        for attr in attrs:
            if hasattr(module, attr):
                sub = getattr(module, attr)
                # Unwrap single level `module` indirection if present
                try:
                    if hasattr(sub, "module") and isinstance(sub.module, nn.Linear):
                        setattr(module, attr, sub.module)
                except Exception:
                    pass


def _try_enable_lora(model: nn.Module, cfg: LoraAdaptConfig) -> nn.Module | None:
    """Attach LoRA adapters to the model using PEFT when available.

    Returns the PEFT-wrapped model if successful; otherwise None.
    """
    try:
        from peft import LoraConfig, get_peft_model

        # Unwrap quant_noise so adapters can target nn.Linear modules
        _unwrap_quant_noise_modules(model)

        lcfg = LoraConfig(
            r=cfg.r,
            lora_alpha=cfg.alpha,
            lora_dropout=cfg.dropout,
            target_modules=list(cfg.target_modules),
            inference_mode=False,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        peft_model = get_peft_model(model, lcfg)
        peft_model.print_trainable_parameters()
        try:
            if hasattr(peft_model, "set_adapter"):
                peft_model.set_adapter("default")
            if hasattr(peft_model, "enable_adapter_layers"):
                peft_model.enable_adapter_layers()
        except Exception:
            pass
        # Introspect injected modules for debugging
        try:
            injected = []
            for name, mod in peft_model.named_modules():
                if hasattr(mod, "lora_A") or hasattr(mod, "lora_B"):
                    injected.append((name, mod.__class__.__name__))
            if injected:
                preview = ", ".join([f"{n}:{c}" for n, c in injected[:6]])
                logger.info(
                    f"LoRA injection summary: {len(injected)} modules with lora_* params. Examples: {preview}"
                )
                # Debug: analyze which types of modules got LoRA
                module_types = {}
                for name, cls in injected:
                    # Extract module type (e.g., 'k_proj', 'v_proj', 'fc1')
                    module_type = name.split(".")[-1] if "." in name else name
                    module_types[module_type] = module_types.get(module_type, 0) + 1
                logger.info(f"LoRA module distribution: {dict(module_types)}")
            else:
                logger.warning(
                    "PEFT did not expose any modules with lora_* attributes."
                )
        except Exception:
            pass
        return peft_model
    except Exception as e:
        logger.warning(
            f"PEFT not available or failed to attach LoRA adapters: {e}. "
            "Proceeding without LoRA (base model frozen)."
        )
        return None


def _freeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2):
    """Compute SimCLR-style InfoNCE loss over a batch.

    z1, z2: shape [B, D]
    """
    z1 = nn.functional.normalize(z1, dim=-1)
    z2 = nn.functional.normalize(z2, dim=-1)
    B = z1.size(0)
    reps = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = torch.matmul(reps, reps.t()) / temperature  # [2B, 2B]

    # Mask self-similarity
    mask = torch.eye(2 * B, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float("-inf"))

    # Positives: i <-> i+B and i+B <-> i
    targets = torch.arange(B, device=sim.device)
    targets = torch.cat([targets + B, targets], dim=0)

    loss = nn.functional.cross_entropy(sim, targets)
    return loss


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def _vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_coeff: float = 25.0,
    var_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    eps: float = 1e-4,
):
    """Loss VICReg: invariance + variance + covariance regularization.

    z1, z2: [B, D]
    """
    # invariance
    sim_loss = nn.functional.mse_loss(z1, z2)

    # variance (per-dimension std close to 1)
    def _var_loss(z: torch.Tensor):
        std = torch.sqrt(z.var(dim=0) + eps)
        return torch.mean(nn.functional.relu(1 - std))

    var_loss = _var_loss(z1) + _var_loss(z2)

    # covariance (decorrelate features)
    def _cov_loss(z: torch.Tensor):
        z = z - z.mean(dim=0)
        n, d = z.shape
        cov = (z.T @ z) / (n - 1)
        off = _off_diagonal(cov)
        return (off.pow(2).sum()) / d

    cov_loss = _cov_loss(z1) + _cov_loss(z2)

    loss = sim_coeff * sim_loss + var_coeff * var_loss + cov_coeff * cov_loss
    return loss


def _random_time_shift(x: torch.Tensor, max_shift: float = 0.1) -> torch.Tensor:
    """Circular time shift by up to max_shift fraction of length (per-sample)."""
    B, T = x.shape[-2], x.shape[-1]
    shift = (torch.rand(B, device=x.device) * 2 * max_shift - max_shift) * T
    shift = shift.long()
    out = torch.zeros_like(x)
    for i in range(B):
        s = shift[i].item()
        out[i] = torch.roll(x[i], shifts=s, dims=-1)
    return out


def _additive_noise(x: torch.Tensor, snr_db: float = 10.0) -> torch.Tensor:
    rms = (x.pow(2).mean(dim=-1, keepdim=True)).sqrt().clamp_min(1e-8)
    noise = torch.randn_like(x)
    rms_n = (noise.pow(2).mean(dim=-1, keepdim=True)).sqrt().clamp_min(1e-8)
    scale = rms / (10 ** (snr_db / 20)) / rms_n
    return x + scale * noise


def _random_crop(x: torch.Tensor, crop_ratio: float = 0.9) -> torch.Tensor:
    """Randomly crop to crop_ratio length and pad back to original size."""
    B, T = x.shape[-2], x.shape[-1]
    new_T = max(1, int(T * crop_ratio))
    start = (torch.rand(B, device=x.device) * (T - new_T + 1)).long()
    out = torch.zeros_like(x)
    for i in range(B):
        seg = x[i, :, start[i] : start[i] + new_T]
        out[i, :, :new_T] = seg
    return out


def _apply_time_mask(x: torch.Tensor, max_ratio: float) -> torch.Tensor:
    """Zero-out a random temporal segment up to max_ratio of length (per-sample)."""
    B, C, T = x.shape
    out = x.clone()
    for i in range(B):
        mask_len = max(1, int(T * random.uniform(0.0, max_ratio)))
        start = random.randint(0, max(0, T - mask_len))
        out[i, :, start : start + mask_len] = 0
    return out


def _maybe_trim_pad(x: torch.Tensor, T: int) -> torch.Tensor:
    if x.size(-1) == T:
        return x
    if x.size(-1) > T:
        return x[..., :T]
    pad = T - x.size(-1)
    return nn.functional.pad(x, (0, pad), mode="constant", value=0)


def _apply_strong_augs(x: torch.Tensor, cfg: LoraAdaptConfig) -> torch.Tensor:
    """Apply a sequence of stronger waveform-level augs using sox effects and biquads.

    Improved to handle batch processing more efficiently and consistently.
    """
    B, C, T = x.shape
    out = x.clone()

    # Pre-generate random decisions for the batch for consistency
    eq_mask = torch.rand(B) < cfg.eq_prob
    tempo_mask = torch.rand(B) < 0.5
    pitch_mask = torch.rand(B) < 0.5
    reverb_mask = torch.rand(B) < cfg.reverb_prob

    for i in range(B):
        xi = out[i]

        # Random EQ (bandreject / lowpass / highpass)
        if eq_mask[i]:
            choice = random.choice(["lowpass", "highpass", "bandreject"])  # noqa: S311
            if choice == "lowpass":
                cutoff = random.uniform(2000, 8000)
                xi = torchaudio.functional.lowpass_biquad(xi, cfg.sample_rate, cutoff)
            elif choice == "highpass":
                cutoff = random.uniform(50, 500)
                xi = torchaudio.functional.highpass_biquad(xi, cfg.sample_rate, cutoff)
            else:
                center = random.uniform(300, 4000)
                xi = torchaudio.functional.bandreject_biquad(
                    xi, cfg.sample_rate, center, Q=0.707
                )

        # Random tempo / pitch using sox (may change length)
        if tempo_mask[i]:
            tempo = random.uniform(cfg.tempo_min, cfg.tempo_max)
            try:
                xi, _ = torchaudio.sox_effects.apply_effects_tensor(
                    xi, cfg.sample_rate, [["tempo", f"{tempo:.3f}"]]
                )
                xi = _maybe_trim_pad(xi, T)
            except RuntimeError:
                # Handle Sox effects failures gracefully
                pass

        if pitch_mask[i] and cfg.pitch_semitones > 0:
            semis = random.uniform(-cfg.pitch_semitones, cfg.pitch_semitones)
            try:
                xi, _ = torchaudio.sox_effects.apply_effects_tensor(
                    xi, cfg.sample_rate, [["pitch", f"{semis:.3f}"]]
                )
                xi = _maybe_trim_pad(xi, T)
            except RuntimeError:
                # Handle Sox effects failures gracefully
                pass

        if reverb_mask[i]:
            try:
                xi, _ = torchaudio.sox_effects.apply_effects_tensor(
                    xi, cfg.sample_rate, [["reverb", "50", "50", "100"]]
                )
                xi = _maybe_trim_pad(xi, T)
            except RuntimeError:
                # Handle Sox effects failures gracefully
                pass

        out[i] = xi

    # Time masking - apply batch-wise with consistent probability
    if random.random() < cfg.time_mask_prob:
        out = _apply_time_mask(out, cfg.time_mask_max_ratio)

    # Always apply some jitter: crop -> shift -> noise
    # Use consistent crop ratio across batch for better contrastive learning
    crop_ratio = random.uniform(0.85, 0.98)
    out = _random_crop(out, crop_ratio=crop_ratio)
    out = _random_time_shift(out, max_shift=cfg.time_shift_max)
    out = _additive_noise(out, snr_db=cfg.add_noise_snr_db)

    return out


def _two_views(
    waveform: torch.Tensor, cfg: LoraAdaptConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create two augmented views of a batch of waveforms.

    waveform: [B, C, T] or [B, T]; will squeeze to [B, 1, T].
    """
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(1)
    B, C, T = waveform.shape

    # Ensure minimum length for feature extraction when strong augmentations are used
    min_length = 400
    if cfg.strong_aug and T < min_length:
        pad_length = min_length - T
        waveform = torch.nn.functional.pad(
            waveform, (0, pad_length), mode="constant", value=0
        )
        T = waveform.size(-1)
        logger.debug(
            f"Padded batch from {T - pad_length} to {T} samples for augmentation"
        )

    x = waveform.reshape(B, C, T)
    if cfg.strong_aug:
        x1 = _apply_strong_augs(x, cfg)
        x2 = _apply_strong_augs(x, cfg)
    else:
        x1 = _additive_noise(_random_time_shift(_random_crop(x)), snr_db=15.0)
        x2 = _additive_noise(_random_time_shift(_random_crop(x)), snr_db=15.0)
    return x1, x2


def _pool_embedding(emb: torch.Tensor) -> torch.Tensor:
    """Mean-pool temporal embeddings if needed: [B, T, D] -> [B, D]."""
    if emb.ndim == 3:
        return emb.mean(dim=1)
    return emb.squeeze()


def adapt_model_with_lora(
    model: nn.Module,
    dataloader,
    device: torch.device | str,
    cfg: LoraAdaptConfig,
):
    """Adapt the model on the target dataset using SimCLR and LoRA adapters.

    The base model is frozen; only LoRA adapters (if enabled) and a small projection
    head are trained. After adaptation, the projection head is discarded and the
    model remains with the adapted LoRA weights active.
    """
    if not cfg.enable:
        return model

    logger.info("Starting LoRA adaptation on target dataset (unsupervised)")
    model.to(device)
    model.train()

    _freeze_all(model)
    peft_model = _try_enable_lora(model, cfg)
    used_lora = peft_model is not None
    if used_lora:
        model = peft_model  # use LoRA-wrapped model

    # Infer embedding dimension with a small forward
    sample = dataloader.dataset[0][0]
    sample = sample.to(device)
    with torch.no_grad():
        emb, _ = model.extract_features(sample)  # type: ignore[attr-defined]
        emb_dim = _pool_embedding(emb).shape[-1]

    proj = ProjectionHead(emb_dim, cfg.projection_dim).to(device)

    # Collect trainable params (LoRA + projection)
    if used_lora:
        # Get LoRA parameters
        lora_params = [p for p in model.parameters() if p.requires_grad]
        logger.debug(f"Collected {len(lora_params)} LoRA parameters")
        # Get projection parameters
        proj_params = list(proj.parameters())
        logger.debug(f"Collected {len(proj_params)} projection parameters")
        # Combine all parameters
        params = lora_params + proj_params
    else:
        params = list(proj.parameters())

    logger.debug(f"Total parameters before validation: {len(params)}")
    logger.debug(f"Parameter types: {[type(p).__name__ for p in params[:5]]}")
    # Filter to only valid parameters
    params = [p for p in params if isinstance(p, nn.Parameter)]
    logger.debug(f"Valid parameters after filtering: {len(params)}")

    if len(params) == 0:
        logger.error("CRITICAL: No valid parameters for optimization!")
        return model

    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Debug parameter collection
    logger.debug(f"Collected params type: {type(params)}")
    logger.debug(
        f"Params length: {len(params) if hasattr(params, '__len__') else 'No length'}"
    )
    if len(params) > 0:
        logger.debug(f"First param type: {type(params[0])}")
        logger.debug(f"First few param types: {[type(p) for p in params[:5]]}")

    # Validation: ensure we have trainable parameters
    try:
        trainable_count = sum(
            1 for p in params if hasattr(p, "requires_grad") and p.requires_grad
        )
        total_params = sum(
            p.numel() for p in params if hasattr(p, "requires_grad") and p.requires_grad
        )
        logger.info(
            f"Adaptation setup: {trainable_count} trainable param groups, {total_params:,} total parameters"
        )

        if trainable_count == 0:
            logger.error(
                "CRITICAL: No trainable parameters found! Adaptation will fail."
            )
            return model
    except Exception as e:
        logger.error(f"Failed to validate parameters: {e}")
        logger.error(f"Params debug: {params[:3] if len(params) > 0 else 'Empty'}")
        return model

    # Verify LoRA integration if enabled
    if used_lora:
        lora_param_count = sum(
            1 for n, p in model.named_parameters() if "lora_" in n and p.requires_grad
        )
        if lora_param_count == 0:
            logger.error(
                "CRITICAL: LoRA enabled but no lora_* parameters found! Check PEFT integration."
            )
            return model
        else:
            logger.info(
                f"LoRA integration verified: {lora_param_count} trainable LoRA parameters"
            )

            # Try to ensure all LoRA modules are properly enabled
            try:
                # Force enable all adapters
                if hasattr(model, "enable_adapter_layers"):
                    model.enable_adapter_layers()
                if hasattr(model, "set_adapter"):
                    model.set_adapter("default")

                # Additional check: ensure LoRA modules are in train mode
                lora_modules_found = 0
                for name, module in model.named_modules():
                    if (
                        "lora_" in name.lower()
                        or hasattr(module, "lora_A")
                        or hasattr(module, "lora_B")
                    ):
                        if hasattr(module, "train"):
                            module.train()
                        lora_modules_found += 1

                if lora_modules_found > 0:
                    logger.info(
                        f"Ensured {lora_modules_found} LoRA modules are in training mode"
                    )

            except Exception as e:
                logger.warning(f"Failed to fully initialize LoRA state: {e}")

    # Sanity-check snapshot: LoRA param norms and pre-training embeddings
    def _lora_named_params(m: nn.Module):
        return [(n, p) for n, p in m.named_parameters() if p.requires_grad]

    lora_before = None
    if used_lora:
        try:
            lora_before = {n: p.detach().clone() for n, p in _lora_named_params(model)}
            if len(lora_before) == 0:
                logger.warning("No trainable LoRA params found after PEFT wrapping.")
        except Exception:
            lora_before = None

    # Helper: robust feature extraction with optional gradients
    def _extract_features_with_optional_grad(x: torch.Tensor, want_grad: bool):
        # BEATs expects [C, T] format
        original_shape = x.shape
        logger.debug(f"Input to feature extraction: {original_shape}")

        # Convert to proper [C, T] format for BEATs
        if x.ndim == 1:
            # [T] -> [1, T] (add channel dimension)
            x = x.unsqueeze(0)
        elif x.ndim == 3:
            # [B, C, T] -> [C, T] (this shouldn't happen in per-sample processing, but handle it)
            if x.size(0) == 1:
                x = x.squeeze(0)  # [1, C, T] -> [C, T]
            else:
                logger.error(
                    f"Unexpected batch size {x.size(0)} in single-sample processing"
                )
                x = x[0]  # Take first sample
        # x should now be [C, T]
        if x.ndim != 2:
            raise ValueError(
                f"After processing, expected 2D tensor [C, T], got {x.ndim}D: {x.shape}"
            )

        # Check minimum length for BEATs (needs at least 400 samples for window_size)
        min_length = 400
        if x.size(-1) < min_length:
            pad_length = min_length - x.size(-1)
            x = torch.nn.functional.pad(x, (0, pad_length), mode="constant", value=0)
            logger.debug(
                f"Padded from {x.size(-1) - pad_length} to {x.size(-1)} samples"
            )
        logger.debug(f"Final shape to BEATs: {x.shape}")

        try:
            return model.extract_features(x, with_grad=want_grad)  # type: ignore[attr-defined]
        except TypeError:
            # BEATs doesn't support with_grad parameter
            try:
                result = model.extract_features(x)  # type: ignore[attr-defined]
                if (
                    want_grad
                    and hasattr(result[0], "requires_grad")
                    and not result[0].requires_grad
                ):
                    logger.warning(
                        "Gradients requested but result has requires_grad=False"
                    )
                return result
            except Exception as e:
                logger.error(
                    f"BEATs failed: input {original_shape} -> {x.shape}, error: {e}"
                )
                logger.error(
                    f"Tensor info: dtype={x.dtype}, device={x.device}, min={x.min():.4f}, max={x.max():.4f}"
                )
                raise

    # Grab a small batch to compare embeddings before/after adaptation
    pre_emb: torch.Tensor | None = None
    try:
        first_batch = next(iter(dataloader))[0].to(device)  # [B, C, T] or [B, T]
        B = min(first_batch.shape[0], 4)
        # Ensure model is in eval mode for consistent baseline embeddings
        was_training = model.training
        model.eval()
        with torch.no_grad():
            # BEATs requires per-sample processing
            zs = []
            for i in range(B):
                # Extract single sample: first_batch[i] should be [C, T] after indexing
                e, _ = _extract_features_with_optional_grad(
                    first_batch[i], want_grad=False
                )
                zs.append(_pool_embedding(e))
            pre_emb = torch.stack(zs, dim=0)
        # Restore original training state
        model.train(was_training)
    except Exception as e:
        logger.warning(f"Failed to compute baseline embeddings: {e}")
        pre_emb = None

    step = 0
    running_loss = 0.0
    loss_history = []

    # Gradient accumulation tracking
    accumulation_step = 0

    # LoRA gradient tracking over time
    lora_gradient_history = {
        "steps": [],
        "lora_A_active": [],
        "lora_B_active": [],
        "lora_A_avg_grad": [],
        "lora_B_avg_grad": [],
        "total_active": [],
        "gradient_ratio": [],  # lora_B / lora_A gradient ratio
    }

    # Ensure model is in training mode for adaptation
    model.train()

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in dataloader:
            waveforms = batch[0].to(device)
            x1, x2 = _two_views(waveforms, cfg)

            # Forward two views through base model in batched manner
            B = x1.size(0)

            # BEATs requires per-sample processing: [C, T] format per call
            # Cannot process batches directly, so iterate through samples
            zs1 = []
            zs2 = []
            for i in range(B):
                # Extract single sample: x1[i] should be [C, T] after indexing
                emb1_i, _ = _extract_features_with_optional_grad(x1[i], want_grad=True)
                emb2_i, _ = _extract_features_with_optional_grad(x2[i], want_grad=True)
                z1_i = proj(_pool_embedding(emb1_i))
                z2_i = proj(_pool_embedding(emb2_i))
                zs1.append(z1_i)
                zs2.append(z2_i)
            z1 = torch.stack(zs1, dim=0)
            z2 = torch.stack(zs2, dim=0)

            if cfg.objective.lower() == "vicreg":
                loss = _vicreg_loss(
                    z1,
                    z2,
                    sim_coeff=cfg.vicreg_sim_coeff,
                    var_coeff=cfg.vicreg_var_coeff,
                    cov_coeff=cfg.vicreg_cov_coeff,
                )
            else:
                loss = _info_nce_loss(z1, z2, cfg.temperature)

            # Scale loss for gradient accumulation
            loss = loss / cfg.gradient_accumulation_steps

            # Zero gradients at start of accumulation cycle
            if accumulation_step == 0:
                opt.zero_grad(set_to_none=True)

            loss.backward()
            # Create a fresh parameter list for gradient clipping to avoid any corruption
            clip_params = []
            if used_lora:
                clip_params.extend([p for p in model.parameters() if p.requires_grad])
            clip_params.extend(proj.parameters())

            # Validate all are nn.Parameter
            clip_params = [p for p in clip_params if isinstance(p, nn.Parameter)]

            try:
                nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
            except Exception as e:
                logger.error(f"Gradient clipping failed even with fresh params: {e}")
                logger.error(
                    f"Fresh clip_params types: {[type(p).__name__ for p in clip_params[:5]]}"
                )
                logger.warning("Skipping gradient clipping - continuing without it")
            # Enhanced diagnostics: grad norms and gradient flow validation
            try:
                if used_lora:
                    # Get only LoRA parameters from the model
                    current_lora_params = [
                        p for p in model.parameters() if p.requires_grad
                    ]
                    lora_grads = [
                        p.grad.detach().norm().item()
                        for p in current_lora_params
                        if p.grad is not None
                    ]
                    mean_lora_grad = (
                        sum(lora_grads) / len(lora_grads) if lora_grads else 0.0
                    )
                    max_lora_grad = max(lora_grads) if lora_grads else 0.0

                    # Detailed gradient flow analysis with time tracking
                    zero_grad_params = []
                    nonzero_grad_params = []
                    null_grad_params = []

                    # Separate tracking for lora_A and lora_B
                    lora_A_active = []
                    lora_B_active = []
                    lora_A_grads = []
                    lora_B_grads = []

                    for n, p in model.named_parameters():
                        if "lora_" in n and p.requires_grad:
                            if p.grad is None:
                                null_grad_params.append(n)
                            elif p.grad.abs().sum() == 0:
                                zero_grad_params.append(n)
                            else:
                                grad_norm = p.grad.abs().sum().item()
                                nonzero_grad_params.append((n, grad_norm))

                                # Track lora_A vs lora_B separately
                                if "lora_A" in n:
                                    lora_A_active.append(n)
                                    lora_A_grads.append(grad_norm)
                                elif "lora_B" in n:
                                    lora_B_active.append(n)
                                    lora_B_grads.append(grad_norm)

                    zero_grad_count = len(zero_grad_params) + len(null_grad_params)
                    total_lora_params = len(current_lora_params)

                    # Update gradient tracking history
                    lora_gradient_history["steps"].append(step)
                    lora_gradient_history["lora_A_active"].append(len(lora_A_active))
                    lora_gradient_history["lora_B_active"].append(len(lora_B_active))
                    lora_gradient_history["lora_A_avg_grad"].append(
                        sum(lora_A_grads) / len(lora_A_grads) if lora_A_grads else 0.0
                    )
                    lora_gradient_history["lora_B_avg_grad"].append(
                        sum(lora_B_grads) / len(lora_B_grads) if lora_B_grads else 0.0
                    )
                    lora_gradient_history["total_active"].append(
                        len(lora_A_active) + len(lora_B_active)
                    )

                    # Calculate gradient ratio (B/A) for tracking balance
                    a_avg = (
                        sum(lora_A_grads) / len(lora_A_grads) if lora_A_grads else 1e-8
                    )
                    b_avg = (
                        sum(lora_B_grads) / len(lora_B_grads) if lora_B_grads else 1e-8
                    )
                    gradient_ratio = b_avg / a_avg if a_avg > 1e-8 else float("inf")
                    lora_gradient_history["gradient_ratio"].append(gradient_ratio)

                    # Report gradient flow at key intervals
                    should_report = step == 0 or step % 50 == 0 or step % 100 == 0

                    if should_report:
                        logger.info(
                            f"=== LoRA Gradient Flow Analysis (Step {step}) ==="
                        )
                        logger.info(f"Total LoRA params: {total_lora_params}")
                        # Calculate expected counts dynamically (should be total_lora_params / 2)
                        expected_A_B_count = total_lora_params // 2
                        logger.info(
                            f"LoRA A active: {len(lora_A_active)}/{expected_A_B_count} (avg_grad={sum(lora_A_grads)/len(lora_A_grads) if lora_A_grads else 0:.6f})"
                        )
                        logger.info(
                            f"LoRA B active: {len(lora_B_active)}/{expected_A_B_count} (avg_grad={sum(lora_B_grads)/len(lora_B_grads) if lora_B_grads else 0:.6f})"
                        )
                        logger.info(f"Gradient ratio (B/A): {gradient_ratio:.2f}")
                        logger.info(
                            f"Total active: {len(lora_A_active) + len(lora_B_active)}/{total_lora_params} ({100*(len(lora_A_active) + len(lora_B_active))/total_lora_params:.1f}%)"
                        )

                        if nonzero_grad_params:
                            logger.info("Modules receiving gradients:")
                            gradient_summary = {}
                            for name, grad_norm in nonzero_grad_params[:10]:
                                module_type = (
                                    name.split(".")[-2] if "." in name else "unknown"
                                )
                                if module_type not in gradient_summary:
                                    gradient_summary[module_type] = []
                                gradient_summary[module_type].append((name, grad_norm))

                            for module_type, params in gradient_summary.items():
                                logger.info(
                                    f"  {module_type}: {len(params)} params, avg_grad={sum(p[1] for p in params)/len(params):.6f}"
                                )

                        if null_grad_params:
                            logger.warning(
                                "Modules with NULL gradients (not in forward path):"
                            )
                            module_types = {}
                            for name in null_grad_params[:10]:
                                module_type = (
                                    name.split(".")[-2] if "." in name else "unknown"
                                )
                                module_types[module_type] = (
                                    module_types.get(module_type, 0) + 1
                                )
                            for module_type, count in module_types.items():
                                logger.warning(
                                    f"  {module_type}: {count} params with null gradients"
                                )

                        if zero_grad_params:
                            logger.warning(
                                "Modules with ZERO gradients (in path but no gradient):"
                            )
                            module_types = {}
                            lora_types = {}
                            for name in zero_grad_params:
                                module_type = (
                                    name.split(".")[-2] if "." in name else "unknown"
                                )
                                module_types[module_type] = (
                                    module_types.get(module_type, 0) + 1
                                )
                                # Check if it's lora_A or lora_B
                                if "lora_A" in name:
                                    lora_types["lora_A"] = (
                                        lora_types.get("lora_A", 0) + 1
                                    )
                                elif "lora_B" in name:
                                    lora_types["lora_B"] = (
                                        lora_types.get("lora_B", 0) + 1
                                    )
                            for module_type, count in module_types.items():
                                logger.warning(
                                    f"  {module_type}: {count} params with zero gradients"
                                )
                            if lora_types:
                                logger.warning(
                                    f"LoRA matrix breakdown (zero grads): {lora_types}"
                                )

                            # Show some specific examples
                            logger.warning(
                                f"Zero gradient examples: {zero_grad_params[:5]}"
                            )

                        if zero_grad_count == total_lora_params:
                            logger.error(
                                "CRITICAL: No gradients flowing to ANY LoRA params!"
                            )
                        elif zero_grad_count > total_lora_params * 0.5:
                            logger.warning(
                                f"CONCERNING: {zero_grad_count}/{total_lora_params} ({100*zero_grad_count/total_lora_params:.1f}%) LoRA params have no gradients"
                            )
                        else:
                            logger.info(
                                f"Acceptable gradient flow: {total_lora_params - zero_grad_count}/{total_lora_params} LoRA params receiving gradients"
                            )
                else:
                    mean_lora_grad = float("nan")
                    max_lora_grad = float("nan")

                head_grads = [
                    p.grad.detach().norm().item()
                    for p in proj.parameters()
                    if p.grad is not None
                ]
                mean_head_grad = (
                    sum(head_grads) / len(head_grads) if head_grads else 0.0
                )
                max_head_grad = max(head_grads) if head_grads else 0.0

                if step == 0:
                    logger.info(
                        f"Grad norms (first step): lora_mean={mean_lora_grad:.6f} lora_max={max_lora_grad:.6f} proj_mean={mean_head_grad:.6f} proj_max={max_head_grad:.6f}"
                    )

                # Periodic gradient monitoring
                if step % 100 == 0 and step > 0:
                    logger.debug(
                        f"Step {step} grad norms: lora_mean={mean_lora_grad:.6f} proj_mean={mean_head_grad:.6f}"
                    )

                    # Check for vanishing/exploding gradients
                    if used_lora and mean_lora_grad < 1e-8:
                        logger.warning(
                            f"Step {step}: Very small LoRA gradients ({mean_lora_grad:.2e}), possible vanishing gradient issue"
                        )
                    elif used_lora and mean_lora_grad > 10.0:
                        logger.warning(
                            f"Step {step}: Large LoRA gradients ({mean_lora_grad:.2e}), possible exploding gradient issue"
                        )

            except Exception as e:
                logger.warning(f"Failed to compute gradient diagnostics: {e}")

            # Increment accumulation step
            accumulation_step += 1

            # Only step optimizer at end of accumulation cycle
            if accumulation_step == cfg.gradient_accumulation_steps:
                opt.step()
                accumulation_step = 0  # Reset accumulation counter
                step += 1  # Only increment global step after optimizer step

            epoch_steps += 1
            # Store unscaled loss for logging (multiply back by accumulation steps)
            current_loss = loss.item() * cfg.gradient_accumulation_steps
            running_loss += current_loss
            epoch_loss += current_loss
            loss_history.append(current_loss)

            if step % 20 == 0:
                avg_loss = running_loss / 20
                logger.info(
                    f"LoRA adapt epoch {epoch+1} step {step}: loss={current_loss:.4f} (avg_20={avg_loss:.4f})"
                )
                running_loss = 0.0

            if cfg.max_steps and step >= cfg.max_steps:
                break

        # Log epoch summary
        if epoch_steps > 0:
            epoch_avg = epoch_loss / epoch_steps
            logger.info(
                f"Epoch {epoch+1} completed: avg_loss={epoch_avg:.4f} ({epoch_steps} steps)"
            )

        if cfg.max_steps and step >= cfg.max_steps:
            break

    # Post-training diagnostics
    if used_lora and lora_before is not None:
        try:
            deltas = []
            norms = []
            for n, p in _lora_named_params(model):
                if n in lora_before:
                    dp = (p.detach() - lora_before[n]).float()
                    deltas.append(dp.norm().item())
                    norms.append(p.detach().float().norm().item())
            if deltas:
                mean_delta = sum(deltas) / len(deltas)
                max_delta = max(deltas)
                mean_norm = sum(norms) / len(norms) if norms else float("nan")
                logger.info(
                    f"LoRA param delta norms: mean={mean_delta:.6f} max={max_delta:.6f} current_mean_norm={mean_norm:.6f}"
                )
        except Exception:
            pass

    # Enhanced post-adaptation diagnostics
    try:
        if pre_emb is not None:
            # Evaluate post-adaptation embeddings in eval mode for determinism
            was_training = model.training
            model.eval()
            with torch.no_grad():
                first_batch = next(iter(dataloader))[0].to(device)
                B = min(first_batch.shape[0], pre_emb.shape[0])

                # BEATs requires per-sample processing
                zs = []
                for i in range(B):
                    # Extract single sample: first_batch[i] should be [C, T] after indexing
                    e, _ = _extract_features_with_optional_grad(
                        first_batch[i], want_grad=False
                    )
                    zs.append(_pool_embedding(e))
                post_emb = torch.stack(zs, dim=0)

                # Compute multiple similarity metrics
                pre_n = nn.functional.normalize(pre_emb[:B], dim=-1)
                post_n = nn.functional.normalize(post_emb[:B], dim=-1)
                cos_sim = (pre_n * post_n).sum(dim=-1)

                # L2 distance
                l2_dist = torch.norm(pre_emb[:B] - post_emb[:B], dim=-1)

                # Embedding magnitude change
                pre_mag = torch.norm(pre_emb[:B], dim=-1)
                post_mag = torch.norm(post_emb[:B], dim=-1)
                mag_change = (post_mag - pre_mag) / pre_mag

                logger.info(f"Adaptation metrics (over {B} samples):")
                logger.info(
                    f"  Cosine similarity pre/post: {cos_sim.mean().item():.6f} ± {cos_sim.std().item():.6f}"
                )
                logger.info(
                    f"  L2 distance pre/post: {l2_dist.mean().item():.6f} ± {l2_dist.std().item():.6f}"
                )
                logger.info(
                    f"  Magnitude change: {mag_change.mean().item():.6f} ± {mag_change.std().item():.6f}"
                )

            # Restore training state
            model.train(was_training)

    except Exception as e:
        logger.warning(f"Failed to compute post-adaptation diagnostics: {e}")

    # Comprehensive training summary with LoRA gradient trends
    if loss_history:
        final_losses = loss_history[-min(10, len(loss_history)) :]
        logger.info(
            "Final loss trend (last {} steps): {}",
            len(final_losses),
            [f"{loss_val:.4f}" for loss_val in final_losses],
        )
        if len(loss_history) > 20:
            early_losses = loss_history[:10]
            logger.info(
                f"Early vs final loss: {sum(early_losses)/len(early_losses):.4f} -> {sum(final_losses)/len(final_losses):.4f}"
            )

    # LoRA gradient evolution summary
    if lora_gradient_history["steps"]:
        logger.info("=== LoRA Gradient Evolution Summary ===")
        logger.info(f"Training steps: {len(lora_gradient_history['steps'])}")

        # Initial vs final gradient patterns
        initial_A = (
            lora_gradient_history["lora_A_active"][0]
            if lora_gradient_history["lora_A_active"]
            else 0
        )
        initial_B = (
            lora_gradient_history["lora_B_active"][0]
            if lora_gradient_history["lora_B_active"]
            else 0
        )
        final_A = (
            lora_gradient_history["lora_A_active"][-1]
            if lora_gradient_history["lora_A_active"]
            else 0
        )
        final_B = (
            lora_gradient_history["lora_B_active"][-1]
            if lora_gradient_history["lora_B_active"]
            else 0
        )

        expected_count = (
            total_lora_params // 2 if "total_lora_params" in locals() else 72
        )
        logger.info(
            f"LoRA A evolution: {initial_A}/{expected_count} -> {final_A}/{expected_count} active params ({final_A - initial_A:+d})"
        )
        logger.info(
            f"LoRA B evolution: {initial_B}/{expected_count} -> {final_B}/{expected_count} active params ({final_B - initial_B:+d})"
        )

        # Gradient magnitude trends
        initial_A_grad = (
            lora_gradient_history["lora_A_avg_grad"][0]
            if lora_gradient_history["lora_A_avg_grad"]
            else 0
        )
        final_A_grad = (
            lora_gradient_history["lora_A_avg_grad"][-1]
            if lora_gradient_history["lora_A_avg_grad"]
            else 0
        )
        initial_B_grad = (
            lora_gradient_history["lora_B_avg_grad"][0]
            if lora_gradient_history["lora_B_avg_grad"]
            else 0
        )
        final_B_grad = (
            lora_gradient_history["lora_B_avg_grad"][-1]
            if lora_gradient_history["lora_B_avg_grad"]
            else 0
        )

        logger.info(
            f"LoRA A avg gradient: {initial_A_grad:.6f} -> {final_A_grad:.6f} ({final_A_grad/initial_A_grad if initial_A_grad > 1e-8 else float('inf'):.2f}x)"
        )
        logger.info(
            f"LoRA B avg gradient: {initial_B_grad:.6f} -> {final_B_grad:.6f} ({final_B_grad/initial_B_grad if initial_B_grad > 1e-8 else float('inf'):.2f}x)"
        )

        # Overall utilization trend
        initial_total = (
            lora_gradient_history["total_active"][0]
            if lora_gradient_history["total_active"]
            else 0
        )
        final_total = (
            lora_gradient_history["total_active"][-1]
            if lora_gradient_history["total_active"]
            else 0
        )
        total_expected = expected_count * 2
        logger.info(
            f"Total LoRA utilization: {initial_total}/{total_expected} -> {final_total}/{total_expected} ({100*final_total/total_expected:.1f}%, {final_total-initial_total:+d})"
        )

        # Gradient ratio evolution (helpful for understanding LoRA balance)
        initial_ratio = (
            lora_gradient_history["gradient_ratio"][0]
            if lora_gradient_history["gradient_ratio"]
            else float("inf")
        )
        final_ratio = (
            lora_gradient_history["gradient_ratio"][-1]
            if lora_gradient_history["gradient_ratio"]
            else float("inf")
        )
        logger.info(
            f"Gradient ratio (B/A) evolution: {initial_ratio:.2f} -> {final_ratio:.2f}"
        )

        if len(lora_gradient_history["steps"]) > 10:
            # Show trend over middle of training
            mid_idx = len(lora_gradient_history["steps"]) // 2
            mid_total = lora_gradient_history["total_active"][mid_idx]
            logger.info(
                f"Mid-training utilization: {mid_total}/{total_expected} ({100*mid_total/total_expected:.1f}%)"
            )

            # Detect if LoRA utilization is improving
            trend = (
                "improving"
                if final_total > initial_total
                else "stable"
                if final_total == initial_total
                else "declining"
            )
            logger.info(f"LoRA utilization trend: {trend}")

    # Ensure model is in eval mode for downstream evaluation
    model.eval()

    # Validate LoRA adapter state
    if used_lora:
        try:
            # Check if LoRA adapters are properly enabled
            if hasattr(model, "set_adapter"):
                model.set_adapter("default")
            if hasattr(model, "enable_adapter_layers"):
                model.enable_adapter_layers()

            # Count active LoRA parameters
            active_lora_params = sum(
                1
                for n, p in model.named_parameters()
                if "lora_" in n and p.requires_grad
            )
            total_lora_params = sum(
                1 for n, p in model.named_parameters() if "lora_" in n
            )
            logger.info(
                f"LoRA adapter state: {active_lora_params}/{total_lora_params} LoRA params active"
            )
        except Exception as e:
            logger.warning(f"Failed to validate LoRA adapter state: {e}")

    logger.info("LoRA adaptation finished; model set to eval mode")
    return model
