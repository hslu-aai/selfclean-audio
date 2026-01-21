import torch
import torch.nn as nn

from selfclean_audio.ssl_adapt import _pool_embedding


def test_pool_embedding_preserves_grad_flow():
    """_pool_embedding must not detach the graph; verify gradients flow."""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 6, requires_grad=True)
    y = _pool_embedding(x)  # [2, 6]
    # Simple scalar loss
    loss = (y**2).sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum().item() > 0.0


def test_peft_lora_receives_grad_and_updates():
    """Attach LoRA to a tiny block and ensure lora_* params get grads and update.
    This test requires `peft` to be installed; fail with a clear message if missing.
    """
    try:
        from peft import LoraConfig, get_peft_model  # type: ignore
    except Exception as e:
        raise AssertionError(
            "peft is required for LoRA tests. Install `peft` to run this test."
        ) from e

    torch.manual_seed(0)

    class TinyAttn(nn.Module):
        def __init__(self, d: int = 8):
            super().__init__()
            self.q_proj = nn.Linear(d, d)
            self.k_proj = nn.Linear(d, d)
            self.v_proj = nn.Linear(d, d)
            self.out_proj = nn.Linear(d, d)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Keep it simple but non-linear to propagate meaningful gradients
            x = torch.relu(self.q_proj(x))
            x = self.out_proj(x)
            return x

    class TinyBlock(nn.Module):
        def __init__(self, d: int = 8):
            super().__init__()
            self.self_attn = TinyAttn(d)
            self.fc1 = nn.Linear(d, 2 * d)
            self.fc2 = nn.Linear(2 * d, d)

        def extract_features(self, x: torch.Tensor):  # mimic audio models
            if x.ndim == 1:
                x = x.unsqueeze(0)  # [1, D]
            y = self.self_attn(x)
            y = torch.relu(self.fc1(y))
            y = self.fc2(y)
            return y.mean(dim=0), y  # (pooled, temporal)

    model = TinyBlock(d=8)
    # Configure LoRA for the target module names used in adaptation
    lcfg = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        bias="none",
        task_type="FEATURE_EXTRACTION",
        inference_mode=False,
    )
    model = get_peft_model(model, lcfg)

    # Collect LoRA params
    lora_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert any("lora_" in n for n, _ in lora_params)

    opt = torch.optim.AdamW((p for _, p in lora_params), lr=1e-3)

    # Snapshot before
    before = {n: p.detach().clone() for n, p in lora_params if "lora_" in n}

    # One forward/backward/step with a deterministic input
    x = torch.randn(3, 8)
    pooled, seq = model.extract_features(x)
    loss = pooled.pow(2).sum() + seq.pow(2).sum()
    opt.zero_grad(set_to_none=True)
    loss.backward()

    # Assert at least one LoRA param has gradient
    lora_grads = [p.grad for n, p in lora_params if "lora_" in n]
    assert any(
        g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
        for g in lora_grads
    )

    opt.step()

    # Verify LoRA params changed
    deltas = []
    for n, p in lora_params:
        if "lora_" in n:
            dp = (p.detach() - before[n]).abs().sum().item()
            deltas.append(dp)
    assert any(d > 0 for d in deltas)
