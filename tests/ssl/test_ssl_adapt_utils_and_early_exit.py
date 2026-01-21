import torch
from torch.utils.data import DataLoader

from selfclean_audio.ssl_adapt import (
    LoraAdaptConfig,
    ProjectionHead,
    _info_nce_loss,
    _pool_embedding,
    _two_views,
    _vicreg_loss,
    adapt_model_with_lora,
)


class _StubModel(torch.nn.Module):
    def __init__(self, emb_dim=8, t_steps=4):
        super().__init__()
        self.emb_dim = emb_dim
        self.t_steps = t_steps

    def extract_features(self, batch: torch.Tensor):  # type: ignore
        # Return a 1D main embedding to avoid extra batch dim after pooling
        emb = torch.randn(self.emb_dim, device=batch.device)
        emb_t = torch.randn(self.t_steps, self.emb_dim, device=batch.device)
        return emb, emb_t


def test_ssl_utils_basic_ops():
    """Goal: Cover SSL utility functions (losses, projection, pooling, and augment views)."""
    x = torch.randn(3, 5)
    y = torch.randn(3, 5)
    # Loss functions run
    _ = _info_nce_loss(x, y, temperature=0.2)
    _ = _vicreg_loss(x, y)

    # ProjectionHead forward
    ph = ProjectionHead(in_dim=5, proj_dim=4)
    out = ph(x)
    assert out.shape == (3, 4)

    # Pooling
    z3 = torch.randn(2, 4, 6)
    z_pooled = _pool_embedding(z3)
    assert z_pooled.shape == (2, 6)

    # Two views with weak aug path
    w = torch.randn(2, 1, 32)
    cfg = LoraAdaptConfig(strong_aug=False)
    v1, v2 = _two_views(w, cfg)
    assert v1.shape == w.shape
    assert v2.shape == w.shape


def test_adapt_model_early_exit_and_single_step():
    """Goal: Validate adapt_model_with_lora returns model when disabled and runs a single training step without PEFT."""
    # Early exit when disabled
    model = _StubModel()
    cfg = LoraAdaptConfig(enable=False)
    assert adapt_model_with_lora(model, [], device="cpu", cfg=cfg) is model

    # Single-step quick run without PEFT
    class _DS:
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return torch.randn(1, 32), f"p{idx}", torch.tensor(0)

    dl = DataLoader(_DS(), batch_size=1)
    cfg2 = LoraAdaptConfig(enable=True, epochs=1, max_steps=1, strong_aug=False)
    _ = adapt_model_with_lora(_StubModel(), dl, device="cpu", cfg=cfg2)
