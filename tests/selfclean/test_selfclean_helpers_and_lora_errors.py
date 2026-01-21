import pytest
import torch
from torch.utils.data import DataLoader

from selfclean_audio.selfclean_audio import (
    PretrainingSSL,
    SelfCleanAudio,
    create_memmap,
    create_memmap_path,
    extract_temporal_stats_batch,
)


class _StubModel(torch.nn.Module):
    def __init__(self, emb_dim=4, t_steps=3):
        super().__init__()
        self.emb_dim = emb_dim
        self.t_steps = t_steps

    def extract_features(self, batch: torch.Tensor):  # type: ignore
        if batch.ndim == 1:
            batch = batch.unsqueeze(0)
        b = 1
        emb = torch.randn(b, self.emb_dim, device=batch.device)
        emb_t = torch.randn(b, self.t_steps, self.emb_dim, device=batch.device)
        return emb, emb_t


def test_create_memmap_and_path(tmp_path):
    """Goal: Cover memmap directory creation and file creation utility functions."""
    # None -> temp dir
    p = create_memmap_path(None)
    assert p.exists()
    # Explicit path
    p2 = create_memmap_path(tmp_path)
    assert p2.exists()
    # Create memmap file
    m = create_memmap(p2, "emb.dat", 5, 8)
    assert m.shape == (5, 8)
    m[0] = 1.0
    m.flush()


def test_extract_temporal_stats_shape():
    """Goal: Ensure extract_temporal_stats_batch outputs the expected [N, 8] shape from [N, T, D]."""
    x = torch.randn(2, 5, 7)
    feats = extract_temporal_stats_batch(x)
    assert feats.shape == (2, 8)


def test_lora_enabled_requires_sample_rate(monkeypatch):
    """Goal: Validate LoRA-enabled run_on_dataloader enforces dataset.sample_rate presence and raises accordingly."""
    # Monkeypatch to avoid heavy model loading
    import SelfClean.selfclean.core.src.pkg.embedder as embedder

    monkeypatch.setattr(
        embedder.Embedder, "load_pretrained", lambda *a, **k: _StubModel()
    )

    # Build SelfCleanAudio with LoRA enabled
    sc = SelfCleanAudio(
        pretraining_ssl=PretrainingSSL.BEATS,  # value unused due to monkeypatch
        model_path="dummy",
        memmap=False,
        device="cpu",
        lora_enable=True,
        adapt_epochs=1,
    )

    class _NoRate:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return torch.randn(1, 16), "p.wav", torch.tensor(0)

    dl = DataLoader(_NoRate(), batch_size=1)
    with pytest.raises(ValueError):
        sc.run_on_dataloader(dl)
