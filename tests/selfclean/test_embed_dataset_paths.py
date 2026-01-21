import numpy as np
import torch
from torch.utils.data import DataLoader

from selfclean_audio.selfclean_audio import _infer_shapes, embed_dataset


class _DS:
    def __init__(self):
        self.sample_rate = 16000
        self.files = ["a.wav", "b.wav"]

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        # Waveform [C, T]
        wav = torch.randn(1, 8)
        path = self.files[idx]
        label = torch.tensor(idx % 2)
        noisy = torch.tensor(0)
        return wav, path, label, noisy


class _Model:
    def extract_features(self, batch: torch.Tensor):  # type: ignore
        # Return main embedding [1, D] and temporal embeddings [1, T, D]
        B = 1
        T = 4
        D = 6
        emb = torch.randn(B, D, device=batch.device)
        emb_t = torch.randn(B, T, D, device=batch.device)
        return emb, emb_t


def test_embed_dataset_memmap_and_normalize(tmp_path):
    """Goal: Exercise embed_dataset with normalization and memmap storage, verifying shapes and lengths."""
    dl = DataLoader(_DS(), batch_size=1)
    model = _Model()
    emb, paths, labels, noisy = embed_dataset(
        dataloader=dl,
        model=model,
        normalize=True,
        memmap=True,
        memmap_path=tmp_path,
        device="cpu",
        workdir=str(tmp_path),
    )
    # Should be a memmap and have correct length
    assert isinstance(emb, np.memmap)
    assert emb.shape[0] == len(dl.dataset)
    assert len(paths) == len(dl.dataset)


def test_infer_shapes_with_no_model():
    """Goal: Ensure _infer_shapes infers embedding dimension correctly when model is None."""
    dl = DataLoader(_DS(), batch_size=1)
    sample, emb_dim = _infer_shapes(dl, model=None, device="cpu")
    # emb_dim equals waveform length (8)
    assert emb_dim == 8
