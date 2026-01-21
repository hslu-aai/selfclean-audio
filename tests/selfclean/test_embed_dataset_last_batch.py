import torch
from torch.utils.data import DataLoader, Dataset

from selfclean_audio.selfclean_audio import embed_dataset


class _TinyWaveDataset(Dataset):
    def __init__(self, n: int, t: int = 8):
        self.n = n
        self.t = t

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        # waveform encodes idx as constant value
        val = float(idx)
        wav = torch.full((1, self.t), val, dtype=torch.float32)
        path = f"/tmp/sample_{idx}.wav"
        label = torch.tensor(idx % 3, dtype=torch.long)
        noisy = torch.tensor(0, dtype=torch.long)
        return wav, path, label, noisy


class _DummyModel:
    def extract_features(self, x: torch.Tensor):  # type: ignore[override]
        # x: [C, T] or [T]
        if x.ndim == 1:
            val = x.mean()
        else:
            val = x.mean()
        # emb: [1, D], emb_t: [T, D]
        D = 3
        T = x.shape[-1]
        emb = torch.full((1, D), float(val.item()), dtype=torch.float32)
        emb_t = torch.full((T, D), float(val.item()), dtype=torch.float32)
        return emb, emb_t


def test_embed_dataset_handles_last_batch_without_overflow():
    ds = _TinyWaveDataset(n=5, t=8)
    dl = DataLoader(ds, batch_size=4, shuffle=False)
    model = _DummyModel()

    emb_space, paths, labels, noisy = embed_dataset(
        dataloader=dl,
        model=model,  # type: ignore[arg-type]
        normalize=False,
        memmap=False,
        tqdm_desc=None,
        device="cpu",
        workdir="./",
        save_plots=False,
    )

    # Expect shape [N, D]
    assert emb_space.shape[0] == len(ds)
    assert emb_space.shape[1] == 3

    # Embedding row i should equal i repeated over D
    import numpy as np

    for i in range(len(ds)):
        assert np.allclose(emb_space[i], np.array([i, i, i], dtype=np.float32))

    # Paths and labels sizes should match
    assert len(paths) == len(ds)
    assert labels.shape[0] == len(ds)
    assert noisy.shape[0] == len(ds)
