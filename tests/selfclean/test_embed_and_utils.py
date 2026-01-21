from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from selfclean_audio.config import _detect_template_type
from selfclean_audio.selfclean_audio import embed_dataset, extract_temporal_stats_batch


class StubModel(torch.nn.Module):
    def __init__(self, emb_dim=16, t_steps=5):
        super().__init__()
        self.emb_dim = emb_dim
        self.t_steps = t_steps

    def extract_features(self, batch: torch.Tensor):  # type: ignore
        # batch shape: [C, T] or [1, T]
        if batch.ndim == 1:
            batch = batch.unsqueeze(0)
        b = 1  # single waveform
        emb = torch.randn(b, self.emb_dim, device=batch.device)
        emb_t = torch.randn(b, self.t_steps, self.emb_dim, device=batch.device)
        return emb, emb_t


class TinyAudioDataset(Dataset):
    def __init__(self, n=4, length=400, sr=16000, root: Path | None = None):
        self.n = n
        self.length = length
        self.sample_rate = sr
        self.root = Path(root) if root else Path("./")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        wav = torch.randn(1, self.length)
        path = f"/tmp/audio_{idx}.wav"
        label = torch.tensor(idx % 2, dtype=torch.long)
        noisy_label = torch.tensor(0, dtype=torch.long)
        return wav, path, label, noisy_label


def test_extract_temporal_stats_batch_shapes():
    N, T, D = 3, 7, 4
    x = torch.randn(N, T, D)
    out = extract_temporal_stats_batch(x)
    assert out.shape == (N, 8)


def test_embed_dataset_with_stub_model(tmp_path):
    ds = TinyAudioDataset(n=3, length=200)
    dl = DataLoader(ds, batch_size=1)
    model = StubModel(emb_dim=12, t_steps=4)
    emb, paths, labels, noisy_labels = embed_dataset(
        dataloader=dl,
        model=model,
        normalize=False,
        memmap=False,
        memmap_path=tmp_path,
        tqdm_desc="test",
        device="cpu",
        workdir=str(tmp_path),
        save_plots=True,
    )
    assert emb.shape == (len(ds), model.emb_dim)
    assert labels.shape[0] == len(ds)
    assert noisy_labels.shape[0] == len(ds)
    # Plots created
    f1 = tmp_path / "Figure1.png"
    f2 = tmp_path / "Figure2.png"
    assert f1.exists()
    assert f2.exists()
    # Clean up plot artifacts
    try:
        f1.unlink()
        f2.unlink()
    except FileNotFoundError:
        pass


def test_detect_template_type_mapping():
    from omegaconf import OmegaConf

    from selfclean_audio.config import TemplateType

    for issue, expected in [
        ("duplicates", TemplateType.DUPLICATES),
        ("label_errors", TemplateType.LABEL_ERRORS),
        ("off_topic_noise", TemplateType.OFF_TOPIC),
    ]:
        cfg = OmegaConf.create({"ISSUE_TYPE": issue})
        assert _detect_template_type(cfg) == expected
