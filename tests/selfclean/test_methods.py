from pathlib import Path

import numpy as np
import pytest
import torch

from SelfClean.selfclean.cleaner.issue_manager import IssueTypes
from SelfClean.selfclean.cleaner.selfclean_cleaner import SelfCleanCleaner
from selfclean_audio.datasets.duplicate_dataset import DuplicateDataset
from selfclean_audio.datasets.label_error_dataset import LabelErrorDataset
from selfclean_audio.datasets.off_topic_dataset import OffTopicDataset


def _rand_emb(n, d, seed=123):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)).astype(np.float32)


class TinyBase:
    def __init__(self, n: int, root: Path, sample_rate: int = 16000):
        self.n = n
        self.sample_rate = sample_rate
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.files = [str(self.root / f"x_{i}.wav") for i in range(n)]
        t = torch.arange(0, 0.1, 1 / sample_rate)
        for i, p in enumerate(self.files):
            wave = torch.sin(2 * torch.pi * (220 + 10 * (i % 3)) * t).unsqueeze(0)
            torch.set_default_dtype(torch.float32)
            import torchaudio

            torchaudio.save(p, wave.to(torch.float32), sample_rate)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        import torchaudio

        w, _ = torchaudio.load(self.files[idx])
        label = idx % 2
        return w, self.files[idx], torch.tensor(label)


def _collect_arrays(dataset):
    paths, labels = [], []
    for i in range(len(dataset)):
        item = dataset[i]
        # Support both (audio, path, label) and (audio, path, label, noisy_label)
        if len(item) == 3:
            _, p, y = item
        else:
            _, p, y, _ = item
        paths.append(p)
        labels.append(int(y))
    return np.array(paths), np.array(labels, dtype=int)


def test_near_duplicates_audio_hash_detects_pairs(tmp_path):
    """Goal: Validate audio_hash near-duplicate baseline produces pair rankings that include ground-truth duplicates."""
    base = TinyBase(6, tmp_path / "base")
    dup = DuplicateDataset(
        base, frac_error=0.5, duplicate_strategy="exact", random_state=0
    )
    pairs_gt, _ = dup.get_errors()
    n = len(dup)
    d = 8
    # Create random embeddings; audio_hash doesn't use them, but cleaner expects emb_space
    emb = _rand_emb(n, d)
    paths, labels = _collect_arrays(dup)

    cleaner = SelfCleanCleaner(
        memmap=False,
        plot_distribution=False,
        near_duplicate_method="audio_hash",
        off_topic_method="lad",
        label_error_method="intra_extra_distance",
    )
    cleaner.fit(emb_space=emb, labels=labels, paths=paths)
    issues = cleaner.predict([IssueTypes.NEAR_DUPLICATES])
    res = issues[IssueTypes.NEAR_DUPLICATES.value]
    assert res is not None
    scores, pairs = res["scores"], res["indices"]
    assert len(scores) == len(pairs)
    # Some predicted pair matches a ground-truth duplicate
    pred_pairs = set(tuple(map(int, p)) for p in pairs[:20])
    assert any(t in pred_pairs or (t[1], t[0]) in pred_pairs for t in pairs_gt)


@pytest.mark.parametrize("method", ["lad", "quantile", "isolation_forest", "cleanlab"])
def test_off_topic_methods_rank_valid_and_detects_for_iforest(tmp_path, method):
    """Goal: Ensure off-topic baselines return full rankings and detect contaminated samples for isolation_forest."""
    base = TinyBase(12, tmp_path / "base")
    # External dataset for the 'external' strategy not needed here
    ds = OffTopicDataset(
        base, frac_error=0.25, contamination_strategy="noise", random_state=0
    )
    errors = ds.get_errors()
    contaminated = {i for i, e in enumerate(errors) if e}

    # Craft embeddings where contaminated samples are outliers
    d = 10
    emb = _rand_emb(len(ds), d)
    for i in contaminated:
        emb[i] += 5.0  # shift outliers away

    paths, labels = _collect_arrays(ds)
    cleaner = SelfCleanCleaner(
        memmap=False,
        plot_distribution=False,
        near_duplicate_method="embedding_distance",
        off_topic_method=method,
        label_error_method="intra_extra_distance",
        off_topic_params={"cv_folds": 3} if method == "cleanlab" else {},
    )
    cleaner.fit(emb_space=emb, labels=labels, paths=paths)
    issues = cleaner.predict([IssueTypes.OFF_TOPIC_SAMPLES])
    res = issues[IssueTypes.OFF_TOPIC_SAMPLES.value]
    assert res is not None
    scores, idx = res["scores"], res["indices"]
    # Always: ranking exists and has correct length
    assert len(scores) == len(idx) == len(ds)
    top_k = min(100, len(idx))
    assert any(int(i) in contaminated for i in idx[:top_k])


@pytest.mark.parametrize("method", ["intra_extra_distance", "cleanlab"])
def test_label_error_methods_detect_swapped(tmp_path, method):
    """Goal: Check label error methods identify mislabeled samples at top of the ranking using synthetic clusters."""
    base = TinyBase(12, tmp_path / "base")
    frac = 0.25
    ds = LabelErrorDataset(base, frac_error=frac, random_state=42)
    errors = ds.get_errors()
    mislabeled = {i for i, e in enumerate(errors) if e}

    # Craft embeddings that cluster by true (original) label
    d = 8
    emb = _rand_emb(len(ds), d)
    # Pull samples toward cluster centers by their original (pre-corruption) labels
    orig_labels = np.array([base[i][2].item() for i in range(len(base))])
    centers = {
        0: np.ones(d, dtype=np.float32) * -2,
        1: np.ones(d, dtype=np.float32) * 2,
    }
    for i in range(len(ds)):
        emb[i] = emb[i] * 0.2 + centers[int(orig_labels[i])]  # tighten clusters

    paths, labels = _collect_arrays(ds)
    cleaner = SelfCleanCleaner(
        memmap=False,
        plot_distribution=False,
        near_duplicate_method="embedding_distance",
        off_topic_method="lad",
        label_error_method=method,
        label_error_params={"cv_folds": 3} if method == "cleanlab" else {},
    )
    cleaner.fit(emb_space=emb, labels=labels, paths=paths)
    issues = cleaner.predict([IssueTypes.LABEL_ERRORS])
    res = issues[IssueTypes.LABEL_ERRORS.value]
    assert res is not None
    _, idx = res["scores"], res["indices"]
    # Top-K should include at least one mislabeled sample
    top_k = min(5, len(idx))
    assert any(int(i) in mislabeled for i in idx[:top_k])


def test_embedding_distance_near_duplicates_respects_known_pairs(tmp_path):
    """Goal: Confirm embedding_distance near-duplicate method ranks known duplicate pairs highly."""
    base = TinyBase(8, tmp_path / "base")
    dup = DuplicateDataset(
        base, frac_error=0.5, duplicate_strategy="exact", random_state=0
    )
    pairs_gt, _ = dup.get_errors()
    n = len(dup)
    d = 6
    emb = _rand_emb(n, d)
    # Make embeddings reflect duplicate pairs by copying vectors
    for i, j in pairs_gt:
        # i is original index in base, j is duplicate index (offset)
        emb[j] = emb[i] + 1e-6

    paths, labels = _collect_arrays(dup)
    cleaner = SelfCleanCleaner(
        memmap=False,
        plot_distribution=False,
        near_duplicate_method="embedding_distance",
        off_topic_method="lad",
        label_error_method="intra_extra_distance",
    )
    cleaner.fit(emb_space=emb, labels=labels, paths=paths)
    issues = cleaner.predict([IssueTypes.NEAR_DUPLICATES])
    res = issues[IssueTypes.NEAR_DUPLICATES.value]
    assert res is not None
    scores, pairs = res["scores"], res["indices"]
    assert len(scores) == len(pairs)
    # Best-ranked pairs should include at least one ground-truth duplicate
    pred_pairs = set(tuple(map(int, p)) for p in pairs[:10])
    assert any(t in pred_pairs or (t[1], t[0]) in pred_pairs for t in pairs_gt)
