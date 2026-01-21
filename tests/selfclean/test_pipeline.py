import numpy as np
import pytest

from SelfClean.selfclean.cleaner.issue_manager import IssueTypes
from SelfClean.selfclean.cleaner.selfclean_cleaner import SelfCleanCleaner


def _make_embeddings(n=30, d=8):
    rng = np.random.default_rng(42)
    emb = rng.normal(size=(n, d)).astype(np.float32)
    # Make an obvious near-duplicate pair (0, 1)
    emb[1] = emb[0] + 1e-6
    return emb


def _make_labels(n=30):
    # Two classes, alternating labels
    return np.array([i % 2 for i in range(n)], dtype=int)


def test_label_errors_confident_learning_full_ranking():
    """Goal: Assert confident learning label-error method returns a sorted full ranking for all samples."""
    n, d = 30, 8
    emb = _make_embeddings(n, d)
    labels = _make_labels(n)

    cleaner = SelfCleanCleaner(
        memmap=False,
        plot_distribution=False,
        near_duplicate_method="embedding_distance",
        off_topic_method="lad",  # not used here
        label_error_method="cleanlab",  # confident learning implementation
    )
    cleaner.fit(emb_space=emb, labels=labels)

    issues = cleaner.predict(issues_to_detect=[IssueTypes.LABEL_ERRORS])
    out = issues[IssueTypes.LABEL_ERRORS.value]
    assert out is not None
    scores, indices = out["scores"], out["indices"]
    # Full ranking for all samples
    assert len(scores) == n
    assert len(indices) == n
    # Scores are sorted descending by error likelihood
    assert np.all(scores[:-1] >= scores[1:])


def test_off_topic_supervised_confidence_full_ranking():
    """Goal: Assert cleanlab off-topic method returns a sorted full ranking for all samples."""
    n, d = 30, 8
    emb = _make_embeddings(n, d)
    labels = _make_labels(n)

    cleaner = SelfCleanCleaner(
        memmap=False,
        plot_distribution=False,
        near_duplicate_method="embedding_distance",  # not used here
        off_topic_method="cleanlab",  # supervised confidence implementation
        label_error_method="intra_extra_distance",
    )
    cleaner.fit(emb_space=emb, labels=labels)

    issues = cleaner.predict(issues_to_detect=[IssueTypes.OFF_TOPIC_SAMPLES])
    out = issues[IssueTypes.OFF_TOPIC_SAMPLES.value]
    assert out is not None
    scores, indices = out["scores"], out["indices"]
    # Full ranking for all samples
    assert len(scores) == n
    assert len(indices) == n
    # Scores are sorted descending by off-topic likelihood
    assert np.all(scores[:-1] >= scores[1:])


def test_near_duplicates_embedding_distance_contains_obvious_pair():
    """Goal: Ensure embedding_distance near-duplicate method ranks an obvious near-duplicate pair at the top."""
    n, d = 20, 8
    emb = _make_embeddings(n, d)
    labels = _make_labels(n)

    cleaner = SelfCleanCleaner(
        memmap=False,
        plot_distribution=False,
        near_duplicate_method="embedding_distance",
        off_topic_method="lad",
        label_error_method="intra_extra_distance",
    )
    cleaner.fit(emb_space=emb, labels=labels)

    issues = cleaner.predict(issues_to_detect=[IssueTypes.NEAR_DUPLICATES])
    out = issues[IssueTypes.NEAR_DUPLICATES.value]
    assert out is not None
    scores, pairs = out["scores"], out["indices"]
    assert len(scores) == len(pairs)
    # The most similar pair should include (0, 1)
    top_pairs = set(tuple(map(int, p)) for p in pairs[:10])
    assert (0, 1) in top_pairs or (1, 0) in top_pairs


def test_dejavu_baseline_raises_when_dependencies_missing():
    """Goal: Verify Dejavu baseline raises when dependencies/files are missing, covering error-handling path."""
    n, d = 4, 4
    emb = _make_embeddings(n, d)
    labels = _make_labels(n)

    cleaner = SelfCleanCleaner(
        memmap=False,
        plot_distribution=False,
        near_duplicate_method="dejavu",
        near_duplicate_params={},
        off_topic_method="lad",
        label_error_method="intra_extra_distance",
    )
    cleaner.fit(
        emb_space=emb,
        labels=labels,
        paths=np.array([f"/no/such/file_{i}.wav" for i in range(n)]),
    )

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises((ImportError, FileNotFoundError, RuntimeError)):
            cleaner.predict(issues_to_detect=[IssueTypes.NEAR_DUPLICATES])
