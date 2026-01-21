import numpy as np

from SelfClean.selfclean.cleaner.issue_manager import IssueManager, IssueTypes
from selfclean_audio.selfclean_audio import SelfCleanAudio


def _make_manager_for_indices(issue_type: IssueTypes, indices):
    return IssueManager({issue_type.value: {"indices": np.asarray(indices)}})


def test_calculate_scores_nonpair_known_ranking():
    """Goal: Check evaluation metrics (AUROC/AP/AUPRG) on a perfect non-pair ranking with sufficient length for @K."""

    # Build a sufficiently long ranking (>= 20) to satisfy calculator's @k logging
    class _DS:
        def get_errors(self):
            # 25 items, first 5 are true issues
            return [1] * 5 + [0] * 20

    # Predicted ranking puts all true issues at the top, then the rest
    pred_order = list(range(0, 5)) + list(range(5, 25))
    im = _make_manager_for_indices(IssueTypes.LABEL_ERRORS, pred_order)

    sc = object.__new__(SelfCleanAudio)
    sc.issues_to_detect = [IssueTypes.LABEL_ERRORS]

    out = SelfCleanAudio.calculate_scores(sc, im, noisy_labels=None, dataset=_DS())
    scores = out.issue_dict["Scores-label_errors"]
    # Perfect separation
    assert scores["evaluation/AUROC"] == 1.0
    assert scores["evaluation/AP"] == 1.0
    assert scores["evaluation/AUPRG"] == 1.0


def test_calculate_scores_pairs_known_ranking():
    """Goal: Check evaluation metrics (AUROC/AP/AUPRG) on a perfect pair ranking and avoid @K index errors."""

    # Dataset returns ground-truth duplicate pairs, ensure predicted ranking has >= 20 entries
    class _DS:
        def get_errors(self):
            # 5 true pairs among many distractors
            true_pairs = {(0, 3), (1, 4), (2, 5), (6, 7), (8, 9)}
            return true_pairs, ["original", "duplicate"]

    # Predicted ranking: put all true pairs first, then many non-issue pairs to reach length >= 20
    true_pairs_ordered = [(0, 3), (1, 4), (2, 5), (6, 7), (8, 9)]
    fillers = []
    # generate filler distinct pairs
    for i in range(10):
        for j in range(i + 1, 10):
            if (i, j) not in true_pairs_ordered and (j, i) not in true_pairs_ordered:
                fillers.append((i, j))
            if len(fillers) >= 20:
                break
        if len(fillers) >= 20:
            break
    pred_pairs = true_pairs_ordered + fillers
    im = _make_manager_for_indices(IssueTypes.NEAR_DUPLICATES, pred_pairs)

    sc = object.__new__(SelfCleanAudio)
    sc.issues_to_detect = [IssueTypes.NEAR_DUPLICATES]

    out = SelfCleanAudio.calculate_scores(sc, im, noisy_labels=None, dataset=_DS())
    scores = out.issue_dict["Scores-near_duplicates"]
    assert scores["evaluation/AUROC"] == 1.0
    assert scores["evaluation/AP"] == 1.0
    assert scores["evaluation/AUPRG"] == 1.0


def test_calculate_scores_raises_without_get_errors():
    """Goal: Confirm calculate_scores raises when dataset lacks get_errors for both pair and non-pair flows."""

    # No get_errors on dataset should raise for both paths
    class _DS:
        pass

    # Nonpair case
    im1 = _make_manager_for_indices(IssueTypes.LABEL_ERRORS, [0, 1, 2])
    sc = object.__new__(SelfCleanAudio)
    sc.issues_to_detect = [IssueTypes.LABEL_ERRORS]
    try:
        SelfCleanAudio.calculate_scores(sc, im1, noisy_labels=None, dataset=_DS())
        raised_nonpair = False
    except ValueError:
        raised_nonpair = True
    assert raised_nonpair

    # Pair case
    im2 = _make_manager_for_indices(IssueTypes.NEAR_DUPLICATES, [(0, 1), (1, 2)])
    sc.issues_to_detect = [IssueTypes.NEAR_DUPLICATES]
    try:
        SelfCleanAudio.calculate_scores(sc, im2, noisy_labels=None, dataset=_DS())
        raised_pair = False
    except ValueError:
        raised_pair = True
    assert raised_pair
