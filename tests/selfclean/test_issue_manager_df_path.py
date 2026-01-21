from typing import cast

import numpy as np
import pandas as pd

from SelfClean.selfclean.cleaner.issue_manager import IssueManager, IssueTypes


def test_issue_manager_dataframe_mapping():
    """Goal: Validate IssueManager DataFrame export expands indices and maps paths for pairs and single indices."""
    # Build issue_dict with a pair indices array and a single-index array
    near_dup_indices = np.array([[0, 2], [1, 3], [4, 5]])
    near_dup_scores = np.array([0.1, 0.2, 0.3])
    label_err_indices = np.array([5, 2, 0])
    issue_dict = {
        IssueTypes.NEAR_DUPLICATES.value: {
            "indices": near_dup_indices,
            "scores": near_dup_scores,
        },
        IssueTypes.LABEL_ERRORS.value: {
            "indices": label_err_indices,
            "scores": np.array([0.9, 0.5, 0.1]),
            "auto_issues": [0, 2],  # mark via index expansion
        },
    }
    meta = {"paths": np.array([f"p{i}" for i in range(10)])}
    im = IssueManager(issue_dict, meta)

    df_result = im.get_issues(IssueTypes.NEAR_DUPLICATES.value, return_as_df=True)
    assert isinstance(df_result, pd.DataFrame)
    df = cast(pd.DataFrame, df_result)
    # Expect path columns expanded for both endpoints of the pair
    assert any("paths_indices_1" in c for c in df.columns)
    assert any("paths_indices_2" in c for c in df.columns)

    df2_result = im.get_issues(IssueTypes.LABEL_ERRORS.value, return_as_df=True)
    assert isinstance(df2_result, pd.DataFrame)
    df2 = cast(pd.DataFrame, df2_result)
    # auto_issues column exists and flags the provided indices
    assert "auto_issues" in df2.columns
    assert df2["auto_issues"].sum() == 2
