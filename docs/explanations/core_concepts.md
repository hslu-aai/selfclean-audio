# Core Concepts

This page explains the fundamental concepts of the `selfclean-audio` library.

## Dataset Issues

`selfclean-audio` is designed to detect the following types of issues in audio datasets:

-   **Near-Duplicates:** These are audio samples that are very similar to each other, but not identical. They can be caused by various factors, such as different encoding formats, small amounts of noise, or minor edits.
-   **Off-Topic Samples:** These are audio samples that do not belong to the same category as the rest of the dataset. For example, a dataset of bird sounds might contain a sample of a cat meowing.
-   **Label Errors:** These are audio samples that have been assigned the wrong label. For example, a sample of a dog barking might be labeled as a cat meowing.

## Baselines

For each type of issue, `selfclean-audio` provides a set of strong baselines for detection:

-   **Near-Duplicates:**
    -   `embedding_distance`: Computes the distance between audio embeddings.
    -   `audio_hash`: Uses audio fingerprinting to detect duplicates.
    -   `dejavu`: A powerful audio fingerprinting and recognition algorithm.
-   **Off-Topic:**
    -   `lad`: Likelihood Anomaly Detection.
    -   `quantile`: Anomaly detection based on quantiles.
    -   `isolation_forest`: An unsupervised anomaly detection algorithm.
    -   `cleanlab`: Uses confident learning to detect outliers.
-   **Label Errors:**
    -   `intra_extra_distance`: Compares the distance of a sample to its own class and other classes.
    -   `cleanlab`: Uses confident learning to find label errors.

## Evaluation

`selfclean-audio` provides a comprehensive evaluation framework to assess the performance of the issue detection methods. The following metrics are used:

-   **AUROC (Area Under the Receiver Operating Characteristic curve):** A measure of the overall performance of a binary classifier.
-   **AP (Average Precision):** Another measure of the overall performance of a binary classifier.
-   **Recall@K:** The proportion of true issues that are ranked in the top K.
-   **Precision@K:** The proportion of the top K ranked samples that are true issues.
