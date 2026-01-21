# Decision Records

This section documents the key research and development decisions made during the project.

We track decisions directly on this page for now. When formal decision records
are added, they will live under `docs/explanations/decisions/`.

## Audio Issue Categorization

We categorize audio quality issues at two levels:

- File Level: issues affecting the entire audio file.
  - Out-of-distribution audio
  - Duplicate files
  - Silent audio

- Segment (Audio) Level: issues affecting parts of the audio.
  - Looping artifacts
  - Gaussian noise
  - White noise
  - Background noise from other sources

To simulate these issues, we mix noise into the original audio using a
Signal-to-Noise Ratio (SNR) mixer adapted from our internal experiments.

## Embedding Pooling Strategy

Since SelfClean processes each sample independently and does not account for
temporal structure, our current approach treats any detected issue (e.g., white
noise) as a file-level problem. However, future work could explore more
fine-grained localization of issues within segments.

To better aggregate temporal information, we've implemented multiple pooling strategies for embedding generation:

```python
if pool == "CLS":  # Use class token
    emb = emb[:, 0, :]
elif pool == "Mean":  # Average pooling
    emb = emb.mean(dim=1)
elif pool == "Reshape":  # Flatten all tokens
    emb = emb.reshape(-1)
```

## Datasets Used

| Dataset  | Classes | Samples |  Duration | Sampling Rate |
| --------------- | --------------- | --------------- | --------------- | --------------- |
| ARCA23K | -- | 17,979 | 7.92 | 44100 |
| AudioSet20K | 527 | 39,436 | 9.89 | 32000 |
| Pianos | 8 | 668 | 4.86 | 16000|
| WMMS | 31 | 1695 | 10.42 | 16000|
| GTZAN | 10 | 930 | 30.02 | 22050|
