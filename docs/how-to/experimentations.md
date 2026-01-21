# Experimentations

```{note}
Synthetic datasets and model weights are available here:
https://drive.switch.ch/index.php/s/2PAeHBFyw9GFtBv
```

To standardize experiments and keep configurations reproducible, we use
`Lazy`-initialized config files.

For example, to run experiments on the `pianos` dataset, create a folder under
`./configs/pianos/` and place a config like `BEAT_alpha_03.py` with the model
and synthetic alpha noise rate.

```python
import numpy as np

from selfclean_audio.config import LazyCall as L
from selfclean_audio.datasets import FolderAudioDataset
from selfclean_audio.selfclean_audio import PretrainingSSL, SelfCleanAudio

SEED = 42

dataset = L(FolderAudioDataset)(
    root="/home/alvaro/projects/selfclean_audio/piano_file_alpha03/dataset/",
    convert_mono=True,
    sample_rate=16000,
)

dataloader = dict(num_workers=8, batch_size=16, drop_last=False, pin_memory=True)

selfclean_audio = L(SelfCleanAudio)(
    # distance calculation
    distance_function_path="sklearn.metrics.pairwise.",
    distance_function_name="cosine_similarity",
    chunk_size=100,
    precision_type_distance=np.float32,
    # memory management
    memmap=True,
    memmap_path=None,
    # plotting
    plot_distribution=False,
    plot_top_N=None,
    output_path=None,
    figsize=(10, 8),
    # model
    pretraining_ssl=PretrainingSSL.BEATS,
    model_path="/home/alvaro/Documents/BEATs_iter3.pt",
    # utils
    random_seed=SEED,
    device="cuda",
)

params = dict(
    seed=SEED,
    cudnn_benchmark=True,
    cudnn_deterministic=False,
)
```

Once your configuration file is ready, run the following command:

```bash
python3 -m selfclean_audio --config configs/pianos/BEAT_alpha_03.py
```

Make sure that the filename in the `--config` argument matches your actual configuration file.
