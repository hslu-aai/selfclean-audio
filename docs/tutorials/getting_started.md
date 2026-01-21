# Getting Started

This tutorial provides a step-by-step guide on how to use `selfclean-audio` to detect issues in an example dataset.

## 1. Installation

First, you need to install the `selfclean-audio` library. You can do this using pip:

```bash
pip install selfclean-audio
```

## 2. Prepare your dataset

For this tutorial, we will use a small example dataset of audio files. You can create a directory with a few audio files (e.g., `.wav` files) and a `metadata.csv` file with the following format:

```
filename,label
audio1.wav,cat
audio2.wav,dog
audio3.wav,cat
...
```

## 3. Run issue detection

Now, you can use the `selfclean-audio` command-line interface (CLI) to detect issues in your dataset. Heres an example command to detect near-duplicates:

```bash
python -m selfclean_audio \
  --config config/templates/file_template.py \
  --output-dir outputs/my_dataset_duplicates \
  datamodule.train_dataset.root=/path/to/your/dataset \
  datamodule.train_dataset.meta_path=/path/to/your/metadata.csv \
  MODEL_TYPE=beats \
  ISSUE_TYPE=duplicates \
  near_duplicate_method=embedding_distance
```

Replace `/path/to/your/dataset` and `/path/to/your/metadata.csv` with the actual paths to your dataset directory and metadata file.

## 4. Analyze the results

The results will be saved in the `outputs/my_dataset_duplicates` directory. You can find the following files:

-   `Score-duplicates.csv`: A CSV file with the duplicate scores for each pair of audio files.
-   `config.log`: A log file with the configuration used for the run.

You can then use the scores in `Score-duplicates.csv` to identify and remove the near-duplicates from your dataset.
