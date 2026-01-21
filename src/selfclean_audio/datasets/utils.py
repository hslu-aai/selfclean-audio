# Copyright (c) Lucerne University of Applied Sciences and Arts.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os

import torch


def resize_waveform(waveform: torch.Tensor, target_length: int):
    """
    Resize a waveform tensor to the target length by trimming or padding.

    Args:
        waveform (torch.Tensor): Input waveform tensor of shape (channels, length).
        target_length (int): Desired length of the output waveform.

    Returns:
        torch.Tensor: Resized waveform tensor with the specified length.
    """
    length = waveform.shape[1]
    if length < target_length:
        # Pad to the target length
        padding = target_length - length
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    elif length > target_length:
        # Trim to the target length
        waveform = waveform[:, :target_length]
    return waveform


def fast_scandir(path: str, exts: list[str], recursive: bool = False):
    """
    Quickly scan a directory for files with specified extensions.
    From github.com/drscotthawley/aeiou/blob/main/aeiou/core.py

    Args:
        path (str): Directory path to scan.
        exts (list[str]): List of file extensions to filter (e.g., ['.wav', '.mp3']).
        recursive (bool): If True, scan subdirectories recursively. Defaults to False.

    Returns:
        tuple[list[str], list[str]]: A tuple containing a list of subfolder paths
        and a list of matched file paths.
    """
    subfolders, files = [], []

    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(path):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    if os.path.splitext(f.name)[1].lower() in exts:
                        files.append(f.path)
            except Exception:
                pass
    except Exception:
        pass

    if recursive:
        for path in list(subfolders):
            sf, f = fast_scandir(path, exts, recursive=recursive)
            subfolders.extend(sf)
            files.extend(f)  # type: ignore

    return subfolders, files
