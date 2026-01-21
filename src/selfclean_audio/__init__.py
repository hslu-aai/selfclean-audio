# Copyright (c) Lucerne University of Applied Sciences and Arts.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""Top-level API
=============

This module provides access to the core functionality of the `SelfCleanAudio` package.

.. data:: __version__
    :type: str

    The version number of the package, as calculated by ``setuptools_scm``
    (https://github.com/pypa/setuptools_scm). This will be automatically
    updated based on the current version control system (e.g., Git tags).
"""

from . import config, datasets, selfclean_audio, utils
from ._version import __version__

__all__ = [
    "__version__",
    "selfclean_audio",
    "datasets",
    "config",
    "utils",
]
