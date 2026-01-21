import os
import subprocess
import sys

from selfclean_audio import __version__


def test_cli_version():
    """Goal: Check `python -m selfclean_audio --version` outputs the package __version__."""
    cmd = [sys.executable, "-m", "selfclean_audio", "--version"]
    env = os.environ.copy()
    # Avoid MKL threading issues in minimal subprocess environment
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    env.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
    out = subprocess.check_output(cmd, env=env).decode().strip()
    assert out == __version__
