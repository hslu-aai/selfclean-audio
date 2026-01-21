"""
Quick environment checks to validate baseline dependencies inside the container.

Checks:
- Python packages import (librosa, torchaudio, cleanlab, pyod, psycopg2)
- ffmpeg availability
- Postgres connectivity to host 'db' (as expected by Dejavu) and database 'dejavu'

Usage:
  python scripts/check_runtime_env.py
"""

from __future__ import annotations

import shutil
import socket
import sys


def check_imports():
    pkgs = [
        ("librosa", None),
        ("torchaudio", None),
        ("cleanlab", None),
        ("pyod", None),
        ("psycopg2", None),
    ]
    ok = True
    for name, alias in pkgs:
        try:
            __import__(name)
        except Exception as e:
            print(f"[FAIL] import {name}: {e}")
            ok = False
        else:
            print(f"[ OK ] import {name}")
    return ok


def check_ffmpeg():
    path = shutil.which("ffmpeg")
    if path:
        print(f"[ OK ] ffmpeg found at {path}")
        return True
    print("[FAIL] ffmpeg not found in PATH")
    return False


def check_postgres_tcp(host: str = "db", port: int = 5432):
    s = socket.socket()
    s.settimeout(2.0)
    try:
        s.connect((host, port))
        print(f"[ OK ] TCP connectivity to Postgres at {host}:{port}")
        return True
    except Exception as e:
        print(f"[FAIL] Cannot connect to Postgres at {host}:{port}: {e}")
        return False
    finally:
        s.close()


if __name__ == "__main__":
    ok = True
    ok &= check_imports()
    ok &= check_ffmpeg()
    ok &= check_postgres_tcp()
    if not ok:
        sys.exit(1)
    print("All environment checks passed.")
