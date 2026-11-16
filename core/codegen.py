"""
codegen.py — All Stages Generated-Code Location
===============================================
Every acados solver writes its generated C code and its JSON description
under build/ (gitignored) instead of the repo root, so the root holds only
sources. Paths are relative to the working directory, which is the repo root
(run.sh / python -m from there).
"""

import os

BUILD_DIR = 'build'


def build_path(name: str) -> str:
    """build/<name>; creates build/ so acados can write its JSON there."""
    os.makedirs(BUILD_DIR, exist_ok=True)
    return os.path.join(BUILD_DIR, name)
