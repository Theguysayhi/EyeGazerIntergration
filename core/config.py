"""
core/config.py
Shared configuration loader.

All modules inside the `core` package import from here to ensure they
always find config.ini at the project root, regardless of how deep in
the package hierarchy they live.
"""

from __future__ import annotations

import configparser
import os

# Project root is one level up from this file (core/config.py → root)
ROOT_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH: str = os.path.join(ROOT_DIR, "config.ini")


def load_config() -> configparser.ConfigParser:
    """Return a ConfigParser loaded from the project-root config.ini."""
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";",))
    cfg.read(CONFIG_PATH)
    return cfg


# Module-level singleton – cheap to share across imports
cfg: configparser.ConfigParser = load_config()
