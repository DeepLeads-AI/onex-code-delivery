"""Puts the repo root on sys.path so ``import layer_profile...`` resolves when
pytest is run from the repo root without an editable install.

These tests never touch a database, a model or the network.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
