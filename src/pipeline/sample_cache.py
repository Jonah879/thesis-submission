"""
sample_cache.py — Save and load the full sampled dataset (with retrieved tweets).

Public API:
    save(tasks, cache_path)  — Save the tasks list to a pickle file.
    load(cache_path) -> list[dict]  — Load the tasks list from a pickle file.
    save_checkpoint(state, path)  — Save an arbitrary resume-state dict.
    load_checkpoint(path) -> dict  — Load a resume-state dict.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def save(tasks: List[dict], cache_path: Path) -> None:
    """Save the full tasks list (after Phase 2) to a pickle file."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(tasks, f)
    logger.info("Sample cache saved: %d candidates → %s", len(tasks), cache_path)


def load(cache_path: Path) -> List[dict]:
    """Load the tasks list from a pickle file."""
    cache_path = Path(cache_path)
    with open(cache_path, "rb") as f:
        tasks = pickle.load(f)
    logger.info("Sample cache loaded: %d candidates from %s", len(tasks), cache_path)
    return tasks


def save_checkpoint(state: Dict[str, Any], path: Path) -> None:
    """
    Save an in-progress resume state (e.g. from build_sample_cache.py) to a
    pickle file. Overwrites the previous checkpoint each call, so a crash
    between calls loses at most the work since the last save.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(state, f)
    tmp_path.replace(path)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    """Load a resume state previously written by save_checkpoint()."""
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)
