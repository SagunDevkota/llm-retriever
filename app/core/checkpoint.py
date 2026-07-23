"""Atomic, resumable persistence of the ``nodes_by_id`` tree to JSON."""

import json
from pathlib import Path

from core.config import CHECKPOINT_FILE


def save_checkpoint(nodes_by_id, path: str = CHECKPOINT_FILE):
    temp_file = path + ".tmp"

    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(nodes_by_id, f, ensure_ascii=False, indent=2)
        # Atomic replace so a crash mid-write can't corrupt the checkpoint.
        Path(temp_file).replace(path)
    except OSError as e:
        raise OSError(f"Failed to write checkpoint to {path}: {e}") from e


def load_checkpoint(path: str = CHECKPOINT_FILE):
    if not Path(path).exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
