from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional


def setup_logger(out_dir: str, name: str = "train") -> logging.Logger:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(out / f"{name}.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


class JsonlWriter:
    """Append-only JSONL metrics writer."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: Dict[str, Any]) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")
