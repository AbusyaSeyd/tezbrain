"""Shared helpers for organizing training artifacts."""
from pathlib import Path
from typing import Dict


def prepare_artifact_dirs(base_dir: str = "artifacts") -> Dict[str, Path]:
    """
    Create and return common artifact directories.
    
    The following subfolders are created under ``base_dir``:
    - models   : checkpoints (.pth)
    - metrics  : JSON/CSV summaries
    - plots    : figures and visualizations
    - logs     : log files
    """
    base = Path(base_dir)
    dirs: Dict[str, Path] = {
        "base": base,
        "models": base / "models",
        "metrics": base / "metrics",
        "plots": base / "plots",
        "logs": base / "logs",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


