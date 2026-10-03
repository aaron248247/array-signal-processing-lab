"""Saving and loading experiment results.

Each experiment writes two files into ``results/``:

* ``<name>.json`` - configuration, random seed, scalar metrics, timing and
  software versions (human-readable).
* ``<name>.npz``  - numerical arrays needed to regenerate the figures.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, complex):
        return {"re": obj.real, "im": obj.imag}
    if isinstance(obj, float) and not np.isfinite(obj):
        return str(obj)
    return obj


def environment_info() -> dict:
    import scipy

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }


def save_results(
    name: str,
    config: dict,
    metrics: dict,
    arrays: dict | None = None,
    elapsed_s: float | None = None,
    results_dir: Path | None = None,
) -> tuple[Path, Path | None]:
    """Write ``<name>.json`` (and ``<name>.npz`` if ``arrays``) to ``results_dir``."""
    out = Path(results_dir) if results_dir is not None else RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "experiment": name,
        "config": config,
        "metrics": metrics,
        "elapsed_s": elapsed_s,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": environment_info(),
    }
    jpath = out / f"{name}.json"
    jpath.write_text(json.dumps(_jsonable(record), indent=2, sort_keys=False) + "\n")
    npath = None
    if arrays:
        npath = out / f"{name}.npz"
        np.savez_compressed(npath, **arrays)
    return jpath, npath


def load_results(name: str, results_dir: Path | None = None) -> tuple[dict, dict]:
    """Return ``(record, arrays)`` for experiment ``name``."""
    out = Path(results_dir) if results_dir is not None else RESULTS_DIR
    record = json.loads((out / f"{name}.json").read_text())
    npath = out / f"{name}.npz"
    arrays = {}
    if npath.exists():
        with np.load(npath) as z:
            arrays = {k: z[k] for k in z.files}
    return record, arrays
