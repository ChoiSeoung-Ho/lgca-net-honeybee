"""Global seeding utilities.

`set_seed` fixes the Python, NumPy and (when available) PyTorch RNGs and puts
cuDNN into deterministic mode. Torch is imported lazily / guarded so that the
non-deep-learning scripts in this package (feature extraction, trivial
baselines, near-duplicate analysis, seed statistics, table generation) can
import this module on a machine without PyTorch installed.
"""

from __future__ import annotations

import os
import random

import numpy as np

try:  # pragma: no cover - exercised only when torch is installed
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Seed every RNG we rely on.

    Parameters
    ----------
    seed:
        Integer seed applied to ``random``, ``numpy``, ``torch`` (CPU and all
        CUDA devices) and to ``PYTHONHASHSEED``.
    deterministic:
        When True set ``cudnn.deterministic = True`` and
        ``cudnn.benchmark = False``. This is what the manuscript reports; it
        costs some throughput but removes cuDNN algorithm-selection
        nondeterminism.

    Returns
    -------
    int
        The seed that was applied (convenient for logging).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return seed


def torch_available() -> bool:
    """True when PyTorch could be imported."""
    return _HAS_TORCH


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding hook (keeps augmentation-free loading reproducible)."""
    base = np.random.get_state()[1][0]
    seed = (int(base) + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)
