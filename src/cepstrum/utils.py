"""Small helpers shared across the package (mono conversion, analysis windows)."""
from __future__ import annotations

from typing import Union

import numpy as np

WindowSpec = Union[str, np.ndarray, None]


def to_mono(x) -> np.ndarray:
    """Convert a signal to mono by averaging over channels.

    Args:
        x: Signal of shape ``[N]`` or ``[ch, N]`` (array-like).

    Returns:
        Float array of shape ``[N]``.

    Raises:
        ValueError: If ``x`` is not 1-D or 2-D.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        return x.mean(axis=0)
    raise ValueError("signal must be 1-D [N] or 2-D [ch, N]")


def get_window(window: WindowSpec, n: int) -> np.ndarray:
    """Periodic analysis window.

    Args:
        window: ``"hann"``, ``"hamming"``, ``"rect"`` or ``None`` (both rectangular), or an array
            of length ``n``. No default; callers choose (see their own defaults).
        n: Window length [samples].

    Returns:
        Float array of shape ``[n]``.

    Raises:
        ValueError: Unknown window name, or an array whose length is not ``n``.
    """
    if window is None or (isinstance(window, str) and window == "rect"):
        return np.ones(n)
    if isinstance(window, str):
        k = np.arange(n)
        if window == "hann":
            return 0.5 - 0.5 * np.cos(2 * np.pi * k / n)
        if window == "hamming":
            return 0.54 - 0.46 * np.cos(2 * np.pi * k / n)
        raise ValueError(f"unknown window: {window!r}")
    w = np.asarray(window, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"window must have length {n}")
    return w
