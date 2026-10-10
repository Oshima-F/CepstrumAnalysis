"""Small helpers shared across the package.

Inputs
    to_mono(x)          x: array-like, [N] or [ch, N]
    get_window(w, n)    w: "hann" | "hamming" | "rect" | None | ndarray[n]; n: frame length
Outputs
    to_mono             float ndarray [N] (channels averaged)
    get_window          float ndarray [n], periodic window (None / "rect" -> ones)
Parameters
    none (WindowSpec is the accepted window type alias)
"""
from __future__ import annotations

from typing import Union

import numpy as np

WindowSpec = Union[str, np.ndarray, None]


def to_mono(x) -> np.ndarray:
    """Return a 1-D float array. Multichannel input ``[ch, N]`` is averaged over channels."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        return x.mean(axis=0)
    raise ValueError("signal must be 1-D [N] or 2-D [ch, N]")


def get_window(window: WindowSpec, n: int) -> np.ndarray:
    """Periodic window of length ``n`` (None / "rect" -> ones)."""
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
