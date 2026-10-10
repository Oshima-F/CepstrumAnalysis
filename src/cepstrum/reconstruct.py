"""Time-signal reconstruction from (liftered) cepstra.

Inputs
    reconstruct(ceps, lifter=None, *, phase)
        ceps    Cepstrum (kind "complex" or "real"; "power" is not invertible)
        lifter  optional callable / ndarray applied before reconstruction
Outputs
    float ndarray: n_fft samples, or the original length if ``ceps.length`` is known
Parameters
    phase   "minimum" (only option so far; used for real cepstra, which carry no phase)
            complex cepstra are inverted exactly (phase, delay and sign are stored)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .core import Cepstrum
from .lifter import Lifter, apply_lifter


def _minimum_phase_log_spectrum(c: np.ndarray) -> np.ndarray:
    """Fold a real cepstrum into the causal (minimum-phase) complex cepstrum -> log spectrum."""
    n = c.shape[0]
    folded = np.zeros(n)
    folded[0] = c[0]
    folded[1:(n + 1) // 2] = 2 * c[1:(n + 1) // 2]
    if n % 2 == 0:
        folded[n // 2] = c[n // 2]
    return np.fft.fft(folded)


def reconstruct(ceps: Cepstrum, lifter: Optional[Lifter] = None, *, phase: str = "minimum") -> np.ndarray:
    """Rebuild a time signal from a cepstrum, optionally liftered first.

    complex -> exact inverse (phase, delay and sign are stored in ``ceps``).
    real    -> phase is not available and is supplied by ``phase`` (only "minimum" for now).
    power   -> not invertible.
    """
    if lifter is not None:
        ceps = apply_lifter(ceps, lifter)
    if ceps.kind == "power":
        raise ValueError("power cepstrum is not invertible")
    if ceps.kind == "complex":
        x = np.fft.ifft(ceps.sign * np.exp(np.fft.fft(ceps.data))).real
        x = np.roll(x, ceps.delay)
    elif ceps.kind == "real":
        if phase != "minimum":
            raise NotImplementedError(f"phase={phase!r} is not implemented")
        x = np.fft.ifft(np.exp(_minimum_phase_log_spectrum(ceps.data))).real
    else:
        raise ValueError(f"unknown kind: {ceps.kind!r}")
    return x if ceps.length is None else x[: ceps.length]
