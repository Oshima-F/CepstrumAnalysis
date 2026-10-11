"""Input validation: errors for invalid input, warnings for suspicious signals (silence, clipping).

The warnings are subclasses of ``UserWarning``, so they can be silenced or turned into errors with
the standard ``warnings`` machinery::

    warnings.simplefilter("ignore", cepstrum.SilenceWarning)
    warnings.simplefilter("error", cepstrum.ClippingWarning)
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

CLIP_MIN_RUN = 3     # consecutive samples stuck at the signal's own max (or min) that count as clipping
CLIP_RTOL = 1e-9     # "stuck" = within this fraction of the peak amplitude of the extreme value


class CepstrumWarning(UserWarning):
    """Base class of the warnings issued for suspicious input."""


class SilenceWarning(CepstrumWarning):
    """The signal is digital silence (all zeros) or constant: its cepstrum carries no information."""


class ClippingWarning(CepstrumWarning):
    """The signal appears clipped (runs of samples stuck at its extreme value)."""


def check_fs(fs) -> float:
    """Validate a sampling frequency.

    Args:
        fs: Sampling frequency [Hz]. Required.

    Returns:
        ``fs`` as a float.

    Raises:
        ValueError: If ``fs`` is not a finite positive number.
    """
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be a finite positive number, got {fs}")
    return fs


def _stuck_mask(x: np.ndarray, target: float, tol: float) -> np.ndarray:
    """Samples in runs (length >= CLIP_MIN_RUN) of values within ``tol`` of ``target``."""
    near = np.abs(x - target) <= tol
    run = near.copy()
    for k in range(1, CLIP_MIN_RUN):
        run[: len(x) - k] &= near[k:]
        run[len(x) - k:] = False
    stuck = run.copy()  # every sample belonging to a qualifying run
    for k in range(1, CLIP_MIN_RUN):
        stuck[k:] |= run[: len(x) - k]
    return stuck


def _clipped_fraction(x: np.ndarray) -> float:
    """Fraction of samples stuck at the positive or the negative extreme of ``x`` (each tested alone).

    Runs at exactly zero are ignored: zero padding and silence are not clipping.
    """
    tol = CLIP_RTOL * float(np.max(np.abs(x)))
    stuck = np.zeros(len(x), dtype=bool)
    for target in (float(x.max()), float(x.min())):
        if target != 0.0:
            stuck |= _stuck_mask(x, target, tol)
    return float(stuck.mean())


def prepare_signal(x, stacklevel: int = 3) -> np.ndarray:
    """Validate a time signal, warn if it looks silent or clipped, and return it as mono.

    Multichannel input is checked per channel for clipping and then averaged over channels.

    Args:
        x: Signal ``[N]`` or ``[ch, N]``. Required.
        stacklevel: Stack level the warnings point to. Default 3 (the caller of the public
            function that called this).

    Returns:
        Float array ``[N]``.

    Raises:
        ValueError: Complex or non-numeric data, a shape other than ``[N]`` / ``[ch, N]``, an
            empty signal, or NaN / inf samples.
        SilenceWarning: (warning) Digital silence or a constant signal.
        ClippingWarning: (warning) Runs of samples stuck at the extreme value.
    """
    arr = np.asarray(x)
    if np.iscomplexobj(arr):
        raise ValueError("the time signal must be real (got complex data)")
    try:
        arr = arr.astype(float)
    except (TypeError, ValueError) as e:
        raise ValueError(f"the time signal must be numeric: {e}") from None
    if arr.ndim not in (1, 2):
        raise ValueError("signal must be 1-D [N] or 2-D [ch, N]")
    if arr.shape[-1] == 0:
        raise ValueError("the signal is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("the signal contains NaN or inf")

    mono = arr if arr.ndim == 1 else arr.mean(axis=0)
    if np.ptp(mono) == 0.0:
        warnings.warn("the signal is silent or constant; its cepstrum carries no information",
                      SilenceWarning, stacklevel=stacklevel)
        return mono
    for ch, sig in enumerate(arr.reshape(-1, arr.shape[-1])):
        frac = _clipped_fraction(sig)
        if frac > 0.0:
            where = "" if arr.ndim == 1 else f" in channel {ch}"
            warnings.warn(f"the signal appears clipped{where}: {100 * frac:.2f}% of the samples are stuck "
                          f"at its maximum or minimum ({sig.min():.6g} .. {sig.max():.6g})", ClippingWarning,
                          stacklevel=stacklevel)
            break
    return mono


def warn_if_silent_spectrum(mag: np.ndarray, stacklevel: int = 3) -> None:
    """Warn if a magnitude spectrum is identically zero.

    Args:
        mag: Magnitude spectrum. Required.
        stacklevel: Stack level the warning points to. Default 3.
    """
    if not np.any(mag > 0):
        warnings.warn("the spectrum is identically zero; its cepstrum carries no information",
                      SilenceWarning, stacklevel=stacklevel)
