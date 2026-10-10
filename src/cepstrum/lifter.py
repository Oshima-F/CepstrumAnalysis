"""Lifters: weights over (signed) quefrency in seconds, applied to Cepstrum / Cepstrogram.

Quefrency arguments can be given in seconds (= Hz^-1) positionally, or by keyword as
``khz_inv`` (kHz^-1, = ms numerically) or ``period_hz`` (ripple period in Hz, = 1/quefrency).

Inputs
    lowpass(cutoff | khz_inv= | period_hz=)       one value; keep |q| <= cutoff
    highpass(cutoff | khz_inv= | period_hz=)      one value; keep |q| >= cutoff (drops c[0])
    bandpass(low, high | khz_inv=(a,b) | period_hz=(a,b))
    notch(center | khz_inv= | period_hz=, width | width_khz_inv=)
    apply_lifter(ceps, lifter)    ceps: Cepstrum or Cepstrogram; lifter: callable(signed q [s]) or
                                  ndarray[n_fft]
    lifter_weights(lifter, n_fft, fs)
Outputs
    lowpass/highpass/bandpass/notch -> callable(q [s]) -> 0/1 weights (exactly one unit form allowed)
    apply_lifter                    -> new Cepstrum / Cepstrogram (data * weights)
    lifter_weights                  -> ndarray [n_fft]
Parameters
    none beyond the arguments above
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Optional, TypeVar, Union

import numpy as np

from .units import to_seconds

LifterFn = Callable[[np.ndarray], np.ndarray]
Lifter = Union[LifterFn, np.ndarray]
T = TypeVar("T")


def lowpass(cutoff: Optional[float] = None, *, khz_inv: Optional[float] = None,
            period_hz: Optional[float] = None) -> LifterFn:
    """Keep |q| <= cutoff: smooth envelope. ``period_hz=300`` keeps ripples wider than 300 Hz."""
    c = float(to_seconds(cutoff, khz_inv=khz_inv, period_hz=period_hz))
    return lambda q: (np.abs(q) <= c).astype(float)


def highpass(cutoff: Optional[float] = None, *, khz_inv: Optional[float] = None,
             period_hz: Optional[float] = None) -> LifterFn:
    """Keep |q| >= cutoff: fine structure (pitch/harmonics). Removes c[0]."""
    c = float(to_seconds(cutoff, khz_inv=khz_inv, period_hz=period_hz))
    return lambda q: (np.abs(q) >= c).astype(float)


def bandpass(low: Optional[float] = None, high: Optional[float] = None, *,
             khz_inv: Optional[tuple] = None, period_hz: Optional[tuple] = None) -> LifterFn:
    """Keep low <= |q| <= high. By keyword pass a pair, e.g. ``period_hz=(150, 100)`` (either order)."""
    if low is not None or high is not None:
        if khz_inv is not None or period_hz is not None or low is None or high is None:
            raise ValueError("give both low and high in seconds, or one keyword pair")
        lo, hi = float(low), float(high)
    else:
        pair = np.atleast_1d(to_seconds(khz_inv=khz_inv, period_hz=period_hz))
        if pair.shape != (2,):
            raise ValueError("keyword form needs a pair")
        lo, hi = float(pair.min()), float(pair.max())
    return lambda q: ((np.abs(q) >= lo) & (np.abs(q) <= hi)).astype(float)


def notch(center: Optional[float] = None, width: Optional[float] = None, *,
          khz_inv: Optional[float] = None, period_hz: Optional[float] = None,
          width_khz_inv: Optional[float] = None) -> LifterFn:
    """Remove |q| within ``width``/2 of ``center`` (width: seconds or ``width_khz_inv``)."""
    c = float(to_seconds(center, khz_inv=khz_inv, period_hz=period_hz))
    w = float(to_seconds(width, khz_inv=width_khz_inv))
    return lambda q: (np.abs(np.abs(q) - c) > w / 2).astype(float)


def lifter_weights(lifter: Lifter, n_fft: int, fs: float) -> np.ndarray:
    """Evaluate a lifter (callable of signed quefrency [s], or array of length n_fft)."""
    if callable(lifter):
        q = np.fft.fftfreq(n_fft) * n_fft / fs
        w = np.asarray(lifter(q), dtype=float)
    else:
        w = np.asarray(lifter, dtype=float)
    if w.shape != (n_fft,):
        raise ValueError(f"lifter must produce {n_fft} weights, got shape {w.shape}")
    return w


def apply_lifter(ceps: T, lifter: Lifter) -> T:
    """Multiply a :class:`Cepstrum` or :class:`Cepstrogram` by a lifter (returns a new object)."""
    w = lifter_weights(lifter, ceps.n_fft, ceps.fs)
    w = w[:, None] if ceps.data.ndim == 2 else w
    return replace(ceps, data=ceps.data * w)
