"""Lifters: weights over (signed) quefrency in seconds, applied to Cepstrum / Cepstrogram.

Quefrency arguments can be given in seconds (= Hz^-1) positionally, or by keyword as
``khz_inv`` (kHz^-1, = ms numerically) or ``period_hz`` (ripple period in Hz, = 1/quefrency).
Exactly one form must be given; none of the lifter builders has a default value.
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
    """Lowpass lifter: keep ``|q| <= cutoff`` (smooth envelope).

    Args:
        cutoff: Cutoff quefrency [s = Hz^-1].
        khz_inv: Cutoff in kHz^-1 (alternative to ``cutoff``).
        period_hz: Cutoff as a ripple period [Hz]; ``period_hz=300`` keeps ripples wider than 300 Hz.

    Returns:
        Callable mapping signed quefrency [s] to 0/1 weights.

    Raises:
        ValueError: If not exactly one of the three arguments is given.
    """
    c = float(to_seconds(cutoff, khz_inv=khz_inv, period_hz=period_hz))
    return lambda q: (np.abs(q) <= c).astype(float)


def highpass(cutoff: Optional[float] = None, *, khz_inv: Optional[float] = None,
             period_hz: Optional[float] = None) -> LifterFn:
    """Highpass lifter: keep ``|q| > cutoff`` (fine structure, e.g. pitch). Removes c[0].

    The complement of :func:`lowpass` with the same cutoff: the cutoff bin belongs to the low side.

    Args:
        cutoff: Cutoff quefrency [s = Hz^-1].
        khz_inv: Cutoff in kHz^-1 (alternative to ``cutoff``).
        period_hz: Cutoff as a ripple period [Hz] (alternative to ``cutoff``).

    Returns:
        Callable mapping signed quefrency [s] to 0/1 weights.

    Raises:
        ValueError: If not exactly one of the three arguments is given.
    """
    c = float(to_seconds(cutoff, khz_inv=khz_inv, period_hz=period_hz))
    return lambda q: (np.abs(q) > c).astype(float)


def bandpass(low: Optional[float] = None, high: Optional[float] = None, *,
             khz_inv: Optional[tuple] = None, period_hz: Optional[tuple] = None) -> LifterFn:
    """Bandpass lifter: keep ``low < |q| <= high``.

    Equal to ``highpass(low)`` times ``lowpass(high)``: like the low / high split, an edge bin belongs to the
    lower-quefrency side.

    Args:
        low: Lower quefrency edge [s].
        high: Upper quefrency edge [s]. Give ``low`` and ``high`` together, or one keyword pair.
        khz_inv: Pair of edges in kHz^-1, e.g. ``(6, 10)`` (either order).
        period_hz: Pair of edges as ripple periods [Hz], e.g. ``(150, 100)`` (either order).

    Returns:
        Callable mapping signed quefrency [s] to 0/1 weights.

    Raises:
        ValueError: If the forms are mixed, incomplete, or the keyword value is not a pair.
    """
    if low is not None or high is not None:
        if khz_inv is not None or period_hz is not None or low is None or high is None:
            raise ValueError("give both low and high in seconds, or one keyword pair")
        lo, hi = float(low), float(high)
    else:
        pair = np.atleast_1d(to_seconds(khz_inv=khz_inv, period_hz=period_hz))
        if pair.shape != (2,):
            raise ValueError("keyword form needs a pair")
        lo, hi = float(pair.min()), float(pair.max())
    return lambda q: ((np.abs(q) > lo) & (np.abs(q) <= hi)).astype(float)


def notch(center: Optional[float] = None, width: Optional[float] = None, *,
          khz_inv: Optional[float] = None, period_hz: Optional[float] = None,
          width_khz_inv: Optional[float] = None) -> LifterFn:
    """Notch lifter: remove ``|q|`` within ``width / 2`` of ``center`` (a narrow band of the cepstrum).

    Args:
        center: Centre quefrency [s].
        width: Full width [s].
        khz_inv: Centre in kHz^-1 (alternative to ``center``).
        period_hz: Centre as a ripple period [Hz] (alternative to ``center``).
        width_khz_inv: Full width in kHz^-1 (alternative to ``width``).

    Returns:
        Callable mapping signed quefrency [s] to 0/1 weights.

    Raises:
        ValueError: If not exactly one centre form and one width form are given.
    """
    c = float(to_seconds(center, khz_inv=khz_inv, period_hz=period_hz))
    w = float(to_seconds(width, khz_inv=width_khz_inv))
    return lambda q: (np.abs(np.abs(q) - c) > w / 2).astype(float)


def lifter_weights(lifter: Lifter, n_fft: int, fs: float) -> np.ndarray:
    """Evaluate a lifter on the quefrency grid of an ``n_fft``-point cepstrum.

    Args:
        lifter: Callable of signed quefrency [s], or weight array of length ``n_fft``.
        n_fft: FFT length of the cepstrum.
        fs: Sampling frequency [Hz].

    Returns:
        Float array of shape ``[n_fft]``.

    Raises:
        ValueError: If the weights do not have shape ``[n_fft]``.
    """
    if callable(lifter):
        q = np.fft.fftfreq(n_fft) * n_fft / fs
        w = np.asarray(lifter(q), dtype=float)
    else:
        w = np.asarray(lifter, dtype=float)
    if w.shape != (n_fft,):
        raise ValueError(f"lifter must produce {n_fft} weights, got shape {w.shape}")
    return w


def apply_lifter(ceps: T, lifter: Lifter) -> T:
    """Multiply a cepstrum by a lifter.

    Args:
        ceps: A :class:`Cepstrum` or :class:`Cepstrogram`.
        lifter: Callable of signed quefrency [s], or weight array of length ``n_fft``.

    Returns:
        A new object of the same type with ``data`` multiplied by the weights (on every frame,
        for a cepstrogram).
    """
    w = lifter_weights(lifter, ceps.n_fft, ceps.fs)
    w = w[:, None] if ceps.data.ndim == 2 else w
    return replace(ceps, data=ceps.data * w)
