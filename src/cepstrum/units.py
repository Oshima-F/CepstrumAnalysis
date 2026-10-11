"""Quefrency units.

Quefrency is the dual of frequency, so [s] == [Hz^-1]: a coefficient at quefrency q is a
cos(2*pi*q*f) ripple in the log spectrum, i.e. q ripples (cycles) per Hz, with a ripple period
of 1/q Hz.  With SI prefixes 1 kHz^-1 == 1 ms, so the numbers are the familiar ones, read as
"how many spectral peaks per kHz".  (Cycles, not radians: no factor 2*pi.)

    q = 8 ms = 8 kHz^-1  ->  8 peaks per kHz  ->  ripple period 125 Hz

Units: ``"s"``, ``"Hz^-1"`` (same as ``"s"``), ``"ms"``, ``"kHz^-1"`` (same as ``"ms"``) and
``"period_Hz"`` (ripple period 1/|q|).
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

# unit name -> multiplier from seconds (= Hz^-1)
_SCALE = {"s": 1.0, "Hz^-1": 1.0, "ms": 1e3, "kHz^-1": 1e3}
UNITS = tuple(_SCALE) + ("period_Hz",)


def from_seconds(q, unit: str) -> np.ndarray:
    """Convert quefrency in seconds to another unit.

    Args:
        q: Quefrency [s] (array-like or scalar). Required.
        unit: One of ``UNITS``. Required.

    Returns:
        Float array in ``unit``. For ``"period_Hz"`` this is ``1/|q|``, ``inf`` at ``q = 0``.

    Raises:
        ValueError: Unknown ``unit``.
    """
    q = np.asarray(q, dtype=float)
    if unit == "period_Hz":
        out = np.full(q.shape, np.inf)
        np.divide(1.0, np.abs(q), out=out, where=q != 0)
        return out
    if unit not in _SCALE:
        raise ValueError(f"unknown unit {unit!r}; choose from {UNITS}")
    return q * _SCALE[unit]


def to_seconds(
    value: Optional[Union[float, np.ndarray]] = None,
    *,
    khz_inv: Optional[Union[float, np.ndarray]] = None,
    period_hz: Optional[Union[float, np.ndarray]] = None,
):
    """Quefrency in seconds from exactly one of three equivalent forms.

    Args:
        value: Quefrency [s = Hz^-1].
        khz_inv: Quefrency in kHz^-1 (= ms numerically).
        period_hz: Ripple period [Hz]; the quefrency is its reciprocal.

    Returns:
        Float array (0-d for scalar input) in seconds.

    Raises:
        ValueError: If not exactly one of the three arguments is given, ``period_hz`` is not
            positive, or the resulting quefrency is negative or not finite.
    """
    given = [v for v in (value, khz_inv, period_hz) if v is not None]
    if len(given) != 1:
        raise ValueError("specify exactly one of: seconds, khz_inv, period_hz")
    if period_hz is not None:
        period = np.asarray(period_hz, dtype=float)
        if not np.all(np.isfinite(period)) or np.any(period <= 0):
            raise ValueError("period_hz must be finite and positive")
        q = 1.0 / period
    elif khz_inv is not None:
        q = np.asarray(khz_inv, dtype=float) / 1e3
    else:
        q = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(q)) or np.any(q < 0):
        raise ValueError("a quefrency must be finite and non-negative")
    return q


def quefrency_axis(ceps, unit: str = "kHz^-1") -> np.ndarray:
    """Signed quefrency axis of a cepstrum, laid out like ``data`` (negative quefrency at the end).

    Args:
        ceps: A :class:`Cepstrum` or :class:`Cepstrogram`. Required.
        unit: One of ``UNITS``. Default ``"kHz^-1"``.

    Returns:
        Float array of shape ``[n_fft]``.
    """
    return from_seconds(ceps.quefrency, unit)
