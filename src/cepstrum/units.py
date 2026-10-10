"""Quefrency units.

Quefrency is the dual of frequency, so [s] == [Hz^-1]: a coefficient at quefrency q is a
cos(2*pi*q*f) ripple in the log spectrum, i.e. q ripples (cycles) per Hz, with a ripple period
of 1/q Hz.  With SI prefixes 1 kHz^-1 == 1 ms, so the numbers are the familiar ones, read as
"how many spectral peaks per kHz".  (Cycles, not radians: no factor 2*pi.)

    q = 8 ms = 8 kHz^-1  ->  8 peaks per kHz  ->  ripple period 125 Hz

Inputs
    from_seconds(q, unit)               q: quefrency [s], array-like
    to_seconds(value | khz_inv= | period_hz=)    exactly one given
    quefrency_axis(ceps, unit)          ceps: Cepstrum or Cepstrogram
    unit_label(unit)
Outputs
    from_seconds / quefrency_axis -> ndarray in ``unit``; to_seconds -> seconds; unit_label -> str
Parameters
    unit    "s" | "Hz^-1" | "ms" | "kHz^-1" | "period_Hz"  (quefrency_axis default "kHz^-1")
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

# unit name -> multiplier from seconds (= Hz^-1)
_SCALE = {"s": 1.0, "Hz^-1": 1.0, "ms": 1e3, "kHz^-1": 1e3}
UNITS = tuple(_SCALE) + ("period_Hz",)

_LABEL = {
    "s": "quefrency [s]",
    "Hz^-1": "quefrency [Hz$^{-1}$]",
    "ms": "quefrency [ms]",
    "kHz^-1": "quefrency [kHz$^{-1}$]  (= peaks per kHz)",
    "period_Hz": "ripple period [Hz]",
}


def from_seconds(q, unit: str) -> np.ndarray:
    """Convert quefrency in seconds to ``unit`` ("period_Hz" -> 1/|q|, inf at q=0)."""
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
    """Quefrency in seconds from exactly one of: seconds, ``khz_inv`` [kHz^-1], ``period_hz`` [Hz]."""
    given = [v for v in (value, khz_inv, period_hz) if v is not None]
    if len(given) != 1:
        raise ValueError("specify exactly one of: seconds, khz_inv, period_hz")
    if khz_inv is not None:
        return np.asarray(khz_inv, dtype=float) / 1e3
    if period_hz is not None:
        return 1.0 / np.asarray(period_hz, dtype=float)
    return np.asarray(value, dtype=float)


def quefrency_axis(ceps, unit: str = "kHz^-1") -> np.ndarray:
    """Signed quefrency axis of a Cepstrum / Cepstrogram in ``unit`` (layout matches ``data``)."""
    return from_seconds(ceps.quefrency, unit)


def unit_label(unit: str) -> str:
    """Axis label for ``unit``."""
    if unit not in _LABEL:
        raise ValueError(f"unknown unit {unit!r}; choose from {UNITS}")
    return _LABEL[unit]
