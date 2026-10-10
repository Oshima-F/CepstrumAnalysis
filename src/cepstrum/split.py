"""Low / high quefrency decomposition shared by the analyze_* functions.

``low`` is the cepstrum multiplied by a lowpass lifter (spectral envelope, includes c[0]) and
``high`` is the complement ``full - low`` (excitation / fine structure), so ``low + high == full``
exactly, also for fractional (tapered) lifters.

Inputs
    make_lifter(lifter, cutoff, khz_inv, period_hz)    lifter OR exactly one cutoff form, or none
    split(full, lifter)                                full: Cepstrum / Cepstrogram
Outputs
    Decomposition(full, low, high, lifter)
        part(name) / log_spectrum(part) [dB, one-sided] / freqs [Hz] / reconstruct(part)
Parameters
    part    "full" | "low" | "high"       (log_spectrum / reconstruct; default "full")
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Generic, Optional, TypeVar, Union

import numpy as np

from .cepstrogram import Cepstrogram, reconstruct_cepstrogram
from .core import Cepstrum
from .lifter import Lifter, apply_lifter, lowpass
from .reconstruct import reconstruct

T = TypeVar("T", bound=Union[Cepstrum, Cepstrogram])
PARTS = ("full", "low", "high")


def make_lifter(lifter: Optional[Lifter] = None, cutoff: Optional[float] = None,
                khz_inv: Optional[float] = None, period_hz: Optional[float] = None) -> Optional[Lifter]:
    """Resolve the lifter: an explicit ``lifter``, or a lowpass from one of cutoff / khz_inv / period_hz."""
    spec = [v for v in (cutoff, khz_inv, period_hz) if v is not None]
    if lifter is not None and spec:
        raise ValueError("give either lifter or a cutoff (cutoff / khz_inv / period_hz), not both")
    if lifter is not None:
        return lifter
    if spec:
        return lowpass(cutoff, khz_inv=khz_inv, period_hz=period_hz)
    return None


@dataclass(frozen=True)
class Decomposition(Generic[T]):
    """Result of an ``analyze_*`` call. ``full``/``low``/``high`` are Cepstrum or Cepstrogram objects.

    With no lifter given, ``low`` equals ``full`` and ``high`` is all zeros.
    """

    full: T
    low: T
    high: T
    lifter: Optional[Lifter] = None

    def part(self, name: str) -> T:
        if name not in PARTS:
            raise ValueError(f"part must be one of {PARTS}")
        return getattr(self, name)

    @property
    def freqs(self) -> np.ndarray:
        """One-sided frequency axis [Hz] for :meth:`log_spectrum`."""
        return np.fft.rfftfreq(self.full.n_fft, 1 / self.full.fs)

    def log_spectrum(self, part: str = "full") -> np.ndarray:
        """One-sided log-magnitude spectrum [dB] carried by ``part`` (real/complex kinds only).

        ``low`` is the spectral envelope; ``high`` is the fine-structure deviation around it.
        Shape ``[freq]`` for a Cepstrum, ``[freq, frame]`` for a Cepstrogram.
        """
        c = self.part(part)
        if c.kind == "power":
            raise ValueError("a power cepstrum carries no invertible log spectrum")
        ln_mag = np.fft.fft(c.data, axis=0).real[: self.full.n_fft // 2 + 1]
        return 20 / np.log(10) * ln_mag

    def reconstruct(self, part: str = "full", **kw) -> np.ndarray:
        """Time signal of ``part`` (``phase=`` is forwarded for real cepstra).

        complex kind: ``low`` and ``high`` are homomorphic components whose (circular, ``n_fft``-point)
        convolution is ``full``; the delay and sign are carried by ``low``, and neither part is
        truncated to the input length.
        """
        c = self.part(part)
        if isinstance(c, Cepstrogram):
            return reconstruct_cepstrogram(c, **kw)
        return reconstruct(c, **kw)


def split(full: T, lifter: Optional[Lifter]) -> Decomposition[T]:
    if lifter is None:
        return Decomposition(full, full, replace(full, data=np.zeros_like(full.data)), None)
    low = apply_lifter(full, lifter)
    high = replace(full, data=full.data - low.data)
    if full.kind == "complex":
        # Components are homomorphic factors (convolution == full): the pure delay and sign belong to
        # ``low`` only, and neither is truncated to the input length (a factor can be longer than x).
        if isinstance(full, Cepstrogram):
            high = replace(high, delays=np.zeros_like(full.delays), signs=np.ones_like(full.signs))
        else:
            low = replace(low, length=None)
            high = replace(high, delay=0, sign=1, length=None)
    return Decomposition(full, low, high, lifter)
