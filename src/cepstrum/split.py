"""Low / high quefrency decomposition shared by the analyze_* functions.

``low`` is the cepstrum multiplied by a lowpass lifter (spectral envelope, includes c[0]) and
``high`` is the complement ``full - low`` (excitation / fine structure), so ``low + high == full``
exactly, also for fractional (tapered) lifters.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Generic, Optional, TypeVar, Union

import numpy as np

from .cepstrogram import Cepstrogram, reconstruct_cepstrogram
from .core import DB_PER_NEPER, Cepstrum
from .lifter import Lifter, apply_lifter, lowpass

T = TypeVar("T", bound=Union[Cepstrum, Cepstrogram])
PARTS = ("full", "low", "high")


def make_lifter(lifter: Optional[Lifter] = None, cutoff: Optional[float] = None,
                khz_inv: Optional[float] = None, period_hz: Optional[float] = None) -> Optional[Lifter]:
    """Resolve the lifter: an explicit one, or a lowpass built from a cutoff.

    Args:
        lifter: Callable of signed quefrency [s], or weight array. Default ``None``.
        cutoff: Lowpass cutoff [s = Hz^-1]. Default ``None``.
        khz_inv: Lowpass cutoff in kHz^-1. Default ``None``.
        period_hz: Lowpass cutoff as a ripple period [Hz]. Default ``None``.

    Returns:
        The lifter, or ``None`` if nothing was given (no split).

    Raises:
        ValueError: If ``lifter`` is combined with a cutoff, or more than one cutoff form is given.
    """
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
    """Result of an ``analyze_*`` call. With no lifter, ``low`` equals ``full`` and ``high`` is zero.

    Args:
        full: The complete :class:`Cepstrum` or :class:`Cepstrogram`.
        low: ``full`` times the lowpass lifter (envelope, includes c[0]).
        high: ``full - low`` (excitation / fine structure).
        lifter: The lifter used, or ``None``. Default ``None``.
    """

    full: T
    low: T
    high: T
    lifter: Optional[Lifter] = None

    def part(self, name: str) -> T:
        """Return one component by name.

        Args:
            name: ``"full"``, ``"low"`` or ``"high"``. Required.

        Raises:
            ValueError: Unknown name.
        """
        if name not in PARTS:
            raise ValueError(f"part must be one of {PARTS}")
        return getattr(self, name)

    @property
    def freqs(self) -> np.ndarray:
        """One-sided frequency axis [Hz] for :meth:`log_spectrum`."""
        return np.fft.rfftfreq(self.full.n_fft, 1 / self.full.fs)

    def log_spectrum(self, part: str = "full") -> np.ndarray:
        """One-sided log-magnitude spectrum [dB] carried by a component.

        ``low`` is the spectral envelope; ``high`` is the fine-structure deviation around it.
        The result is in dB whether or not the cepstrum was computed with ``db=True``.

        Args:
            part: ``"full"``, ``"low"`` or ``"high"``. Default ``"full"``.

        Returns:
            Array of shape ``[freq]`` for a Cepstrum, ``[freq, frame]`` for a Cepstrogram; the
            frequency axis is :attr:`freqs`.

        Raises:
            ValueError: Unknown ``part``.
        """
        c = self.part(part)
        log_mag = np.fft.fft(c.data, axis=0).real[: self.full.n_fft // 2 + 1]
        return log_mag if c.db else DB_PER_NEPER * log_mag  # a db=True cepstrum is already in dB

    def reconstruct(self, part: str = "full") -> np.ndarray:
        """Time signal of a component of a cepstrogram, with the original phase.

        ``"full"`` is the original signal. ``"low"`` has the (envelope) magnitude of the low part
        and the original phase (see :func:`reconstruct_cepstrogram`). ``"high"`` is the
        complement ``full - low`` in the time domain, so ``low + high == full``.

        Args:
            part: ``"full"``, ``"low"`` or ``"high"``. Default ``"full"``.

        Returns:
            Float array of the original signal length.

        Raises:
            ValueError: Not a cepstrogram (a single cepstrum has no frames to take the phase
                from), no stored spectrogram, or an unknown ``part``.
            ReconstructionWarning: (warning) The window and hop leave samples without window
                energy (see :func:`reconstruct_cepstrogram`).
        """
        self.part(part)  # validates the name
        if not isinstance(self.full, Cepstrogram):
            raise ValueError("reconstruction needs a cepstrogram (analyze_cepstrogram): the original phase "
                             "is taken from its complex spectrogram")
        if part == "high":
            return reconstruct_cepstrogram(self.full) - reconstruct_cepstrogram(self.low)
        return reconstruct_cepstrogram(self.part(part))


def split(full: T, lifter: Optional[Lifter]) -> Decomposition[T]:
    """Split a cepstrum into low / high quefrency components.

    Args:
        full: A :class:`Cepstrum` or :class:`Cepstrogram`. Required.
        lifter: Lowpass lifter, or ``None`` for no split. Required (pass ``None`` explicitly).

    Returns:
        A :class:`Decomposition` with ``low + high == full``.
    """
    if lifter is None:
        return Decomposition(full, full, replace(full, data=np.zeros_like(full.data)), None)
    low = apply_lifter(full, lifter)
    return Decomposition(full, low, replace(full, data=full.data - low.data), lifter)
