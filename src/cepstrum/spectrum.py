"""Spectrum container that normalises complex/amplitude/power, one/two-sided input."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .validation import check_fs

Kind = Literal["complex", "amplitude", "power"]
Sided = Literal["one", "two"]


@dataclass(frozen=True)
class Spectrum:
    """Frequency-domain input.

    No scaling correction is applied for one-sided spectra: ``data`` is used as given.

    Args:
        data: 1-D array of X(k) (complex), |X(k)| (amplitude) or |X(k)|^2 (power), in linear
            units (not dB). Length ``n_fft // 2 + 1`` if ``sided="one"``, else ``n_fft``.
        fs: Sampling frequency [Hz]. Required.
        n_fft: FFT length of the original two-sided spectrum. Required.
        kind: ``"complex"``, ``"amplitude"`` or ``"power"``. Default ``"complex"``.
        sided: ``"one"`` or ``"two"``. Default ``"two"``.

    Raises:
        ValueError: Unknown ``kind`` / ``sided``, non-1-D ``data``, NaN / inf in ``data``, an
            invalid ``fs`` or ``n_fft``, or a length that does not match ``n_fft`` and ``sided``.
    """

    data: np.ndarray
    fs: float
    n_fft: int
    kind: Kind = "complex"
    sided: Sided = "two"

    def __post_init__(self):
        data = np.asarray(self.data)
        if data.ndim != 1:
            raise ValueError("spectrum data must be 1-D")
        if self.kind not in ("complex", "amplitude", "power"):
            raise ValueError(f"unknown kind: {self.kind!r}")
        if self.sided not in ("one", "two"):
            raise ValueError(f"unknown sided: {self.sided!r}")
        check_fs(self.fs)
        if self.n_fft < 1:
            raise ValueError("n_fft must be >= 1")
        if not np.all(np.isfinite(data)):
            raise ValueError("spectrum data contains NaN or inf")
        expected = self.n_fft // 2 + 1 if self.sided == "one" else self.n_fft
        if data.shape[0] != expected:
            raise ValueError(f"{self.sided}-sided spectrum needs {expected} points, got {data.shape[0]}")
        object.__setattr__(self, "data", data)

    def to_two_sided(self) -> np.ndarray:
        """Expand to a two-sided complex spectrum.

        Amplitude/power become non-negative reals (zero phase); one-sided input is mirrored with
        complex conjugation.

        Returns:
            Complex array of shape ``[n_fft]``.
        """
        d = self.data
        if self.kind == "power":
            d = np.sqrt(d)
        d = d.astype(complex) if self.kind == "complex" else d.astype(float).astype(complex)
        if self.sided == "two":
            return d
        n_neg = self.n_fft - d.shape[0]
        return np.concatenate([d, np.conj(d[1:1 + n_neg][::-1])])

    @classmethod
    def from_signal(cls, x, fs: float, n_fft: int | None = None) -> "Spectrum":
        """Two-sided complex spectrum of a time signal.

        Args:
            x: Signal ``[N]`` or ``[ch, N]`` (channels averaged). Required.
            fs: Sampling frequency [Hz]. Required.
            n_fft: FFT length, zero-padding ``x`` if longer. Default ``N``.

        Returns:
            A complex, two-sided :class:`Spectrum`.
        """
        from .utils import to_mono

        x = to_mono(x)
        n_fft = len(x) if n_fft is None else n_fft
        return cls(np.fft.fft(x, n_fft), fs, n_fft, "complex", "two")
