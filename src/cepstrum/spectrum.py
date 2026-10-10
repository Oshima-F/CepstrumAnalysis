"""Spectrum container that normalises complex/amplitude/power, one/two-sided input.

Inputs
    Spectrum(data, fs, n_fft, kind, sided)
        data    1-D array: X(k) / |X(k)| / |X(k)|^2, linear units (no dB)
        fs      sampling frequency [Hz]
        n_fft   FFT length of the original (two-sided) spectrum
        kind    "complex" | "amplitude" | "power"      (default "complex")
        sided   "one" (n_fft//2+1 points) | "two" (n_fft points)   (default "two")
    Spectrum.from_signal(x, fs, n_fft=None)    x: [N] or [ch, N]; n_fft defaults to N
Outputs
    Spectrum.to_two_sided() -> complex ndarray [n_fft]; amplitude/power get zero phase
    Spectrum.has_phase      -> True only for kind="complex"
Parameters
    No scaling correction is applied for one-sided spectra (data is used as |X| as is).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Kind = Literal["complex", "amplitude", "power"]
Sided = Literal["one", "two"]


@dataclass(frozen=True)
class Spectrum:
    """Frequency-domain input.

    ``data`` is taken as X(k) (complex), |X(k)| (amplitude) or |X(k)|^2 (power) as is;
    no scaling correction is applied for one-sided spectra.
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
        expected = self.n_fft // 2 + 1 if self.sided == "one" else self.n_fft
        if data.shape[0] != expected:
            raise ValueError(f"{self.sided}-sided spectrum needs {expected} points, got {data.shape[0]}")
        object.__setattr__(self, "data", data)

    @property
    def has_phase(self) -> bool:
        return self.kind == "complex"

    def to_two_sided(self) -> np.ndarray:
        """Two-sided complex spectrum. Amplitude/power become non-negative reals (zero phase)."""
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
        from .utils import to_mono

        x = to_mono(x)
        n_fft = len(x) if n_fft is None else n_fft
        return cls(np.fft.fft(x, n_fft), fs, n_fft, "complex", "two")
