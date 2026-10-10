"""Average (whole-record) cepstrum: real, complex and power.

Inputs
    cepstrum(x, fs=None, *, kind, n_fft, window, eps)
        x       time signal [N] or [ch, N] (channels averaged), or a Spectrum
        fs      sampling frequency [Hz]; required for time signals, taken from a Spectrum otherwise
Outputs
    Cepstrum(data[n_fft], fs, kind, n_fft, delay, sign, length)
        data        cepstrum indexed by quefrency (negative quefrency wraps to the end)
        quefrency   signed quefrency axis [s] (= Hz^-1)
        delay/sign  complex kind only: removed linear-phase delay [samples] and DC sign
Parameters
    kind    "real" (IFFT log|X|) | "complex" (log|X| + j*phase; even n_fft, needs phase)
            | "power" (|IFFT log|X|^2|^2)                       default "real"
    n_fft   FFT length, >= signal length                        default signal length
    window  "hann" | "hamming" | "rect" | ndarray | None        default None
    eps     log floor relative to max|X|                        default 1e-12
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np

from .spectrum import Spectrum
from .utils import WindowSpec, get_window, to_mono

CepKind = Literal["real", "complex", "power"]


@dataclass(frozen=True)
class Cepstrum:
    """Cepstrum ``data[n_fft]`` indexed by quefrency (negative quefrencies wrap to the end).

    ``delay`` / ``sign`` are only meaningful for ``kind="complex"``: the original signal is
    ``sign * roll(x_min, delay)`` where x_min is the signal rebuilt from ``data``.
    """

    data: np.ndarray
    fs: float
    kind: CepKind
    n_fft: int
    delay: int = 0
    sign: int = 1
    length: Optional[int] = None  # original signal length, if known

    @property
    def quefrency(self) -> np.ndarray:
        """Signed quefrency axis in seconds, matching the layout of ``data``."""
        return np.fft.fftfreq(self.n_fft) * self.n_fft / self.fs


def _log_mag(mag: np.ndarray, eps: float) -> np.ndarray:
    floor = max(eps * float(mag.max()), np.finfo(float).tiny)
    return np.log(np.maximum(mag, floor))


def _complex_log_spectrum(full: np.ndarray, eps: float):
    """log X with unwrapped, linear-phase-removed phase. Returns (logX, delay, sign)."""
    n = full.shape[0]
    if n % 2:
        raise ValueError("complex cepstrum requires an even n_fft")
    sign = -1 if full[0].real < 0 else 1
    full = sign * full
    half = full[: n // 2 + 1]
    phi = np.unwrap(np.angle(half))
    r = int(np.round(phi[-1] / np.pi))  # phase at Nyquist = pi * r  ->  delay = -r
    k = np.arange(n // 2 + 1)
    phi = phi - np.pi * r * k / (n // 2)
    phi_full = np.concatenate([phi, -phi[-2:0:-1]])
    return _log_mag(np.abs(full), eps) + 1j * phi_full, -r, sign


def cepstrum(
    x: Union[np.ndarray, Spectrum],
    fs: Optional[float] = None,
    *,
    kind: CepKind = "real",
    n_fft: Optional[int] = None,
    window: WindowSpec = None,
    eps: float = 1e-12,
) -> Cepstrum:
    """Cepstrum of the whole input with a single FFT.

    ``x`` is a time signal (``[N]`` or ``[ch, N]``, channels averaged) or a :class:`Spectrum`.
    """
    length = None
    if isinstance(x, Spectrum):
        spec = x
    else:
        if fs is None:
            raise ValueError("fs is required for time-signal input")
        sig = to_mono(x)
        n_fft = len(sig) if n_fft is None else n_fft
        if n_fft < len(sig):
            raise ValueError("n_fft must be >= signal length")
        length = len(sig)
        spec = Spectrum(np.fft.fft(sig * get_window(window, len(sig)), n_fft), fs, n_fft)

    full = spec.to_two_sided()
    if kind == "real":
        c = np.fft.ifft(_log_mag(np.abs(full), eps)).real
        return Cepstrum(c, spec.fs, kind, spec.n_fft, length=length)
    if kind == "power":
        c = np.abs(np.fft.ifft(2 * _log_mag(np.abs(full), eps))) ** 2
        return Cepstrum(c, spec.fs, kind, spec.n_fft, length=length)
    if kind == "complex":
        if not spec.has_phase:
            raise ValueError("complex cepstrum needs a complex spectrum (phase)")
        log_x, delay, sign = _complex_log_spectrum(full, eps)
        c = np.fft.ifft(log_x).real
        return Cepstrum(c, spec.fs, kind, spec.n_fft, delay, sign, length)
    raise ValueError(f"unknown kind: {kind!r}")
