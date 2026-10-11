"""Average (whole-record) cepstrum: real and power."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np

from .spectrum import Spectrum
from .utils import WindowSpec, get_window
from .validation import check_fs, prepare_signal, warn_if_silent_spectrum

CepKind = Literal["real", "power"]

DB_PER_NEPER = 20 / np.log(10)  # ln|X| -> 20*log10|X|  (= 8.686...)


@dataclass(frozen=True)
class Cepstrum:
    """Cepstrum indexed by quefrency (negative quefrencies wrap to the end).

    Normally created by :func:`cepstrum`, not by hand. The cepstrum is real and even
    (``data[n] == data[-n]``): it is computed from the magnitude spectrum only.

    Args:
        data: Real array of shape ``[n_fft]``.
        fs: Sampling frequency [Hz].
        kind: ``"real"`` or ``"power"``.
        n_fft: FFT length.
        length: Length of the original time signal, if known (reconstruction is truncated to
            it). Default ``None``.
        db: True if the cepstrum was computed from the log spectrum in dB (``20*log10|X|``)
            instead of the natural log. Default ``False``.
    """

    data: np.ndarray
    fs: float
    kind: CepKind
    n_fft: int
    length: Optional[int] = None
    db: bool = False

    @property
    def quefrency(self) -> np.ndarray:
        """Signed quefrency axis in seconds, matching the layout of ``data``."""
        return np.fft.fftfreq(self.n_fft) * self.n_fft / self.fs


def _log_mag(mag: np.ndarray, eps: float) -> np.ndarray:
    floor = max(eps * float(mag.max()), np.finfo(float).tiny)
    return np.log(np.maximum(mag, floor))


def cepstrum(
    x: Union[np.ndarray, Spectrum],
    fs: Optional[float] = None,
    *,
    kind: CepKind = "real",
    n_fft: Optional[int] = None,
    window: WindowSpec = None,
    eps: float = 1e-12,
    db: bool = False,
) -> Cepstrum:
    """Cepstrum of the whole input with a single FFT.

    Only the magnitude spectrum is used; the phase of a complex input is ignored.

    Args:
        x: Time signal ``[N]`` / ``[ch, N]`` (channels averaged) or a :class:`Spectrum`.
        fs: Sampling frequency [Hz]. Required for time signals; taken from the
            :class:`Spectrum` otherwise. Default ``None``.
        kind: ``"real"`` (IFFT of log|X|) or ``"power"`` (|IFFT of log|X|^2|^2). Default
            ``"real"``.
        n_fft: FFT length, at least the signal length. Time signals only. Default: the signal
            length.
        window: ``"hann"``, ``"hamming"``, ``"rect"``, an array of the signal length, or ``None``.
            Time signals only. Default ``None`` (no window).
        eps: Floor of the log, relative to ``max|X|``. Default ``1e-12``.
        db: If True, use the log spectrum in dB (``20*log10|X|``) instead of the natural log
            before the inverse FFT. A real cepstrum is then ``20/ln(10)`` (about 8.686) times
            larger, and the FFT of it is the log spectrum in dB. For the power cepstrum the
            squared magnitude of that dB cepstrum is returned. Default ``False`` (natural log).

    Returns:
        A :class:`Cepstrum`.

    Raises:
        ValueError: ``fs`` missing or not a finite positive number, ``n_fft`` shorter than the
            signal, an unknown ``kind``, or an invalid signal (complex, empty, NaN / inf, wrong
            shape).
        SilenceWarning: (warning) The signal is digital silence or constant (or the spectrum is
            all zero).
        ClippingWarning: (warning) The signal appears clipped. See :mod:`cepstrum.validation`.
    """
    if kind not in ("real", "power"):
        raise ValueError(f"unknown kind: {kind!r}; choose 'real' or 'power'")
    length = None
    if isinstance(x, Spectrum):
        spec = x
        warn_if_silent_spectrum(np.abs(spec.data))
    else:
        if fs is None:
            raise ValueError("fs is required for time-signal input")
        fs = check_fs(fs)
        sig = prepare_signal(x)
        n_fft = len(sig) if n_fft is None else n_fft
        if n_fft < len(sig):
            raise ValueError("n_fft must be >= signal length")
        length = len(sig)
        spec = Spectrum(np.fft.fft(sig * get_window(window, len(sig)), n_fft), fs, n_fft)
    return _from_spectrum(spec, kind, eps, db, length)


def _from_spectrum(spec: Spectrum, kind: CepKind, eps: float, db: bool, length: Optional[int]) -> Cepstrum:
    """Cepstrum of a :class:`Spectrum` (no validation, no warnings)."""
    log_mag = _log_mag(np.abs(spec.to_two_sided()), eps)
    if db:
        log_mag = log_mag * DB_PER_NEPER  # 20*log10|X|; the power cepstrum then needs no extra factor 2
    if kind == "real":
        c = np.fft.ifft(log_mag).real
    else:
        c = np.abs(np.fft.ifft(log_mag if db else 2 * log_mag)) ** 2
    return Cepstrum(c, spec.fs, kind, spec.n_fft, length, db)
