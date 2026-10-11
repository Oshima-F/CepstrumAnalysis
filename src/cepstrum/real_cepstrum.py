"""Real cepstrum analysis: cepstrum plus low / high quefrency parts."""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .core import Cepstrum, cepstrum
from .lifter import Lifter
from .spectrum import Spectrum
from .split import Decomposition, make_lifter, split
from .utils import WindowSpec


def analyze_real(
    x: Union[np.ndarray, Spectrum],
    fs: Optional[float] = None,
    *,
    lifter: Optional[Lifter] = None,
    cutoff: Optional[float] = None,
    khz_inv: Optional[float] = None,
    period_hz: Optional[float] = None,
    n_fft: Optional[int] = None,
    window: WindowSpec = None,
    eps: float = 1e-12,
    db: bool = False,
) -> Decomposition[Cepstrum]:
    """Real cepstrum IFFT(log|X|) of the whole input, split at a lifter.

    Args:
        x: Time signal ``[N]`` / ``[ch, N]`` (channels averaged) or a :class:`Spectrum`
            (complex/amplitude/power, one/two-sided). Required.
        fs: Sampling frequency [Hz]. Required for time signals; taken from the
            :class:`Spectrum` otherwise. Default ``None``.
        lifter: Callable of signed quefrency [s], or weight array of length ``n_fft``, defining
            the low part. Default ``None``.
        cutoff: Lowpass cutoff [s = Hz^-1], instead of ``lifter``. Default ``None``.
        khz_inv: Lowpass cutoff [kHz^-1], instead of ``lifter``. Default ``None``.
        period_hz: Lowpass cutoff as a ripple period [Hz], instead of ``lifter``. Default ``None``.
            Give at most one of ``lifter`` / ``cutoff`` / ``khz_inv`` / ``period_hz``; with none,
            ``low`` is the full cepstrum and ``high`` is zero.
        n_fft: FFT length, at least the signal length (time signals only). Default: the signal
            length.
        window: ``"hann"``, ``"hamming"``, ``"rect"``, an array, or ``None`` (time signals only).
            Default ``None``.
        eps: Floor of the log, relative to ``max|X|``. Default ``1e-12``.
        db: If True, return the cepstrum of the log spectrum in dB (``20*log10|X|``) instead of
            the natural log, i.e. ``20/ln(10)`` times larger (see :func:`cepstrum`). Default
            ``False``.

    Returns:
        A :class:`Decomposition` with ``full`` / ``low`` (envelope) / ``high`` (fine structure),
        ``log_spectrum(part)`` and ``freqs``. (A time signal is reconstructed from a cepstrogram, not
        from a single cepstrum: see :func:`analyze_cepstrogram`.)

    Raises:
        ValueError: Invalid lifter specification, missing ``fs``, or ``n_fft`` shorter than the
            signal.
    """
    full = cepstrum(x, fs, kind="real", n_fft=n_fft, window=window, eps=eps, db=db)
    return split(full, make_lifter(lifter, cutoff, khz_inv, period_hz))
