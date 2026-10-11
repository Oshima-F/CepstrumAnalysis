"""Short-time cepstrum (cepstrogram) analysis with low / high quefrency parts."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .cepstrogram import Cepstrogram, cepstrogram
from .core import CepKind
from .lifter import Lifter
from .split import Decomposition, make_lifter, split
from .utils import WindowSpec


def analyze_cepstrogram(
    x: np.ndarray,
    fs: float,
    *,
    frame_len: int,
    hop: Optional[int] = None,
    window: WindowSpec = "hann",
    n_fft: Optional[int] = None,
    kind: CepKind = "real",
    lifter: Optional[Lifter] = None,
    cutoff: Optional[float] = None,
    khz_inv: Optional[float] = None,
    period_hz: Optional[float] = None,
    eps: float = 1e-12,
    db: bool = False,
    store_spectrogram: bool = True,
) -> Decomposition[Cepstrogram]:
    """Cepstrogram ``[quefrency, frame]`` of a signal, split at a lifter.

    The same lifter is applied to every frame. ``log_spectrum(part)`` of the result gives a
    ``[freq, frame]`` spectrogram in dB (``low`` = envelope spectrogram), and ``reconstruct(part)``
    returns a time signal: the original complex spectrogram times the gain ``|X_l| / |X|``,
    overlap-added (the original phase is kept; ``"full"`` is the original signal).

    Args:
        x: Time signal ``[N]`` or ``[ch, N]`` (channels averaged). Required.
        fs: Sampling frequency [Hz]. Required.
        frame_len: Frame length [samples]. Required (keyword-only).
        hop: Frame shift [samples]. Default ``frame_len // 4``.
        window: ``"hann"``, ``"hamming"``, ``"rect"`` or an array of length ``frame_len``.
            Default ``"hann"``.
        n_fft: FFT length, at least ``frame_len``. Default ``frame_len``.
        kind: ``"real"`` or ``"power"``. Default ``"real"``.
        lifter: Callable of signed quefrency [s], or weight array of length ``n_fft``, defining
            the low part. Default ``None``.
        cutoff: Lowpass cutoff [s = Hz^-1], instead of ``lifter``. Default ``None``.
        khz_inv: Lowpass cutoff [kHz^-1], instead of ``lifter``. Default ``None``.
        period_hz: Lowpass cutoff as a ripple period [Hz], instead of ``lifter``. Default ``None``.
            Give at most one of ``lifter`` / ``cutoff`` / ``khz_inv`` / ``period_hz``; with none,
            ``low`` is the full cepstrogram and ``high`` is zero.
        eps: Floor of the log, relative to the per-frame ``max|X|``. Default ``1e-12``.
        db: If True, compute from the log spectrum in dB (``20*log10|X|``) instead of the
            natural log (see :func:`cepstrum`). Default ``False``.
        store_spectrogram: Keep the complex spectrogram (needed by ``reconstruct``). Default
            ``True``.

    Returns:
        A :class:`Decomposition` whose parts are :class:`Cepstrogram` objects.

    Raises:
        ValueError: ``n_fft < frame_len``, an unknown ``kind``, or an invalid lifter
            specification.
    """
    full = cepstrogram(x, fs, frame_len=frame_len, hop=hop, window=window, n_fft=n_fft, kind=kind, eps=eps, db=db,
                       store_spectrogram=store_spectrogram)
    return split(full, make_lifter(lifter, cutoff, khz_inv, period_hz))
