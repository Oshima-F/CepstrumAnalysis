"""Short-time cepstrum (cepstrogram) analysis with low / high quefrency parts.

Inputs
    x       time signal [N] / [ch, N] (channels averaged)
    fs      sampling frequency [Hz]
Outputs
    Decomposition: full / low / high Cepstrogram objects [quefrency, frame];
    log_spectrum(part) -> [freq, frame] dB, freqs [Hz], reconstruct(part) -> overlap-add signal
Parameters
    frame_len   frame length [samples]                                        required
    hop         frame shift [samples]                                         default frame_len // 4
    window      "hann" | "hamming" | "rect" | ndarray                         default "hann"
    n_fft       FFT length, >= frame_len (even for complex)                   default frame_len
    kind        "real" | "complex" | "power"                                  default "real"
    lifter | cutoff | khz_inv | period_hz   split point, at most one form     default none (no split)
    eps         log floor relative to max|X|                                  default 1e-12
"""
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
) -> Decomposition[Cepstrogram]:
    """Cepstrogram ``[quefrency, frame]`` of ``x`` (``[N]`` or ``[ch, N]``) split at a lifter.

    ``kind`` is "real", "complex" or "power". Lifter arguments are as in :func:`analyze_real` and apply
    to every frame. ``Decomposition.log_spectrum(part)`` gives a ``[freq, frame]`` spectrogram in dB
    (``low`` = envelope spectrogram), and ``reconstruct(part)`` overlap-adds the frames back to a signal.
    """
    full = cepstrogram(x, fs, frame_len=frame_len, hop=hop, window=window, n_fft=n_fft, kind=kind, eps=eps)
    return split(full, make_lifter(lifter, cutoff, khz_inv, period_hz))
