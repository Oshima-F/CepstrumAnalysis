"""Power cepstrum analysis: |IFFT(log|X|^2)|^2 plus low / high quefrency parts.

Inputs
    x       time signal [N] / [ch, N] (channels averaged), or Spectrum (complex/amplitude/power)
    fs      sampling frequency [Hz] (time signals only)
Outputs
    Decomposition: full / low / high Cepstrum objects (kind "power"); not invertible, so
    log_spectrum / reconstruct raise ValueError
Parameters
    lifter | cutoff | khz_inv | period_hz   split point, at most one form    default none (no split)
    n_fft    FFT length                                                       default signal length
    window   "hann" | "hamming" | "rect" | ndarray | None                     default None
    eps      log floor relative to max|X|                                     default 1e-12
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .core import Cepstrum, cepstrum
from .lifter import Lifter
from .spectrum import Spectrum
from .split import Decomposition, make_lifter, split
from .utils import WindowSpec


def analyze_power(
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
) -> Decomposition[Cepstrum]:
    """Power cepstrum split at a lifter (detection of periodic structure, e.g. echoes / harmonics).

    The power cepstrum is not invertible, so ``log_spectrum`` and ``reconstruct`` raise ``ValueError``.
    Lifter arguments are as in :func:`analyze_real`.
    """
    full = cepstrum(x, fs, kind="power", n_fft=n_fft, window=window, eps=eps)
    return split(full, make_lifter(lifter, cutoff, khz_inv, period_hz))
