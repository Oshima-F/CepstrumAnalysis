"""Real cepstrum analysis: cepstrum plus low / high quefrency parts.

Inputs
    x       time signal [N] / [ch, N] (channels averaged), or Spectrum (complex/amplitude/power,
            one/two-sided)
    fs      sampling frequency [Hz] (time signals only)
Outputs
    Decomposition: full / low (envelope) / high (fine structure) Cepstrum objects,
    log_spectrum(part) [dB], freqs [Hz], reconstruct(part)
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
) -> Decomposition[Cepstrum]:
    """Real cepstrum IFFT(log|X|) of the whole input, split at a lifter.

    ``x``: time signal (``[N]`` or ``[ch, N]``, channels averaged) or :class:`Spectrum`
    (complex/amplitude/power, one/two-sided).
    Lifter: pass ``lifter`` (callable of signed quefrency [s] or weight array), or a lowpass cutoff by
    exactly one of ``cutoff`` [s = Hz^-1], ``khz_inv`` [kHz^-1] or ``period_hz`` [Hz]. With none,
    ``low`` is the full cepstrum and ``high`` is zero.

    Returns a :class:`Decomposition` with ``full`` / ``low`` (envelope) / ``high`` (fine structure),
    ``log_spectrum(part)`` and ``reconstruct(part)``.
    """
    full = cepstrum(x, fs, kind="real", n_fft=n_fft, window=window, eps=eps)
    return split(full, make_lifter(lifter, cutoff, khz_inv, period_hz))
