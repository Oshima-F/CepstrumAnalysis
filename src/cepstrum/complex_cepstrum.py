"""Complex cepstrum analysis: cepstrum (with phase) plus low / high quefrency parts.

Inputs
    x       time signal [N] / [ch, N] (channels averaged), or a complex Spectrum (needs phase)
    fs      sampling frequency [Hz] (time signals only)
Outputs
    Decomposition: full / low / high Cepstrum objects (kind "complex"); low and high are homomorphic
    factors (circular convolution == full; delay and sign live in low),
    log_spectrum(part) [dB], freqs [Hz], reconstruct(part)
Parameters
    lifter | cutoff | khz_inv | period_hz   split point, at most one form    default none (no split)
    n_fft    FFT length, must be even                                         default signal length
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


def analyze_complex(
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
    """Complex cepstrum IFFT(log|X| + j*unwrapped phase) split at a lifter.

    ``x`` must carry phase: a time signal or a *complex* :class:`Spectrum` (even ``n_fft``).
    Lifter arguments are as in :func:`analyze_real`. The lifter may be asymmetric in signed quefrency
    (positive q = minimum-phase part, negative q = maximum-phase part).

    ``Decomposition.reconstruct("full")`` returns the original signal; ``"low"`` / ``"high"`` are
    components whose convolution gives the original (homomorphic deconvolution).
    """
    full = cepstrum(x, fs, kind="complex", n_fft=n_fft, window=window, eps=eps)
    return split(full, make_lifter(lifter, cutoff, khz_inv, period_hz))
