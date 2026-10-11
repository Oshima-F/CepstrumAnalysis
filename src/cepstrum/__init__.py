"""Cepstrum analysis: cepstrum, cepstrogram, liftering and reconstruction.

A real-cepstrum library: the cepstrum is computed from the magnitude spectrum only (the phase of a
complex input is ignored). A time signal is reconstructed from a cepstrogram: the original complex
spectrogram is multiplied by the gain ``|X_l| / |X|`` (``X_l``: the liftered cepstrogram's
spectrum), so the phase of the original is kept.

High-level entry points (each returns a :class:`Decomposition` with full / low / high quefrency parts):
``analyze_real``, ``analyze_power``, ``analyze_cepstrogram``.
Low-level building blocks: ``cepstrum``, ``cepstrogram``, ``apply_lifter``, ``reconstruct_cepstrogram``.
"""
from .cepstrogram import Cepstrogram, cepstrogram, reconstruct_cepstrogram
from .cepstrogram_analysis import analyze_cepstrogram
from .core import Cepstrum, cepstrum
from .lifter import apply_lifter, bandpass, highpass, lifter_weights, lowpass, notch
from .power_cepstrum import analyze_power
from .real_cepstrum import analyze_real
from .spectrum import Spectrum
from .split import Decomposition
from .units import UNITS, from_seconds, quefrency_axis, to_seconds
from .validation import CepstrumWarning, ClippingWarning, SilenceWarning

__all__ = [
    "Spectrum", "Cepstrum", "Cepstrogram", "Decomposition",
    "analyze_real", "analyze_power", "analyze_cepstrogram",
    "cepstrum", "cepstrogram",
    "apply_lifter", "lowpass", "highpass", "bandpass", "notch", "lifter_weights",
    "reconstruct_cepstrogram",
    "UNITS", "quefrency_axis", "from_seconds", "to_seconds",
    "CepstrumWarning", "SilenceWarning", "ClippingWarning",
]
