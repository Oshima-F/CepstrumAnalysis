"""Short-time cepstrum (cepstrogram) and its reconstruction to a time signal.

Reconstruction keeps the phase of the original complex spectrogram and replaces its magnitude: a
frame's spectrum ``X`` is multiplied by the gain ``|X_l| / |X|`` (= ``sqrt(P_l / P_o)``), where
``P_o = |X|^2`` is the original power spectrogram and ``P_l`` the power spectrogram carried by the
(liftered) cepstrogram, and the frames are overlap-added with a synthesis window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .core import DB_PER_NEPER, CepKind, _from_spectrum
from .lifter import Lifter, apply_lifter
from .spectrum import Spectrum
from .utils import WindowSpec, get_window
from .validation import check_fs, prepare_signal

_GAIN_FLOOR = 1e-12  # relative to the frame's peak magnitude; matches the default ``eps`` of the cepstrum


@dataclass(frozen=True)
class Cepstrogram:
    """Short-time cepstrum; frame ``i`` starts at sample ``i*hop - pad`` of the input.

    Normally created by :func:`cepstrogram`, not by hand.

    Args:
        data: Array of shape ``[n_fft, n_frames]`` (quefrency x frame).
        fs: Sampling frequency [Hz].
        kind: ``"real"`` or ``"power"``.
        n_fft: FFT length.
        frame_len: Frame length [samples].
        hop: Frame shift [samples].
        window: Analysis window used, shape ``[frame_len]``.
        length: Length of the original signal [samples].
        pad: Zero padding added at the start [samples] (``frame_len - hop``).
        db: True if computed from the log spectrum in dB (``20*log10|X|``) instead of the
            natural log. Default ``False``.
        spectrogram: Complex one-sided spectrogram ``[n_fft // 2 + 1, n_frames]`` of the windowed
            frames (the original, not liftered), kept for reconstruction. ``None`` if it was not
            stored. Default ``None``.
    """

    data: np.ndarray
    fs: float
    kind: CepKind
    n_fft: int
    frame_len: int
    hop: int
    window: np.ndarray
    length: int  # original signal length
    pad: int
    db: bool = False
    spectrogram: Optional[np.ndarray] = None

    @property
    def quefrency(self) -> np.ndarray:
        """Signed quefrency axis in seconds, matching the first axis of ``data``."""
        return np.fft.fftfreq(self.n_fft) * self.n_fft / self.fs

    @property
    def times(self) -> np.ndarray:
        """Frame-centre times [s] relative to the first input sample."""
        return (np.arange(self.data.shape[1]) * self.hop - self.pad + self.frame_len / 2) / self.fs


def cepstrogram(
    x: np.ndarray,
    fs: float,
    *,
    frame_len: int,
    hop: Optional[int] = None,
    window: WindowSpec = "hann",
    n_fft: Optional[int] = None,
    kind: CepKind = "real",
    eps: float = 1e-12,
    db: bool = False,
    store_spectrogram: bool = True,
) -> Cepstrogram:
    """Cepstrum of each windowed frame.

    The complex spectrum of every frame is computed anyway (the cepstrum uses its magnitude), so it
    is kept in the result; it is what :func:`reconstruct_cepstrogram` takes the phase from.

    Args:
        x: Time signal ``[N]`` or ``[ch, N]`` (channels averaged). Required.
        fs: Sampling frequency [Hz]. Required.
        frame_len: Frame length [samples]. Required (keyword-only).
        hop: Frame shift [samples]. Default ``frame_len // 4``.
        window: ``"hann"``, ``"hamming"``, ``"rect"`` or an array of length ``frame_len``.
            Default ``"hann"``.
        n_fft: FFT length, at least ``frame_len``. Default ``frame_len``.
        kind: ``"real"`` or ``"power"``. Default ``"real"``.
        eps: Floor of the log, relative to the per-frame ``max|X|``. Default ``1e-12``.
        db: If True, compute from the log spectrum in dB (``20*log10|X|``) instead of the
            natural log (see :func:`cepstrum`). Default ``False``.
        store_spectrogram: Keep the complex spectrogram in the result (needed to reconstruct a
            time signal; costs memory of the same order as ``data``). Default ``True``.

    Returns:
        A :class:`Cepstrogram` with ``data`` of shape ``[n_fft, n_frames]``.

    Raises:
        ValueError: ``fs`` not a finite positive number, ``frame_len < 1``, ``hop`` outside
            ``1 .. frame_len``, ``n_fft < frame_len``, an unknown ``kind``, or an invalid signal
            (complex, empty, NaN / inf, wrong shape).
        SilenceWarning: (warning) The whole signal is digital silence or constant. Silent
            frames inside a longer signal (pauses) are normal and do not warn.
        ClippingWarning: (warning) The signal appears clipped.
    """
    if kind not in ("real", "power"):
        raise ValueError(f"unknown kind: {kind!r}; choose 'real' or 'power'")
    fs = check_fs(fs)
    if frame_len < 1:
        raise ValueError("frame_len must be >= 1")
    hop = frame_len // 4 if hop is None else hop
    if not 1 <= hop <= frame_len:
        raise ValueError(f"hop must be between 1 and frame_len ({frame_len}), got {hop}")
    n_fft = frame_len if n_fft is None else n_fft
    if n_fft < frame_len:
        raise ValueError("n_fft must be >= frame_len")
    sig = prepare_signal(x)  # whole-signal checks and warnings, once
    win = get_window(window, frame_len)

    pad = frame_len - hop
    n_frames = int(np.ceil((len(sig) + pad) / hop))
    padded = np.zeros(pad + (n_frames - 1) * hop + frame_len)
    padded[pad:pad + len(sig)] = sig

    cols, specs = [], []
    for i in range(n_frames):
        spectrum = np.fft.fft(padded[i * hop:i * hop + frame_len] * win, n_fft)
        cols.append(_from_spectrum(Spectrum(spectrum, fs, n_fft), kind, eps, db, frame_len).data)
        if store_spectrogram:
            specs.append(spectrum[: n_fft // 2 + 1])
    spectrogram = np.stack(specs, axis=1) if store_spectrogram else None
    return Cepstrogram(np.stack(cols, axis=1), fs, kind, n_fft, frame_len, hop, win, len(sig), pad, db,
                       spectrogram)


def reconstruct_cepstrogram(cg: Cepstrogram, lifter: Optional[Lifter] = None) -> np.ndarray:
    """Reconstruct a time signal from a (liftered) cepstrogram, keeping the original phase.

    For every frame the complex spectrum ``X`` of the original is multiplied by the gain
    ``G = |X_l| / |X|`` (= ``sqrt(P_l / P_o)``): ``P_o = |X|^2`` is the original power spectrogram
    and ``P_l`` the power spectrogram carried by the (liftered) cepstrogram, ``ln|X_l|`` being the
    FFT of the cepstrum. The frames are inverse-transformed and overlap-added with the analysis
    window as synthesis window, normalised by ``sum(window^2)``. Without a lifter ``G = 1`` and the
    original signal is returned; the lifter changes the magnitude only, never the phase.

    Args:
        cg: A :class:`Cepstrogram` that holds its complex spectrogram. Required.
        lifter: Optional callable of signed quefrency [s], or weight array of length ``n_fft``,
            applied to every frame first. Default ``None`` (no liftering).

    Returns:
        Float array of the original signal length.

    Raises:
        ValueError: For ``kind="power"`` (not invertible), or if the cepstrogram was made with
            ``store_spectrogram=False``.
    """
    if cg.kind == "power":
        raise ValueError("a power cepstrogram is not invertible")
    if cg.spectrogram is None:
        raise ValueError("this cepstrogram holds no complex spectrogram (store_spectrogram=False); "
                         "the phase needed for reconstruction is missing")
    if lifter is not None:
        cg = apply_lifter(cg, lifter)
    half = cg.n_fft // 2 + 1
    log_mag = np.fft.fft(cg.data, axis=0).real[:half]  # ln|X_l|, or its dB value if cg.db
    if cg.db:
        log_mag = log_mag / DB_PER_NEPER
    spec = cg.spectrogram
    mag = np.abs(spec)
    floor = np.maximum(_GAIN_FLOOR * mag.max(axis=0, keepdims=True), np.finfo(float).tiny)
    gain = np.exp(log_mag) / np.maximum(mag, floor)
    frames = np.fft.irfft(spec * gain, n=cg.n_fft, axis=0)[: cg.frame_len]

    n_frames = cg.data.shape[1]
    total = cg.pad + (n_frames - 1) * cg.hop + cg.frame_len
    out = np.zeros(total)
    env = np.zeros(total)
    for i in range(n_frames):
        s = i * cg.hop
        out[s:s + cg.frame_len] += cg.window * frames[:, i]
        env[s:s + cg.frame_len] += cg.window ** 2
    out = out[cg.pad:cg.pad + cg.length]
    env = env[cg.pad:cg.pad + cg.length]
    return out / np.maximum(env, 1e-8)
