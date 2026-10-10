"""Short-time cepstrum (cepstrogram) and overlap-add reconstruction.

Inputs
    cepstrogram(x, fs, *, frame_len, hop, window, n_fft, kind, eps)
        x           time signal [N] or [ch, N] (channels averaged)
        fs          sampling frequency [Hz]
    reconstruct_cepstrogram(cg, lifter=None, *, phase)    cg: Cepstrogram
Outputs
    Cepstrogram: data[n_fft, n_frames], quefrency [s], times [s] (frame centres), plus the
    window / per-frame delay and sign needed for reconstruction
    reconstruct_cepstrogram -> float ndarray [original length] (overlap-add / window envelope)
Parameters
    frame_len   frame length [samples]                       required
    hop         frame shift [samples]                        default frame_len // 4
    window      "hann" | "hamming" | "rect" | ndarray        default "hann"
    n_fft       FFT length, >= frame_len (even for complex)  default frame_len
    kind        "real" | "complex" | "power"                 default "real"
    eps         log floor relative to max|X|                 default 1e-12
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .core import CepKind, Cepstrum, cepstrum
from .lifter import Lifter, apply_lifter
from .reconstruct import reconstruct
from .utils import WindowSpec, get_window, to_mono


@dataclass(frozen=True)
class Cepstrogram:
    """``data[n_fft, n_frames]``; frame i starts at sample ``i*hop - pad`` of the input."""

    data: np.ndarray
    fs: float
    kind: CepKind
    n_fft: int
    frame_len: int
    hop: int
    window: np.ndarray
    delays: np.ndarray
    signs: np.ndarray
    length: int  # original signal length
    pad: int

    @property
    def quefrency(self) -> np.ndarray:
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
) -> Cepstrogram:
    """Cepstrum of each windowed frame. ``x`` is ``[N]`` or ``[ch, N]`` (channels averaged)."""
    sig = to_mono(x)
    hop = frame_len // 4 if hop is None else hop
    n_fft = frame_len if n_fft is None else n_fft
    if n_fft < frame_len:
        raise ValueError("n_fft must be >= frame_len")
    if kind == "complex" and n_fft % 2:
        raise ValueError("complex cepstrum requires an even n_fft")
    win = get_window(window, frame_len)

    pad = frame_len - hop
    n_frames = int(np.ceil((len(sig) + pad) / hop))
    padded = np.zeros(pad + (n_frames - 1) * hop + frame_len)
    padded[pad:pad + len(sig)] = sig

    cols, delays, signs = [], [], []
    for i in range(n_frames):
        frame = padded[i * hop:i * hop + frame_len] * win
        c = cepstrum(frame, fs, kind=kind, n_fft=n_fft, eps=eps)
        cols.append(c.data)
        delays.append(c.delay)
        signs.append(c.sign)
    return Cepstrogram(np.stack(cols, axis=1), fs, kind, n_fft, frame_len, hop, win,
                       np.array(delays), np.array(signs), len(sig), pad)


def reconstruct_cepstrogram(cg: Cepstrogram, lifter: Optional[Lifter] = None, *, phase: str = "minimum") -> np.ndarray:
    """Per-frame reconstruction + overlap-add, normalised by the window overlap envelope."""
    if lifter is not None:
        cg = apply_lifter(cg, lifter)
    n_frames = cg.data.shape[1]
    total = cg.pad + (n_frames - 1) * cg.hop + cg.frame_len
    out = np.zeros(total)
    env = np.zeros(total)
    for i in range(n_frames):
        c = Cepstrum(cg.data[:, i], cg.fs, cg.kind, cg.n_fft, int(cg.delays[i]), int(cg.signs[i]), cg.frame_len)
        s = i * cg.hop
        out[s:s + cg.frame_len] += reconstruct(c, phase=phase)
        env[s:s + cg.frame_len] += cg.window
    out = out[cg.pad:cg.pad + cg.length]
    env = env[cg.pad:cg.pad + cg.length]
    return out / np.maximum(env, 1e-8)
