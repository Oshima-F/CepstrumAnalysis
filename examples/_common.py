"""Shared helpers for the example scripts (speech input, figure saving, axes).

Module constants: ``DEFAULT_WAV`` (speech used by all examples, data/mao.wav, created with
synth_aivis.py), ``OUT`` (default output directory, examples/output) and ``QLABEL`` (axis label for
quefrency in kHz^-1).
"""
import wave
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import cepstrum as cp  # noqa: E402

DEFAULT_WAV = Path(__file__).parent / "data" / "mao.wav"
OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)


def read_wav(path):
    """Read a 16-bit PCM WAV file.

    Args:
        path: WAV file. Required.

    Returns:
        Tuple ``(x, fs)``: ``x`` is a float array ``[ch, N]`` in [-1, 1], ``fs`` the sampling
        frequency [Hz].

    Raises:
        ValueError: If the file is not 16-bit PCM.
    """
    with wave.open(str(path)) as w:
        fs, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        raise ValueError("only 16-bit PCM is supported")
    return (np.frombuffer(raw, dtype="<i2").astype(float) / 32768.0).reshape(-1, ch).T, fs


def load_segment(path=DEFAULT_WAV, start=0.0, dur=None):
    """Mono excerpt (channels averaged) of a WAV file.

    Args:
        path: WAV file. Default ``DEFAULT_WAV``.
        start: Excerpt start [s]. Default 0.0.
        dur: Excerpt length [s]. Default ``None`` (to the end of the file).

    Returns:
        Tuple ``(x, fs)``: float array ``[n]`` and the sampling frequency [Hz].

    Raises:
        FileNotFoundError: If ``path`` does not exist (create it with synth_aivis.py).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; create it with synth_aivis.py")
    x, fs = read_wav(path)
    x = x.mean(axis=0)
    i0 = int(round(start * fs))
    i1 = len(x) if dur is None else i0 + int(round(dur * fs))
    return x[i0:i1], fs


def f0_from_cepstrum(c, f0_range=(100, 700)):
    """Pitch from the strongest peak of a real cepstrum.

    Args:
        c: A real :class:`cepstrum.Cepstrum`. Required.
        f0_range: Pitch search range [Hz]. Default ``(100, 700)``.

    Returns:
        f0 [Hz].
    """
    q = cp.quefrency_axis(c, "kHz^-1")[: c.n_fft // 2]
    band = (q >= 1000 / f0_range[1]) & (q <= 1000 / f0_range[0])
    return float(1000.0 / q[band][np.argmax(c.data[: c.n_fft // 2][band])])


def open_image(path):
    """Open a saved PNG with the OS default viewer (figures are saved headless, so nothing pops up).

    Args:
        path: PNG file. Required.
    """
    import os
    import subprocess
    import sys

    if sys.platform == "win32":
        os.startfile(path)
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])


def save(fig, name, out_dir=None):
    """Save a matplotlib figure as PNG and close it.

    Args:
        fig: The figure. Required.
        name: File name. Required.
        out_dir: Output directory, created if missing. Default ``None`` (= ``OUT``, examples/output).

    Returns:
        Path of the PNG.
    """
    out = Path(out_dir) if out_dir is not None else OUT
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("saved", path)
    return path


QLABEL = "quefrency [kHz$^{-1}$]  (= spectral peaks per kHz)"
_PERIODS = (2000, 1000, 500, 333, 250, 200, 167, 143, 125, 100, 83, 67, 50, 40)


def period_axis(ax, axis="x", periods=_PERIODS):
    """Add a secondary axis labelled with the ripple period [Hz] = 1000 / q[kHz^-1].

    Call after the primary limits are final. Ticks sit at 'round' periods, so the primary
    axis stays linear while the period (spacing in Hz) can be read directly.

    Args:
        ax: Matplotlib axes whose quefrency axis is in kHz^-1. Required.
        axis: ``"x"`` (secondary axis on top) or ``"y"`` (on the right). Default ``"x"``.
        periods: Candidate ripple periods [Hz] for the ticks; crowded ones are thinned out.
            Default: a built-in list from 2000 to 40 Hz.

    Returns:
        The secondary axis.
    """
    lim = ax.get_xlim() if axis == "x" else ax.get_ylim()
    lo, hi = min(lim), max(lim)
    ticks, last = [], -np.inf
    for p in sorted(periods, reverse=True):  # ascending q; thin out crowded ticks
        pos = 1000.0 / p
        if lo < pos < hi and pos - last >= 0.07 * (hi - lo):
            ticks.append((pos, p))
            last = pos
    sec = ax.secondary_xaxis("top") if axis == "x" else ax.secondary_yaxis("right")
    (sec.set_xticks if axis == "x" else sec.set_yticks)([t for t, _ in ticks])
    (sec.set_xticklabels if axis == "x" else sec.set_yticklabels)([f"{p} Hz" for _, p in ticks], fontsize=7)
    (sec.set_xlabel if axis == "x" else sec.set_ylabel)("ripple period [Hz]")
    return sec


def to_db(log_mag):
    """Convert a natural-log magnitude to dB.

    Args:
        log_mag: ln of a magnitude (array-like). Required.

    Returns:
        ``20 / ln(10) * log_mag``.
    """
    return 20 / np.log(10) * np.asarray(log_mag)
