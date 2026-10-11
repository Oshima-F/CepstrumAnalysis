"""Cepstrogram of a speech WAV (e.g. synthesized by synth_aivis.py): f0 track and spectral envelope.

    python speech_cepstrogram.py [data/mao.wav]

Plots waveform, log spectrogram, envelope spectrogram and cepstrogram with an f0 track for a
16-bit PCM WAV (mono or multichannel, channels averaged; default data/mao.wav) and writes
speech_<wav name>.png. The settings below are the defaults of run(); edit them here or pass the
arguments to run().
"""
import sys
from pathlib import Path

import numpy as np

import cepstrum as cp
from _common import DEFAULT_WAV, QLABEL, open_image, period_axis, plt, read_wav, save

OUT_DIR = None          # None -> examples/output
FRAME_LEN = 2048        # analysis frame [samples] (46 ms at 44.1 kHz)
HOP = 256               # frame shift [samples]
SPLIT_KHZ_INV = 1       # low/high split [kHz^-1]: ripples wider than 1 kHz = envelope, below the pitch ripple
F0_RANGE = (100, 700)   # pitch search range [Hz]; widen/shift for other voices
PEAK_MIN = 0.06         # minimum high-part peak height to call a frame voiced
SPECTRUM_FMAX = 8000    # upper frequency shown in the spectrograms [Hz]


def run(path=DEFAULT_WAV, out_dir=OUT_DIR, f0_range=F0_RANGE, peak_min=PEAK_MIN,
        frame_len=FRAME_LEN, hop=HOP, split_khz_inv=SPLIT_KHZ_INV):
    """Cepstrogram analysis of a speech WAV: f0 track and spectral envelope.

    Args:
        path: Speech WAV (16-bit PCM). Default ``DEFAULT_WAV`` (data/mao.wav).
        out_dir: Directory for the PNG. Default ``OUT_DIR`` (``None`` = examples/output).
        f0_range: Pitch search range [Hz]. Default ``F0_RANGE``.
        peak_min: Minimum high-part peak height to call a frame voiced. Default ``PEAK_MIN``.
        frame_len: Analysis frame [samples]. Default ``FRAME_LEN``.
        hop: Frame shift [samples]. Default ``HOP``.
        split_khz_inv: Low/high split quefrency [kHz^-1]. Default ``SPLIT_KHZ_INV``.

    Returns:
        Dict with ``"path"`` (the PNG), ``"voiced_ratio"`` (fraction of frames above
        ``peak_min``) and ``"median_f0_hz"`` (median pitch over voiced frames).
    """
    path = Path(path)
    x, fs = read_wav(path)
    # low part = spectral envelope; high part = pitch structure
    r = cp.analyze_cepstrogram(x, fs, frame_len=frame_len, hop=hop, kind="real", khz_inv=split_khz_inv)
    cg = r.full
    q = cp.quefrency_axis(cg, "kHz^-1")
    half = cg.n_fft // 2
    t = cg.times

    # f0 track: strongest peak of the high part for ripple periods within f0_range
    band = (q[:half] >= 1000 / f0_range[1]) & (q[:half] <= 1000 / f0_range[0])
    sub = r.high.data[:half][band]
    f0 = 1000.0 / q[:half][band][sub.argmax(axis=0)]
    voiced = sub.max(axis=0) > peak_min
    f0 = np.where(voiced, f0, np.nan)

    freqs = r.freqs
    env_db = r.log_spectrum("low")
    raw_db = r.log_spectrum("full")
    fsel = freqs <= SPECTRUM_FMAX

    fig, ax = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    ax[0].plot(np.arange(x.shape[1]) / fs, x.mean(axis=0), lw=0.3)
    ax[0].set(title=f"{path.name}  (fs={fs} Hz)", ylabel="amplitude")
    vmax = raw_db[fsel].max()
    ax[1].pcolormesh(t, freqs[fsel], raw_db[fsel], shading="auto", cmap="magma", vmin=vmax - 80, vmax=vmax)
    ax[1].set(title="Log spectrogram (from cepstrogram, no liftering)", ylabel="frequency [Hz]")
    ax[2].pcolormesh(t, freqs[fsel], env_db[fsel], shading="auto", cmap="magma", vmin=vmax - 80, vmax=vmax)
    ax[2].set(title=f"Spectral envelope (low part: ripples wider than {1000 / split_khz_inv:.0f} Hz)", ylabel="frequency [Hz]")
    sel = (q[:half] > 0.5) & (q[:half] < 12)
    ax[3].pcolormesh(t, q[:half][sel], cg.data[:half][sel], shading="auto", cmap="magma", vmin=-0.03, vmax=0.25)
    ax[3].plot(t, 1000.0 / f0, "c.", ms=2, label=f"f0 track (peak > {peak_min})")
    ax[3].set(title="Real cepstrogram", xlabel="time [s]", ylabel=QLABEL, ylim=(0.5, 12))
    ax[3].legend(loc="upper right")
    period_axis(ax[3], "y")
    return {"path": save(fig, f"speech_{path.stem}.png", out_dir), "voiced_ratio": float(voiced.mean()),
            "median_f0_hz": float(np.nanmedian(f0))}


if __name__ == "__main__":
    result = run(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WAV)
    print(result)
    open_image(result["path"])  # direct execution only; run() itself never opens a window
