"""Real / complex / power cepstrum of a speech excerpt.

Input
    excerpt of a speech WAV (default data/mao.wav, made by synth_aivis.py): SEG_START, SEG_DUR
Output
    run() -> {"path": PNG of waveform / log spectrum / real+power / complex cepstrum,
              "f0_hz": pitch from the strongest real-cepstrum peak within F0_RANGE}
    PNG is written to OUT_DIR as cepstrum_kinds.png
Parameters (edit below, or pass to run())
"""
import numpy as np

import cepstrum as cp
from _common import (DEFAULT_WAV, QLABEL, f0_from_cepstrum, load_segment, open_image, period_axis, plt, save,
                     to_db)

WAV = DEFAULT_WAV
OUT_DIR = None        # None -> examples/output
SEG_START = 2.9       # excerpt start [s] (a steady vowel in mao.wav)
SEG_DUR = 0.1         # excerpt length [s]
WINDOW = "hann"       # analysis window for real / power cepstrum ("hann" | "hamming" | "rect" | None)
F0_RANGE = (100, 700)  # pitch search range [Hz]
Q_MAX = 8             # quefrency range shown [kHz^-1]
FMAX = 8000           # spectrum range shown [Hz]


def run(path=WAV, out_dir=OUT_DIR, start=SEG_START, dur=SEG_DUR, window=WINDOW, f0_range=F0_RANGE,
        q_max=Q_MAX):
    """Plot the three cepstrum kinds. Returns {"path", "f0_hz"}."""
    y, fs = load_segment(path, start, dur)
    n = len(y)
    n_fft = n + n % 2  # complex cepstrum needs an even FFT length
    t = np.arange(n) / fs

    real = cp.cepstrum(y, fs, kind="real", n_fft=n_fft, window=window)
    power = cp.cepstrum(y, fs, kind="power", n_fft=n_fft, window=window)
    cplx = cp.cepstrum(y, fs, kind="complex", n_fft=n_fft)
    f0 = f0_from_cepstrum(real, f0_range)

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].plot(t * 1e3, y, lw=0.6)
    ax[0, 0].set(title=f"Speech excerpt ({start:.2f}-{start + dur:.2f} s)", xlabel="time [ms]")

    f = np.fft.rfftfreq(n_fft, 1 / fs)
    ax[0, 1].plot(f, to_db(np.fft.fft(real.data).real[: len(f)]), lw=0.6)
    ax[0, 1].set(title="Log magnitude spectrum (from real cepstrum)", xlabel="frequency [Hz]", ylabel="dB",
                 xlim=(0, FMAX))

    q = cp.quefrency_axis(real, "kHz^-1")[: n_fft // 2]
    m = (q > 0.5) & (q < q_max)
    rc, pc = real.data[: n_fft // 2], power.data[: n_fft // 2]
    ax[1, 0].plot(q[m], rc[m], label="real")
    ax[1, 0].plot(q[m], pc[m] / pc[m].max() * rc[m].max(), label="power (scaled)", alpha=0.7)
    ax[1, 0].axvline(1000 / f0, color="r", ls="--", lw=0.8, label=f"f0 = {f0:.0f} Hz")
    ax[1, 0].set(title="Real / power cepstrum", xlabel=QLABEL)
    ax[1, 0].legend()
    period_axis(ax[1, 0])

    qs = np.fft.fftshift(cp.quefrency_axis(cplx, "kHz^-1"))
    cs = np.fft.fftshift(cplx.data)
    m2 = np.abs(qs) < q_max
    ax[1, 1].plot(qs[m2], cs[m2])
    ax[1, 1].set(title=f"Complex cepstrum (delay={cplx.delay} samples, signed quefrency)", xlabel=QLABEL)
    return {"path": save(fig, "cepstrum_kinds.png", out_dir), "f0_hz": f0}


if __name__ == "__main__":
    result = run()
    print(result)
    open_image(result["path"])
