"""Liftering: separate spectral envelope (low quefrency) from pitch structure (high quefrency).

Input
    excerpt of a speech WAV (default data/mao.wav, made by synth_aivis.py): SEG_START, SEG_DUR
Output
    run() -> {"path": PNG of cepstrum + lifter weights, envelope extraction, other lifters,
              "f0_hz": pitch from the strongest real-cepstrum peak within F0_RANGE}
    PNG is written to OUT_DIR as liftering.png
Parameters (edit below, or pass to run())
"""
import numpy as np

import cepstrum as cp
from _common import (DEFAULT_WAV, QLABEL, f0_from_cepstrum, load_segment, open_image, period_axis, plt, save,
                     to_db)

WAV = DEFAULT_WAV
OUT_DIR = None        # None -> examples/output
SEG_START = 2.9       # excerpt start [s] (a steady vowel in mao.wav)
SEG_DUR = 0.05        # excerpt length [s]
WINDOW = "hann"       # analysis window
SPLIT_KHZ_INV = 1     # low/high split [kHz^-1]: ripples wider than 1 kHz form the envelope
F0_RANGE = (100, 700)  # pitch search range [Hz]; the notch/bandpass lifters follow the detected f0
BAND_RATIO = 1.2      # bandpass keeps ripple periods f0/BAND_RATIO .. f0*BAND_RATIO
NOTCH_WIDTH_KHZ_INV = 0.5  # notch width [kHz^-1]
Q_MAX = 8             # quefrency range shown [kHz^-1]
FMAX = 8000           # spectrum range shown [Hz]


def run(path=WAV, out_dir=OUT_DIR, start=SEG_START, dur=SEG_DUR, window=WINDOW, split_khz_inv=SPLIT_KHZ_INV,
        f0_range=F0_RANGE, band_ratio=BAND_RATIO, notch_width_khz_inv=NOTCH_WIDTH_KHZ_INV, q_max=Q_MAX):
    """Returns {"path", "f0_hz"}."""
    x, fs = load_segment(path, start, dur)
    n = len(x)
    # real-cepstrum analysis split at the given quefrency
    r = cp.analyze_real(x, fs, window=window, khz_inv=split_khz_inv)
    c = r.full
    f0 = f0_from_cepstrum(c, f0_range)
    nt = cp.notch(period_hz=f0, width_khz_inv=notch_width_khz_inv)
    split_hz = 1000 / split_khz_inv
    lifters = {  # specified by ripple period [Hz] / kHz^-1 (see docs/requirements.md)
        f"lowpass: ripples wider than {split_hz:.0f} Hz": r.lifter,
        f"highpass: ripples narrower than {split_hz:.0f} Hz": lambda q: 1 - r.lifter(q),
        f"bandpass: {f0 / band_ratio:.0f}-{f0 * band_ratio:.0f} Hz ripples": cp.bandpass(
            period_hz=(f0 * band_ratio, f0 / band_ratio)),
        f"notch: {f0:.0f} Hz ripple (f0)": nt,
    }

    f = r.freqs
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    q = cp.quefrency_axis(c, "kHz^-1")[: n // 2]
    ax[0].plot(q, c.data[: n // 2], "k", label="real cepstrum")
    peak = c.data[: n // 2][q > 0.5].max()
    ax[0].set(title=f"Real cepstrum (f0 = {f0:.0f} Hz -> peak at {1000 / f0:.1f} kHz$^{{-1}}$) and lifter weights",
              xlabel=QLABEL, xlim=(0, q_max), ylim=(-0.3 * peak, 1.4 * peak))
    period_axis(ax[0])
    tw = ax[0].twinx()
    for name, lf in lifters.items():
        tw.plot(q, cp.lifter_weights(lf, n, fs)[: n // 2], lw=1, label=f"lifter: {name}")
    tw.set(ylabel="lifter weight", ylim=(-0.05, 1.5))
    tw.legend(fontsize=7, loc="upper right")
    ax[0].legend(loc="upper center", fontsize=7)

    ax[1].plot(f, r.log_spectrum("full"), color="0.6", label="original")
    ax[1].plot(f, r.log_spectrum("low"), "r", lw=2, label="low part (envelope)")
    ax[1].set(title="Log spectrum: envelope extraction", xlabel="frequency [Hz]", ylabel="dB", xlim=(0, FMAX))
    ax[1].legend()

    ax[2].plot(f, r.log_spectrum("high"), lw=0.8, label="high part (harmonic fine structure)")
    ax[2].plot(f, to_db(np.fft.fft(cp.apply_lifter(c, nt).data).real[: len(f)]), lw=0.8,
               label=f"notch @ {f0:.0f} Hz ripple (pitch removed)")
    ax[2].set(title="Log spectrum after other lifters", xlabel="frequency [Hz]", ylabel="dB", xlim=(0, FMAX))
    ax[2].legend()
    return {"path": save(fig, "liftering.png", out_dir), "f0_hz": f0}


if __name__ == "__main__":
    result = run()
    print(result)
    open_image(result["path"])
