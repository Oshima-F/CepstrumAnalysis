"""Reconstruction: exact round trip, liftered reconstruction, echo removal, cepstrogram OLA.

Input
    speech WAV (default data/mao.wav, made by synth_aivis.py): excerpts for (a)-(c), the whole file for (d)
Output
    run() -> {"path": PNG with 4 panels (a)-(d), written to OUT_DIR as reconstruction.png,
              "roundtrip_error": max abs error of (a), "cepstrogram_roundtrip_error": of (d),
              "echo_corr_before"/"echo_corr_after": correlation with the clean excerpt in (b)}
    (a) complex-cepstrum round trip  (b) echo removal by notch lifter on the complex cepstrum
    (artificial echo added to the excerpt)  (c) real cepstrum -> lowpass -> minimum-phase signal
    (d) complex cepstrogram OLA round trip
Parameters (edit below, or pass to run())
"""
import numpy as np

import cepstrum as cp
from _common import DEFAULT_WAV, QLABEL, load_segment, open_image, period_axis, plt, save, to_db

WAV = DEFAULT_WAV
OUT_DIR = None            # None -> examples/output
SEG_START = 2.9           # excerpt start [s] for (a)-(c) (a steady vowel in mao.wav)
# (a) round trip
A_DURATION = 0.1          # excerpt length [s]
# (b) echo removal
ECHO_DELAY, ECHO_GAIN = 0.010, 0.6   # [s], relative amplitude of the added echo
B_DURATION = 0.1                     # excerpt length [s]
B_N_FFT = 32768                      # FFT length of the complex cepstrum (>> length + echo delay: less aliasing)
NOTCH_ORDERS = 7                     # echo harmonics k*delay (k = 1..NOTCH_ORDERS) to remove
NOTCH_HALFWIDTH = 1.5                # notch half width [samples]
# (c) envelope from the real cepstrum
C_DURATION = 0.05
ENVELOPE_KHZ_INV = 1                 # lowpass cutoff [kHz^-1] (ripples wider than 1 kHz)
FMAX = 8000                          # spectrum range shown [Hz]
IR_MS = 10                           # length of the shown minimum-phase impulse response [ms]
# (d) cepstrogram round trip over the whole file
FRAME_LEN, HOP = 1024, 256           # [samples]
D_LOWPASS_KHZ_INV = 1


def run(path=WAV, out_dir=OUT_DIR):
    """Returns {"path", "roundtrip_error", "cepstrogram_roundtrip_error", "echo_corr_before", "echo_corr_after"}."""
    # (a) complex cepstrum round trip (no liftering) -> exact
    x, fs = load_segment(path, SEG_START, A_DURATION)
    c = cp.cepstrum(x, fs, kind="complex", n_fft=2 * len(x))
    xr = cp.reconstruct(c)
    err_a = np.max(np.abs(x - xr))

    # (b) echo removal by notching the echo quefrencies out of the complex cepstrum
    seg, _ = load_segment(path, SEG_START, B_DURATION)
    d_s, d = ECHO_DELAY, int(ECHO_DELAY * fs)
    n = len(seg) + d  # keep the echo tail: the echoed signal must be exactly seg * (1 + g z^-d)
    src = np.r_[seg, np.zeros(d)]
    echoed = src.copy()
    echoed[d:] += ECHO_GAIN * seg
    ce = cp.cepstrum(echoed, fs, kind="complex", n_fft=B_N_FFT)

    def echo_notch(q):
        w = np.ones_like(q)
        for k in range(1, NOTCH_ORDERS + 1):
            w[np.abs(q - k * d_s) < NOTCH_HALFWIDTH / fs] = 0
        return w

    clean = cp.reconstruct(ce, echo_notch)
    corr_before, corr_after = (np.corrcoef(a[:n], src)[0, 1] for a in (echoed, clean))

    # (c) real cepstrum, lowpass lifter, minimum-phase reconstruction
    v, _ = load_segment(path, SEG_START, C_DURATION)
    cr = cp.cepstrum(v, fs, kind="real", window="hann")
    env = cp.reconstruct(cr, cp.lowpass(khz_inv=ENVELOPE_KHZ_INV))  # minimum-phase impulse response of the envelope

    # (d) cepstrogram: exact round trip with complex kind over the whole file
    y, _ = load_segment(path)
    cg = cp.cepstrogram(y, fs, frame_len=FRAME_LEN, hop=HOP, kind="complex")
    yr = cp.reconstruct_cepstrogram(cg)
    y_lp = cp.reconstruct_cepstrogram(cg, cp.lowpass(khz_inv=D_LOWPASS_KHZ_INV))
    err_d = np.max(np.abs(y - yr))

    fig, ax = plt.subplots(4, 2, figsize=(12, 12))
    t = np.arange(len(x)) / fs * 1e3
    ax[0, 0].plot(t, x, label="original"); ax[0, 0].plot(t, xr, "--", label="reconstructed")
    ax[0, 0].set(title=f"(a) complex cepstrum round trip, max err={err_a:.1e}", xlabel="time [ms]"); ax[0, 0].legend()
    ax[0, 1].semilogy(t, np.abs(x - xr) + 1e-18)
    ax[0, 1].set(title="(a) absolute error", xlabel="time [ms]")

    te = np.arange(n) / fs * 1e3
    ax[1, 0].plot(te, src, lw=0.8, label="speech"); ax[1, 0].plot(te, echoed, lw=0.6, alpha=0.7, label="with echo")
    ax[1, 0].plot(te, clean[:n], "--", lw=0.8, label="notch-liftered")
    ax[1, 0].set(title="(b) echo removal in time domain", xlabel="time [ms]"); ax[1, 0].legend(fontsize=7)
    qs = np.fft.fftshift(cp.quefrency_axis(ce, "kHz^-1"))
    ax[1, 1].plot(qs, np.fft.fftshift(ce.data), lw=0.6)
    ax[1, 1].set(title=f"(b) complex cepstrum; corr with speech {corr_before:.2f} -> {corr_after:.2f}",
                 xlabel=QLABEL, xlim=(-5, 90))
    period_axis(ax[1, 1])

    f = np.fft.rfftfreq(len(v), 1 / fs)
    ax[2, 0].plot(f, to_db(np.log(np.abs(np.fft.rfft(v * np.hanning(len(v)))) + 1e-12)), color="0.6", label="original")
    ax[2, 0].plot(f, to_db(np.log(np.abs(np.fft.rfft(env, len(v))) + 1e-12)), "r", label="reconstructed envelope")
    ax[2, 0].set(title=f"(c) real cepstrum, lowpass {ENVELOPE_KHZ_INV} kHz$^{{-1}}$, min-phase rebuild (spectrum)",
                 xlabel="frequency [Hz]", ylabel="dB", xlim=(0, FMAX))
    ax[2, 0].legend()
    k = int(IR_MS * 1e-3 * fs)
    ax[2, 1].plot(np.arange(k) / fs * 1e3, env[:k])
    ax[2, 1].set(title="(c) min-phase envelope impulse response", xlabel="time [ms]")

    ty = np.arange(len(y)) / fs
    ax[3, 0].plot(ty, y, label="original"); ax[3, 0].plot(ty, yr, "--", lw=0.8, label="OLA reconstruction")
    ax[3, 0].set(title=f"(d) cepstrogram round trip, max err={err_d:.1e}", xlabel="time [s]"); ax[3, 0].legend()
    ax[3, 1].plot(ty, y_lp, lw=0.8)
    ax[3, 1].set(title=f"(d) cepstrogram lowpass ({D_LOWPASS_KHZ_INV} kHz$^{{-1}}$) reconstruction", xlabel="time [s]")
    return {"path": save(fig, "reconstruction.png", out_dir), "roundtrip_error": float(err_a),
            "cepstrogram_roundtrip_error": float(err_d),
            "echo_corr_before": float(corr_before), "echo_corr_after": float(corr_after)}


if __name__ == "__main__":
    result = run()
    print(result)
    open_image(result["path"])
