"""Reconstruction from a cepstrogram: low / high quefrency parts as time signals.

Writes reconstruction.png with four panels for a speech WAV (default data/mao.wav, made by
synth_aivis.py). The time signal is rebuilt as: complex spectrogram X of the original times the gain
|X_l| / |X| (X_l: the spectrum carried by the liftered cepstrogram), inverse FFT, overlap-add; the
phase of the original is kept and only the magnitude changes (docs/definitions.md, section 9).
(a) no lifter: the original signal comes back, (b) the low / high quefrency parts as time signals
(low + high = original), (c) what the low part is: the frame spectra of the low signal move toward
the envelope but do not equal it (the phase is the original one, so the reconstructed spectrogram is
not consistent), and the cepstra of the three signals show where the pitch peak ends up,
(d) the whole file.
The settings below are the defaults of run(); edit them here.
"""
import numpy as np

import cepstrum as cp
from _common import DEFAULT_WAV, QLABEL, load_segment, open_image, period_axis, plt, save, to_db

WAV = DEFAULT_WAV
OUT_DIR = None            # None -> examples/output
SEG_START = 2.9           # excerpt start [s] for (a)-(c) (a steady vowel in mao.wav)
FMAX = 8000               # spectrum range shown [Hz]
ZOOM_MS = 30              # length of the waveform comparisons [ms]
Q_MAX = 8                 # quefrency range shown in (c) [kHz^-1]
# (a) round trip without lifter
A_DURATION = 0.1          # excerpt length [s]
A_FRAME_LEN, A_HOP = 1024, 256
# (b), (c) low / high quefrency parts
C_DURATION = 0.1          # excerpt length [s]
C_FRAME_LEN, C_HOP = 2048, 512
SPLIT_KHZ_INV = 1         # low/high split [kHz^-1] (ripples wider than 1 kHz = envelope)
# (d) whole file
D_FRAME_LEN, D_HOP = 1024, 256


def run(path=WAV, out_dir=OUT_DIR):
    """Run the four reconstruction experiments and plot them.

    Args:
        path: Speech WAV (16-bit PCM). Default ``WAV`` (data/mao.wav).
        out_dir: Directory for the PNG. Default ``OUT_DIR`` (``None`` = examples/output).

    Returns:
        Dict with ``"path"`` (the PNG), ``"roundtrip_error"`` (max abs error of (a)),
        ``"whole_file_roundtrip_error"`` (of (d)) and ``"fine_structure_ratio"`` (rms distance of the
        re-analysed frame log-spectra of the low signal to the envelope, divided by that of the
        original; below 1 means the frame spectra moved toward the envelope). The other settings
        are the module constants above.
    """
    # (a) no lifter -> the original signal
    x, fs = load_segment(path, SEG_START, A_DURATION)
    ya = cp.reconstruct_cepstrogram(cp.cepstrogram(x, fs, frame_len=A_FRAME_LEN, hop=A_HOP))
    err_a = float(np.max(np.abs(x - ya)))

    # (b) low / high quefrency parts as time signals
    xc, _ = load_segment(path, SEG_START, C_DURATION)
    rc = cp.analyze_cepstrogram(xc, fs, frame_len=C_FRAME_LEN, hop=C_HOP, khz_inv=SPLIT_KHZ_INV)
    low, high = rc.reconstruct("low"), rc.reconstruct("high")

    # (c) the frame spectra of the low signal, re-analysed with the same framing
    half = rc.full.n_fft // 2 + 1
    ln_env = np.fft.fft(rc.low.data, axis=0).real[:half]                 # target: envelope of every frame
    ln_x = np.log(np.maximum(np.abs(rc.full.spectrogram), 1e-300))        # original
    re = cp.cepstrogram(low, fs, frame_len=C_FRAME_LEN, hop=C_HOP)
    ln_y = np.fft.fft(re.data, axis=0).real[:half]                        # actual, after reconstruction
    inner = slice(2, -2)                                                  # skip the edge frames
    dist = lambda ln: float(np.sqrt(np.mean((ln[:, inner] - ln_env[:, inner]) ** 2)))
    ratio = dist(ln_y) / dist(ln_x)
    mid = ln_x.shape[1] // 2
    f = np.fft.rfftfreq(rc.full.n_fft, 1 / fs)

    # (d) whole file
    y, _ = load_segment(path)
    rd = cp.analyze_cepstrogram(y, fs, frame_len=D_FRAME_LEN, hop=D_HOP, khz_inv=SPLIT_KHZ_INV)
    yd_full, yd_low = rd.reconstruct("full"), rd.reconstruct("low")
    err_d = float(np.max(np.abs(y - yd_full)))

    zoom = int(ZOOM_MS * 1e-3 * fs)
    ms = lambda n: np.arange(n) / fs * 1e3
    fig, ax = plt.subplots(4, 2, figsize=(12, 12))
    ax[0, 0].plot(ms(zoom), x[:zoom], label="original")
    ax[0, 0].plot(ms(zoom), ya[:zoom], "--", lw=0.8, label="reconstruction")
    ax[0, 0].set(title=f"(a) no lifter: the original comes back, max err={err_a:.1e}", xlabel="time [ms]")
    ax[0, 0].legend(fontsize=7)
    ax[0, 1].semilogy(ms(len(x)), np.abs(x - ya) + 1e-18)
    ax[0, 1].set(title="(a) absolute error", xlabel="time [ms]")

    ax[1, 0].plot(ms(zoom), xc[:zoom], lw=0.9, label="original")
    ax[1, 0].plot(ms(zoom), low[:zoom], lw=0.8, label="low part (envelope magnitude)")
    ax[1, 0].plot(ms(zoom), high[:zoom], lw=0.6, alpha=0.8, label="high part = original - low")
    ax[1, 0].set(title=f"(b) low / high split at {SPLIT_KHZ_INV} kHz$^{{-1}}$ as time signals", xlabel="time [ms]")
    ax[1, 0].legend(fontsize=7)
    fc = np.fft.rfftfreq(len(xc), 1 / fs)
    for sig, name, kw in ((xc, "original", {"color": "0.6", "lw": 1.2}), (low, "low", {"color": "r", "lw": 0.8}),
                          (high, "high", {"color": "C0", "lw": 0.5, "alpha": 0.7})):
        ax[1, 1].plot(fc, to_db(np.log(np.abs(np.fft.rfft(sig * np.hanning(len(sig)))) + 1e-12)), label=name, **kw)
    ax[1, 1].set(title="(b) long-window spectra of the three signals", xlabel="frequency [Hz]", ylabel="dB",
                 xlim=(0, FMAX), ylim=(-80, 40))
    ax[1, 1].legend(fontsize=7)

    ax[2, 0].plot(f, to_db(ln_x[:, mid]), color="0.6", lw=0.8, label="original |X|")
    ax[2, 0].plot(f, to_db(ln_env[:, mid]), "r", lw=1.6, label="target: envelope |X_l|")
    ax[2, 0].plot(f, to_db(ln_y[:, mid]), "C0", lw=0.8, label="re-analysed low signal")
    ax[2, 0].set(title=f"(c) one frame: spectrum moves toward the envelope (distance ratio {ratio:.2f})",
                 xlabel="frequency [Hz]", ylabel="dB", xlim=(0, FMAX))
    ax[2, 0].legend(fontsize=7)
    for sig, name, kw in ((xc, "original", {"color": "0.6", "lw": 1.4}), (low, "low signal", {"color": "r", "lw": 0.8}),
                          (high, "high signal", {"color": "C0", "lw": 0.6, "alpha": 0.8})):
        c = cp.cepstrum(sig, fs, window="hann")
        q = cp.quefrency_axis(c, "kHz^-1")[: c.n_fft // 2]
        m = (q > 0.5) & (q < Q_MAX)
        ax[2, 1].plot(q[m], c.data[: c.n_fft // 2][m], label=name, **kw)
    ax[2, 1].set(title="(c) cepstra of the three signals: the pitch peak is in the high signal", xlabel=QLABEL)
    ax[2, 1].legend(fontsize=7)
    period_axis(ax[2, 1])

    ty = np.arange(len(y)) / fs
    ax[3, 0].plot(ty, y, lw=0.4, label="original")
    ax[3, 0].plot(ty, yd_full, "--", lw=0.3, label="reconstruction (no lifter)")
    ax[3, 0].set(title=f"(d) whole file round trip, max err={err_d:.1e}", xlabel="time [s]")
    ax[3, 0].legend(fontsize=7)
    ax[3, 1].plot(ty, yd_low, lw=0.4)
    ax[3, 1].set(title=f"(d) low part ({SPLIT_KHZ_INV} kHz$^{{-1}}$): envelope magnitude, original phase", xlabel="time [s]")
    return {"path": save(fig, "reconstruction.png", out_dir), "roundtrip_error": err_a,
            "whole_file_roundtrip_error": err_d, "fine_structure_ratio": ratio}


if __name__ == "__main__":
    result = run()
    print(result)
    open_image(result["path"])
