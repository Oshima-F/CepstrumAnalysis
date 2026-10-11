import numpy as np
import pytest

import cepstrum as cp

FS = 16000.0
rng = np.random.default_rng(21)


def _x(n=4000):
    return rng.standard_normal(n) * np.exp(-np.arange(n) / (n / 2))


@pytest.mark.parametrize("kw", [
    {"frame_len": 256, "hop": 64},                                  # hann, 75 % overlap
    {"frame_len": 256, "hop": 128, "window": "hamming"},
    {"frame_len": 256, "hop": 128, "window": "rect"},
    {"frame_len": 255, "hop": 64, "n_fft": 511},                   # odd lengths
    {"frame_len": 200, "hop": 50, "n_fft": 1024},                  # zero padding
])
def test_without_lifter_the_original_signal_comes_back(kw):
    x = _x()
    cg = cp.cepstrogram(x, FS, **kw)
    np.testing.assert_allclose(cp.reconstruct_cepstrogram(cg), x, atol=1e-9)


def test_gain_is_the_ratio_of_magnitudes_and_the_phase_is_kept():
    n = 512
    x = _x(n)
    cg = cp.cepstrogram(x, FS, frame_len=n, hop=n, window="rect")  # one frame, no overlap
    assert cg.data.shape[1] == 1
    lifter = cp.lowpass(khz_inv=2)
    y = cp.reconstruct_cepstrogram(cg, lifter)

    X, Y = np.fft.fft(x), np.fft.fft(y)
    lifted = cp.apply_lifter(cg, lifter)
    mag_l = np.exp(np.fft.fft(lifted.data[:, 0]).real)  # |X_l| from the liftered cepstrum
    np.testing.assert_allclose(np.abs(Y), mag_l, rtol=1e-8)          # magnitude replaced by the lifter's
    np.testing.assert_allclose(np.angle(Y * np.conj(X)), 0, atol=1e-7)  # phase of the original kept


def test_low_plus_high_is_the_original_signal():
    x = _x()
    r = cp.analyze_cepstrogram(x, FS, frame_len=256, hop=64, khz_inv=2)
    low, high, full = r.reconstruct("low"), r.reconstruct("high"), r.reconstruct("full")
    np.testing.assert_allclose(low + high, full, atol=1e-12)
    np.testing.assert_allclose(full, x, atol=1e-9)
    assert not np.allclose(low, x, atol=1e-3)


def test_no_split_means_low_is_everything():
    x = _x()
    r = cp.analyze_cepstrogram(x, FS, frame_len=256, hop=64)
    np.testing.assert_allclose(r.reconstruct("low"), x, atol=1e-9)
    np.testing.assert_allclose(r.reconstruct("high"), 0, atol=1e-9)


def _voiced(n=8000, f0=125.0):
    """Glottal-pulse-like train through three formant resonators (a vowel-like signal)."""
    x = np.zeros(n)
    x[:: int(FS / f0)] = 1.0
    for fr, bw in ((700, 80), (1200, 90), (2600, 120)):
        r = np.exp(-np.pi * bw / FS)
        a1, a2 = -2 * r * np.cos(2 * np.pi * fr / FS), r * r
        y = np.zeros(n)
        for i in range(n):
            y[i] = x[i] - a1 * (y[i - 1] if i else 0) - a2 * (y[i - 2] if i > 1 else 0)
        x = y
    return x / np.abs(x).max()


def test_low_signal_moves_the_frame_spectra_toward_the_envelope_only_partly():
    # For an exactly periodic pulse train the original phase even keeps the pulses' timing, so the
    # long-window cepstrum of the low signal still shows the pitch peak (not so for natural speech).
    x = _voiced()
    r = cp.analyze_cepstrogram(x, FS, frame_len=1024, hop=256, khz_inv=3)  # pitch ripple 8 kHz^-1 > 3: high part
    half = 513
    ln_env = np.fft.fft(r.low.data, axis=0).real[:half]              # target: the envelope of every frame
    ln_x = np.log(np.maximum(np.abs(r.full.spectrogram), 1e-300))
    low = r.reconstruct("low")
    ln_y = np.fft.fft(cp.cepstrogram(low, FS, frame_len=1024, hop=256).data, axis=0).real[:half]
    fine = lambda ln: np.sqrt(np.mean((ln[:, 8:-8] - ln_env[:, 8:-8]) ** 2))
    assert fine(ln_y) < 0.9 * fine(ln_x)    # the fine structure of the frame spectra is reduced ...

    def pitch_peak(sig):                    # ... but not removed: the periodicity of the pulse train survives
        c = cp.cepstrum(sig[2000:6000], FS, window="hann")
        q = c.quefrency[:2000] * 1e3
        return c.data[:2000][(q > 6) & (q < 10)].max()
    assert pitch_peak(low) > 0.5 * pitch_peak(x)


def test_db_cepstrogram_gives_the_same_signal():
    x = _x()
    a = cp.cepstrogram(x, FS, frame_len=256, hop=64)
    b = cp.cepstrogram(x, FS, frame_len=256, hop=64, db=True)
    lifter = cp.lowpass(khz_inv=2)
    np.testing.assert_allclose(cp.reconstruct_cepstrogram(a, lifter), cp.reconstruct_cepstrogram(b, lifter),
                               atol=1e-9)


def test_apply_lifter_keeps_the_original_spectrogram():
    cg = cp.cepstrogram(_x(), FS, frame_len=256, hop=64)
    lo = cp.apply_lifter(cg, cp.lowpass(khz_inv=2))
    assert lo.spectrogram is cg.spectrogram


def test_silent_frames_stay_finite_and_silent():
    x = np.r_[_x(1500), np.zeros(2000), _x(1500)]
    y = cp.reconstruct_cepstrogram(cp.cepstrogram(x, FS, frame_len=256, hop=64), cp.lowpass(khz_inv=2))
    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y[1800:3200])) < 1e-6


def test_power_cepstrogram_is_not_invertible():
    cg = cp.cepstrogram(_x(), FS, frame_len=256, hop=64, kind="power")
    with pytest.raises(ValueError, match="not invertible"):
        cp.reconstruct_cepstrogram(cg)


def test_without_stored_spectrogram_reconstruction_is_refused():
    cg = cp.cepstrogram(_x(), FS, frame_len=256, hop=64, store_spectrogram=False)
    assert cg.spectrogram is None
    with pytest.raises(ValueError, match="phase"):
        cp.reconstruct_cepstrogram(cg)


def test_stored_spectrogram_shape():
    cg = cp.cepstrogram(_x(), FS, frame_len=256, hop=64, n_fft=512)
    assert cg.spectrogram.shape == (257, cg.data.shape[1])
    assert np.iscomplexobj(cg.spectrogram)
