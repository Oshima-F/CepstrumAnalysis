import numpy as np
import pytest

import cepstrum as cp

FS = 16000.0
rng = np.random.default_rng(3)


def _voiced(n=1024, f0=125.0):
    x = np.zeros(n)
    x[:: int(FS / f0)] = 1.0
    for fr, bw in ((700, 80), (1200, 90)):
        r = np.exp(-np.pi * bw / FS)
        a1, a2 = -2 * r * np.cos(2 * np.pi * fr / FS), r * r
        y = np.zeros(n)
        for i in range(n):
            y[i] = x[i] - a1 * (y[i - 1] if i else 0) - a2 * (y[i - 2] if i > 1 else 0)
        x = y
    return x


@pytest.mark.parametrize("fn", [cp.analyze_real, cp.analyze_complex, cp.analyze_power])
def test_low_plus_high_is_full(fn):
    x = rng.standard_normal(256)
    r = fn(x, FS, khz_inv=3, n_fft=512) if fn is cp.analyze_complex else fn(x, FS, khz_inv=3)
    np.testing.assert_allclose(r.low.data + r.high.data, r.full.data, atol=1e-12)


def test_low_and_high_support():
    r = cp.analyze_real(_voiced(), FS, khz_inv=3)
    q = np.abs(cp.quefrency_axis(r.full, "kHz^-1"))
    assert np.all(r.low.data[q > 3] == 0)
    assert np.all(r.high.data[q <= 3] == 0)
    assert r.low.data[0] == r.full.data[0]  # c[0] (level) stays in the low part


def test_cutoff_forms_equivalent():
    x = _voiced()
    ref = cp.analyze_real(x, FS, cutoff=0.003)
    for kw in ({"khz_inv": 3}, {"period_hz": 1000 / 3}, {"lifter": cp.lowpass(0.003)}):
        np.testing.assert_array_equal(cp.analyze_real(x, FS, **kw).low.data, ref.low.data)
    with pytest.raises(ValueError):
        cp.analyze_real(x, FS, cutoff=0.003, khz_inv=3)
    with pytest.raises(ValueError):
        cp.analyze_real(x, FS, lifter=cp.lowpass(0.003), cutoff=0.003)


def test_no_lifter_means_no_split():
    r = cp.analyze_real(_voiced(), FS)
    np.testing.assert_array_equal(r.low.data, r.full.data)
    assert not r.high.data.any()


def test_envelope_and_fine_structure_spectra():
    r = cp.analyze_real(_voiced(), FS, window="hann", khz_inv=3)
    env, fine, full = r.log_spectrum("low"), r.log_spectrum("high"), r.log_spectrum("full")
    np.testing.assert_allclose(env + fine, full, atol=1e-9)
    assert r.freqs.shape == full.shape
    assert np.std(np.diff(env)) < np.std(np.diff(full))


def test_complex_full_reconstruction_and_homomorphic_parts():
    x = _voiced()
    r = cp.analyze_complex(x, FS, khz_inv=3, n_fft=2048)
    np.testing.assert_allclose(r.reconstruct("full"), x, atol=1e-8)
    lo, hi = r.reconstruct("low"), r.reconstruct("high")
    # complex cepstra add <-> signals convolve (circularly, n_fft points)
    assert len(lo) == len(hi) == 2048
    conv = np.fft.ifft(np.fft.fft(lo) * np.fft.fft(hi)).real
    np.testing.assert_allclose(conv[: len(x)], x, atol=1e-8)
    np.testing.assert_allclose(conv[len(x):], 0, atol=1e-8)


def test_power_has_no_spectrum_or_reconstruction():
    r = cp.analyze_power(rng.standard_normal(256), FS, khz_inv=3)
    with pytest.raises(ValueError):
        r.log_spectrum()
    with pytest.raises(ValueError):
        r.reconstruct()


def test_spectrum_input():
    x = rng.standard_normal(128)
    spec = cp.Spectrum(np.abs(np.fft.fft(x))[:65], FS, 128, "amplitude", "one")
    a = cp.analyze_real(spec, khz_inv=2)
    b = cp.analyze_real(x, FS, khz_inv=2)
    np.testing.assert_allclose(a.low.data, b.low.data, atol=1e-12)


def test_cepstrogram_tracks_gliding_pitch():
    fs, dur = 16000, 1.5
    n = int(fs * dur)
    f0 = np.linspace(100, 200, n)
    phase = np.cumsum(f0) / fs
    y = np.zeros(n)
    y[np.where(np.diff(np.floor(phase)) > 0)[0] + 1] = 1.0
    for fr, bw in ((700, 80), (1200, 90), (2600, 120)):
        r_ = np.exp(-np.pi * bw / fs)
        a1, a2 = -2 * r_ * np.cos(2 * np.pi * fr / fs), r_ * r_
        z = np.zeros(n)
        for i in range(n):
            z[i] = y[i] - a1 * (z[i - 1] if i else 0) - a2 * (z[i - 2] if i > 1 else 0)
        y = z
    r = cp.analyze_cepstrogram(y, fs, frame_len=1024, hop=128, khz_inv=3)
    q = cp.quefrency_axis(r.full, "kHz^-1")[:512]
    band = (q > 4) & (q < 12)  # f0 range 83-250 Hz
    ok = (r.full.times > 0.1) & (r.full.times < dur - 0.1)
    peak_q = q[band][r.high.data[:512][band][:, ok].argmax(axis=0)]
    true_q = 1e3 / np.interp(r.full.times[ok], np.arange(n) / fs, f0)
    assert np.median(np.abs(peak_q - true_q)) < 0.05  # kHz^-1 (= ms)


def test_cepstrogram_analysis():
    x = _voiced(4096)
    r = cp.analyze_cepstrogram(x, FS, frame_len=512, hop=128, kind="complex", khz_inv=3)
    assert r.low.data.shape == r.full.data.shape
    np.testing.assert_allclose(r.low.data + r.high.data, r.full.data, atol=1e-12)
    np.testing.assert_allclose(r.reconstruct("full"), x, atol=1e-6)
    env = r.log_spectrum("low")
    assert env.shape == (257, r.full.data.shape[1])
