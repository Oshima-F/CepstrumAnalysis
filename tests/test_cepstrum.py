import numpy as np
import pytest

import cepstrum as cp

FS = 1000.0
rng = np.random.default_rng(0)


def test_multichannel_is_averaged():
    a, b = rng.standard_normal(128), rng.standard_normal(128)
    c1 = cp.cepstrum(np.stack([a, b]), FS)
    c2 = cp.cepstrum((a + b) / 2, FS)
    np.testing.assert_allclose(c1.data, c2.data)


@pytest.mark.parametrize("sided", ["one", "two"])
def test_spectrum_input_matches_signal(sided):
    x = rng.standard_normal(64)
    X = np.fft.fft(x)
    data = X[:33] if sided == "one" else X
    ref = cp.cepstrum(x, FS)
    for kind, d in (("complex", data), ("amplitude", np.abs(data)), ("power", np.abs(data) ** 2)):
        c = cp.cepstrum(cp.Spectrum(d, FS, 64, kind, sided))
        np.testing.assert_allclose(c.data, ref.data, atol=1e-10)


def test_phase_is_ignored():
    x = rng.standard_normal(64)
    a = cp.cepstrum(x, FS)
    b = cp.cepstrum(x[::-1].copy(), FS)  # same magnitude spectrum, different phase
    np.testing.assert_allclose(a.data, b.data, atol=1e-10)


def test_real_cepstrum_is_even():
    c = cp.cepstrum(rng.standard_normal(128), FS)
    np.testing.assert_allclose(c.data[1:], c.data[1:][::-1], atol=1e-12)


def test_spectral_ripple_gives_a_cepstral_peak_at_its_quefrency():
    # x plus a copy delayed by d samples has a log-spectrum ripple of period fs/d, i.e. a peak at q = d/fs
    n, d = 2048, 100
    x = rng.standard_normal(n)
    y = x.copy()
    y[d:] += 0.6 * x[:-d]
    c = cp.cepstrum(y, FS, window="hann")
    peak = np.argmax(c.data[20:n // 2]) + 20
    assert peak == d
    assert c.quefrency[peak] == pytest.approx(d / FS)


def test_unliftered_roundtrip_returns_the_magnitude_spectrum():
    x = rng.standard_normal(256)
    c = cp.cepstrum(x, FS)
    np.testing.assert_allclose(np.exp(np.fft.fft(c.data).real), np.abs(np.fft.fft(x)), rtol=1e-9)
    np.testing.assert_allclose(np.fft.fft(c.data).imag, 0, atol=1e-10)  # real and even -> real spectrum


def test_lowpass_lifter_smooths_log_spectrum():
    x = rng.standard_normal(1024)
    c = cp.cepstrum(x, FS)
    smooth = cp.apply_lifter(c, cp.lowpass(0.02))
    assert smooth.data[25:-25].max() == 0.0
    ls_raw = np.fft.fft(c.data).real
    ls_sm = np.fft.fft(smooth.data).real
    assert np.std(np.diff(ls_sm)) < np.std(np.diff(ls_raw))


@pytest.mark.parametrize("cutoff", [0.0, 0.02, 0.0205])  # 0.02 s is exactly bin 20 at FS = 1000
def test_lowpass_and_highpass_are_complements(cutoff):
    n = 1024
    lo = cp.lifter_weights(cp.lowpass(cutoff), n, FS)
    hi = cp.lifter_weights(cp.highpass(cutoff), n, FS)
    np.testing.assert_array_equal(lo + hi, np.ones(n))
    if cutoff == 0.02:
        assert lo[20] == 1 and hi[20] == 0  # the cutoff bin belongs to the low side


def test_bandpass_is_highpass_times_lowpass():
    n = 1024
    a, b = 0.01, 0.02  # both exactly on bins (10 and 20) at FS = 1000
    band = cp.lifter_weights(cp.bandpass(a, b), n, FS)
    prod = cp.lifter_weights(cp.highpass(a), n, FS) * cp.lifter_weights(cp.lowpass(b), n, FS)
    np.testing.assert_array_equal(band, prod)
    assert band[10] == 0 and band[11] == 1 and band[20] == 1 and band[21] == 0


def test_cepstrogram_shapes_and_axes():
    x = rng.standard_normal(1000)
    cg = cp.cepstrogram(x, FS, frame_len=128, hop=32)
    assert cg.data.shape[0] == 128
    assert cg.data.shape[1] == len(cg.times)
    assert cg.quefrency.shape == (128,)


def test_cepstrogram_lifter_applies_per_frame():
    x = rng.standard_normal(600)
    cg = cp.cepstrogram(x, FS, frame_len=128, hop=64)
    lo = cp.apply_lifter(cg, cp.lowpass(0.01))
    assert lo.data.shape == cg.data.shape
    assert np.all(lo.data[20:-20] == 0)
    y = cp.reconstruct_cepstrogram(lo)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))
