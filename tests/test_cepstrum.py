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
    ref = cp.cepstrum(x, FS, kind="real")
    for kind, d in (("complex", data), ("amplitude", np.abs(data)), ("power", np.abs(data) ** 2)):
        c = cp.cepstrum(cp.Spectrum(d, FS, 64, kind, sided), kind="real")
        np.testing.assert_allclose(c.data, ref.data, atol=1e-10)


def test_complex_cepstrum_needs_phase():
    spec = cp.Spectrum(np.ones(33), FS, 64, "amplitude", "one")
    with pytest.raises(ValueError):
        cp.cepstrum(spec, kind="complex")


@pytest.mark.parametrize("n", [64, 100])
def test_complex_roundtrip_exact(n):
    x = rng.standard_normal(n)
    x[0] = 1.0
    x = np.roll(x, 5)  # non-trivial delay
    for sig in (x, -x):  # also negative DC gain
        c = cp.cepstrum(sig, FS, kind="complex", n_fft=2 * n)
        np.testing.assert_allclose(cp.reconstruct(c), sig, atol=1e-9)


def test_echo_peak_in_real_cepstrum():
    n, d = 2048, 100
    x = rng.standard_normal(n)
    y = x.copy()
    y[d:] += 0.6 * x[:-d]
    c = cp.cepstrum(y, FS, window="hann")
    peak = np.argmax(c.data[20:n // 2]) + 20
    assert peak == d
    assert c.quefrency[peak] == pytest.approx(d / FS)


def test_minimum_phase_real_roundtrip():
    h = np.zeros(256)
    h[:3] = [1.0, -0.5, 0.2]  # minimum phase (zeros inside unit circle)
    c = cp.cepstrum(h, FS, kind="real", n_fft=256)
    np.testing.assert_allclose(cp.reconstruct(c), h, atol=1e-8)


def test_lowpass_lifter_smooths_log_spectrum():
    x = rng.standard_normal(1024)
    c = cp.cepstrum(x, FS)
    smooth = cp.apply_lifter(c, cp.lowpass(0.02))
    assert smooth.data[25:-25].max() == 0.0
    ls_raw = np.fft.fft(c.data).real
    ls_sm = np.fft.fft(smooth.data).real
    assert np.std(np.diff(ls_sm)) < np.std(np.diff(ls_raw))


def test_power_not_invertible():
    c = cp.cepstrum(rng.standard_normal(64), FS, kind="power")
    with pytest.raises(ValueError):
        cp.reconstruct(c)


@pytest.mark.parametrize("kind", ["complex", "real"])
def test_cepstrogram_shapes_and_axes(kind):
    x = rng.standard_normal(1000)
    cg = cp.cepstrogram(x, FS, frame_len=128, hop=32, kind=kind)
    assert cg.data.shape[0] == 128
    assert cg.data.shape[1] == len(cg.times)
    assert cg.quefrency.shape == (128,)


def test_cepstrogram_complex_roundtrip():
    x = rng.standard_normal(1000)
    cg = cp.cepstrogram(x, FS, frame_len=128, hop=32, kind="complex")
    np.testing.assert_allclose(cp.reconstruct_cepstrogram(cg), x, atol=1e-6)


def test_cepstrogram_lifter_applies_per_frame():
    x = rng.standard_normal(600)
    cg = cp.cepstrogram(x, FS, frame_len=128, hop=64, kind="real")
    lo = cp.apply_lifter(cg, cp.lowpass(0.01))
    assert lo.data.shape == cg.data.shape
    assert np.all(lo.data[20:-20] == 0)
    y = cp.reconstruct_cepstrogram(lo)
    assert y.shape == x.shape
