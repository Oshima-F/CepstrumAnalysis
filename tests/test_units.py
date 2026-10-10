import numpy as np
import pytest

import cepstrum as cp

FS = 16000.0


def test_conversions():
    assert cp.from_seconds(0.008, "kHz^-1") == pytest.approx(8.0)
    assert cp.from_seconds(0.008, "ms") == pytest.approx(8.0)
    assert cp.from_seconds(0.008, "Hz^-1") == pytest.approx(0.008)
    assert cp.from_seconds(0.008, "period_Hz") == pytest.approx(125.0)
    assert np.isinf(cp.from_seconds(0.0, "period_Hz"))
    assert cp.from_seconds(-0.008, "period_Hz") == pytest.approx(125.0)
    with pytest.raises(ValueError):
        cp.from_seconds(1.0, "furlongs")


def test_to_seconds_requires_exactly_one():
    assert cp.to_seconds(khz_inv=8) == pytest.approx(0.008)
    assert cp.to_seconds(period_hz=125) == pytest.approx(0.008)
    with pytest.raises(ValueError):
        cp.to_seconds()
    with pytest.raises(ValueError):
        cp.to_seconds(0.008, khz_inv=8)


def test_quefrency_axis_matches_cepstrum():
    c = cp.cepstrum(np.random.default_rng(0).standard_normal(512), FS)
    np.testing.assert_allclose(cp.quefrency_axis(c, "kHz^-1"), c.quefrency * 1e3)
    assert cp.quefrency_axis(c, "period_Hz").shape == c.quefrency.shape


def test_ripple_period_is_physical():
    # A cosine ripple of period 125 Hz in the log spectrum peaks at q = 8 kHz^-1 (= 8 ms).
    n = 4096
    f = np.fft.fftfreq(n, 1 / FS)
    spec = cp.Spectrum(np.exp(0.5 * np.cos(2 * np.pi * f / 125.0)), FS, n, "amplitude", "two")
    c = cp.cepstrum(spec)
    q = cp.quefrency_axis(c, "kHz^-1")
    assert q[np.argmax(c.data[1 : n // 2]) + 1] == pytest.approx(8.0, abs=0.01)


@pytest.mark.parametrize("kw", [{"khz_inv": 3}, {"period_hz": 1000 / 3}, {"cutoff": 0.003}])
def test_lowpass_equivalent_specs(kw):
    w = cp.lifter_weights(cp.lowpass(**kw), 1024, FS)
    ref = cp.lifter_weights(cp.lowpass(0.003), 1024, FS)
    np.testing.assert_array_equal(w, ref)


def test_bandpass_notch_keyword_forms():
    a = cp.lifter_weights(cp.bandpass(0.006, 0.010), 1024, FS)
    b = cp.lifter_weights(cp.bandpass(period_hz=(100, 1000 / 6)), 1024, FS)
    c = cp.lifter_weights(cp.bandpass(khz_inv=(10, 6)), 1024, FS)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)
    n1 = cp.lifter_weights(cp.notch(0.008, 0.002), 1024, FS)
    n2 = cp.lifter_weights(cp.notch(period_hz=125, width_khz_inv=2), 1024, FS)
    np.testing.assert_array_equal(n1, n2)
    with pytest.raises(ValueError):
        cp.bandpass(0.006, period_hz=(100, 200))
