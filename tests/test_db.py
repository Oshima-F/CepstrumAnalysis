import numpy as np

import cepstrum as cp

FS = 16000.0
DB = 20 / np.log(10)
rng = np.random.default_rng(5)


def _x(n=512):
    return rng.standard_normal(n)


def test_default_is_natural_log():
    assert cp.cepstrum(_x(), FS).db is False


def test_real_cepstrum_scales_by_db_per_neper():
    x = _x()
    c, cd = cp.cepstrum(x, FS), cp.cepstrum(x, FS, db=True)
    assert cd.db is True
    np.testing.assert_allclose(cd.data, DB * c.data, rtol=1e-10, atol=1e-12)


def test_fft_of_db_cepstrum_is_the_db_spectrum():
    x = _x()
    cd = cp.cepstrum(x, FS, db=True)
    expected = 20 * np.log10(np.abs(np.fft.fft(x)))
    np.testing.assert_allclose(np.fft.fft(cd.data).real, expected, atol=1e-8)


def test_log_spectrum_is_in_db_either_way():
    x = _x(1024)
    a = cp.analyze_real(x, FS, window="hann", khz_inv=3)
    b = cp.analyze_real(x, FS, window="hann", khz_inv=3, db=True)
    for part in ("full", "low", "high"):
        np.testing.assert_allclose(a.log_spectrum(part), b.log_spectrum(part), atol=1e-8)


def test_low_plus_high_is_full_in_db():
    r = cp.analyze_real(_x(), FS, khz_inv=3, db=True)
    np.testing.assert_allclose(r.low.data + r.high.data, r.full.data, atol=1e-12)


def test_cepstrogram_db():
    x = _x(2000)
    g = cp.cepstrogram(x, FS, frame_len=256, hop=64)
    gd = cp.cepstrogram(x, FS, frame_len=256, hop=64, db=True)
    assert gd.db is True and g.db is False
    np.testing.assert_allclose(gd.data, DB * g.data, rtol=1e-9, atol=1e-11)


def test_cepstrogram_reconstruction_is_the_same_either_way():
    x = _x(2000)
    g = cp.cepstrogram(x, FS, frame_len=256, hop=64)
    gd = cp.cepstrogram(x, FS, frame_len=256, hop=64, db=True)
    np.testing.assert_allclose(cp.reconstruct_cepstrogram(g, cp.lowpass(khz_inv=2)),
                               cp.reconstruct_cepstrogram(gd, cp.lowpass(khz_inv=2)), atol=1e-9)
