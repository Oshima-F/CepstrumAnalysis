import warnings

import numpy as np
import pytest

import cepstrum as cp

FS = 16000.0
rng = np.random.default_rng(11)


def _noise(n=2048):
    return rng.standard_normal(n)


# ---- silence -------------------------------------------------------------------------------

def test_digital_silence_warns():
    with pytest.warns(cp.SilenceWarning, match="silent or constant"):
        cp.cepstrum(np.zeros(256), FS)


def test_constant_signal_warns_and_is_not_reported_as_clipping():
    with pytest.warns(cp.SilenceWarning) as rec:
        cp.cepstrum(np.full(256, 0.5), FS)
    assert not any(issubclass(w.category, cp.ClippingWarning) for w in rec)


def test_silent_spectrum_warns():
    spec = cp.Spectrum(np.zeros(129), FS, 256, "amplitude", "one")
    with pytest.warns(cp.SilenceWarning, match="spectrum"):
        cp.cepstrum(spec)


def test_cepstrogram_warns_once_for_whole_signal_silence():
    with pytest.warns(cp.SilenceWarning) as rec:
        cp.cepstrogram(np.zeros(2000), FS, frame_len=256)
    assert len(rec) == 1


def test_pauses_inside_a_signal_do_not_warn():
    x = np.r_[_noise(1000), np.zeros(3000), _noise(1000)]
    cp.cepstrogram(x, FS, frame_len=256, hop=64)  # a warning would fail the test (filterwarnings=error)


# ---- clipping ------------------------------------------------------------------------------

def test_clipped_signal_warns_with_percentage():
    x = np.clip(_noise() * 2.0, -1.0, 1.0)
    with pytest.warns(cp.ClippingWarning, match=r"clipped: \d+\.\d+%"):
        cp.cepstrum(x, FS)


def test_one_sided_clipping_warns():
    x = _noise()
    assert -x.min() > 0.5 and x.max() > 0.5  # the larger peak is on the other side of the clipped one
    with pytest.warns(cp.ClippingWarning):
        cp.cepstrum(np.minimum(x, 0.5), FS)


def test_zero_padding_and_unipolar_zeros_are_not_clipping():
    h = np.zeros(256)
    h[:3] = [1.0, -0.5, 0.2]
    cp.cepstrum(h, FS)
    cp.cepstrum(np.abs(_noise()) * (np.arange(2048) % 7 == 0), FS)  # many exact zeros at the minimum


def test_clipping_in_one_channel_is_reported_with_the_channel():
    x = np.stack([_noise(), np.clip(_noise() * 3, -1, 1)])
    with pytest.warns(cp.ClippingWarning, match="channel 1"):
        cp.cepstrum(x, FS)


def test_scale_does_not_matter_for_clipping():
    x = np.clip(_noise() * 2.0, -1.0, 1.0) * 1e-6
    with pytest.warns(cp.ClippingWarning):
        cp.cepstrum(x, FS)


@pytest.mark.parametrize("make", [
    lambda: _noise(),                                               # noise
    lambda: np.sin(2 * np.pi * 440 * np.arange(2048) / FS),         # sine at its natural peak
    lambda: np.sin(2 * np.pi * 123.4 * np.arange(2048) / FS) * 0.9,
    lambda: np.eye(1, 2048, 100).ravel() + 0.01 * _noise(),         # isolated pulse
])
def test_clean_signals_do_not_warn(make):
    cp.cepstrum(make(), FS)


def test_clipping_in_cepstrogram_warns():
    x = np.clip(_noise(4000) * 2.0, -1.0, 1.0)
    with pytest.warns(cp.ClippingWarning):
        cp.cepstrogram(x, FS, frame_len=256)


def test_warnings_can_be_silenced_or_escalated():
    x = np.clip(_noise() * 2.0, -1.0, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cp.CepstrumWarning)
        cp.cepstrum(x, FS)
    with warnings.catch_warnings():
        warnings.simplefilter("error", cp.ClippingWarning)
        with pytest.raises(cp.ClippingWarning):
            cp.cepstrum(x, FS)


# ---- invalid input (errors) ----------------------------------------------------------------

@pytest.mark.parametrize("bad, msg", [
    (np.array([1.0, np.nan, 2.0]), "NaN or inf"),
    (np.array([1.0, np.inf, 2.0]), "NaN or inf"),
    (np.ones(8) + 1j, "real"),
    (np.array([]), "empty"),
    (np.ones((2, 2, 2)), "1-D"),
    (np.array(["a", "b"]), "numeric"),
])
def test_invalid_signals_raise(bad, msg):
    with pytest.raises(ValueError, match=msg):
        cp.cepstrum(bad, FS)


@pytest.mark.parametrize("fs", [0.0, -1.0, np.nan, np.inf])
def test_invalid_fs_raises(fs):
    with pytest.raises(ValueError, match="fs"):
        cp.cepstrum(_noise(), fs)
    with pytest.raises(ValueError, match="fs"):
        cp.cepstrogram(_noise(), fs, frame_len=256)


@pytest.mark.parametrize("kw", [{"hop": 0}, {"hop": 300}, {"frame_len": 0}, {"n_fft": 100}])
def test_invalid_cepstrogram_parameters_raise(kw):
    args = {"frame_len": 256, **kw}
    with pytest.raises(ValueError):
        cp.cepstrogram(_noise(), FS, **args)


def test_invalid_spectrum_raises():
    with pytest.raises(ValueError, match="NaN"):
        cp.Spectrum(np.r_[np.ones(128), np.nan], FS, 256, "amplitude", "one")
    with pytest.raises(ValueError, match="fs"):
        cp.Spectrum(np.ones(256), -1.0, 256)
    with pytest.raises(ValueError, match="n_fft"):
        cp.Spectrum(np.ones(0), FS, 0)


@pytest.mark.parametrize("make", [
    lambda: cp.lowpass(-0.001),
    lambda: cp.highpass(khz_inv=-1),
    lambda: cp.lowpass(period_hz=0),
    lambda: cp.lowpass(period_hz=-300),
    lambda: cp.lowpass(np.nan),
    lambda: cp.notch(0.008, -0.001),
])
def test_invalid_lifters_raise(make):
    with pytest.raises(ValueError):
        make()


# ---- reconstruction coverage ---------------------------------------------------------------

def test_hann_without_overlap_warns_on_reconstruction():
    cg = cp.cepstrogram(_noise(1024), FS, frame_len=256, hop=256)  # periodic Hann has w[0] = 0
    with pytest.warns(cp.ReconstructionWarning, match=r"4 of 1024 samples"):
        cp.reconstruct_cepstrogram(cg)


def test_decomposition_reconstruct_warns_too():
    r = cp.analyze_cepstrogram(_noise(1024), FS, frame_len=256, hop=256, khz_inv=1)
    with pytest.warns(cp.ReconstructionWarning):
        r.reconstruct("low")


@pytest.mark.parametrize("kw", [
    {"hop": 128},                                   # Hann with overlap
    {"hop": 256, "window": "hamming"},              # w > 0 everywhere
    {"hop": 256, "window": "rect"},
])
def test_covering_settings_do_not_warn(kw):
    x = _noise(1024)
    cg = cp.cepstrogram(x, FS, frame_len=256, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("error", cp.ReconstructionWarning)
        np.testing.assert_allclose(cp.reconstruct_cepstrogram(cg), x, atol=1e-9)
