# Cepstrum Analysis Library: Requirements (v1)

[日本語](requirements.md) | English

## Overview

The library separates speech into a low-quefrency part, which represents the spectral envelope, and a high-quefrency part, which represents the fine structure.
The fine structure is, for example, the harmonic peaks and valleys caused by the pitch.

It is a DSP library meant to be embedded in other applications. The only dependency is NumPy.
It consists of pure functions and immutable data classes, so results depend only on the arguments.
The mathematical definitions are in [definitions.en.md](definitions.en.md).

Features:

- Real cepstrum (computed from the magnitude spectrum)
- Cepstrogram (short-time cepstrum)
- Spectrum input (complex, amplitude or power; one-sided or two-sided)
- Liftering and separation into low / high quefrency parts
- Reconstruction of time signals from a cepstrogram
- Quefrency unit conversion (seconds, kHz⁻¹, period [Hz])
- Input validation

## 1. Input

| Input | Specification |
|---|---|
| Time signal | `[N]` or `[ch, N]`. Multichannel signals are averaged over channels to mono |
| Spectrum | `Spectrum(data, fs, n_fft, kind, sided)` |

- `kind` is `complex`, `amplitude` or `power`. `power` is converted to amplitude by a square root. Values are linear.
- `sided` is `one` (`n_fft//2+1` points) or `two` (`n_fft` points). Internally the spectrum is converted to a two-sided complex spectrum.
- The cepstrum is computed from the magnitude. The phase of a complex spectrum is not used, so `kind` does not change the result.
- Amplitude and power `data` is used as |X| as given. No correction, such as doubling a one-sided spectrum, is applied.

## 2. Input validation

Invalid input raises `ValueError`:

- Time signal: NaN or inf, complex values, empty, shape other than `[N]` or `[ch, N]`
- Spectrum: NaN or inf
- Arguments: `fs` not a finite positive number, `frame_len < 1`, `hop` outside `1..frame_len`, a negative or non-finite lifter cutoff, `period_hz <= 0`

The following cases emit a warning. All are `CepstrumWarning` (a subclass of `UserWarning`) and can be suppressed or turned into errors with `warnings`.

- `SilenceWarning`: the signal is all zeros or constant, or the spectrum is all zeros
- `ClippingWarning`: the signal stays at its own maximum or minimum for 3 or more consecutive samples
- `ReconstructionWarning`: in reconstruction, some samples have a zero sum of squared windows (e.g. the periodic Hann window with `hop` equal to `frame_len`). Those samples cannot be recovered and come out as about 0

Clipping is judged against the signal's own maximum and minimum, so it does not depend on the amplitude scale. Multichannel signals are checked per channel.
Sticking at zero is not checked, because it cannot be distinguished from zero padding.

For a cepstrogram the checks run once on the whole signal. Silent frames inside the signal are treated as normal.

## 3. Cepstrum

- The real cepstrum is IFFT(ln|X|). It is real and even (`c[n] == c[-n]`).
- `cepstrum()` returns the cepstrum of the whole signal, computed with a single FFT.
- `cepstrogram()` returns the cepstrum of each frame as `[quefrency, frame]`.
  The frame length, shift, window (hann, hamming, rect or an array) and FFT length can be set.
  It has a time axis `times` and a quefrency axis `quefrency` (seconds).
- Before taking the log, the magnitude is floored at "maximum of the magnitude spectrum × `eps`".

With `db=True`, the log spectrum before the inverse FFT is 20·log10|X| instead of ln|X| (default `False`).
The real cepstrum then becomes 20/ln(10) times larger, and its FFT is the spectrum in dB.
The setting is stored in `Cepstrum.db` and `Cepstrogram.db`; `log_spectrum()` and reconstruction detect the unit automatically.
`log_spectrum()` always returns dB. Lifter weights do not depend on `db`.

## 4. Quefrency units

The unit of quefrency q is [s] = [Hz⁻¹].
The coefficient at quefrency q is the size of the cos(2π·q·f) component in the log spectrum, a ripple with period 1/q [Hz] along the frequency axis.

| q | Peaks per kHz | Ripple period |
|---|---|---|
| 1 ms | 1 | 1000 Hz (formant spacing) |
| 8 ms | 8 | 125 Hz (harmonic spacing for f0 = 125 Hz) |

- Values are stored in seconds.
- The main display unit is kHz⁻¹. Since 1 kHz⁻¹ = 1 ms, the numbers are the same as in ms.
- The period [Hz] is shown as a secondary axis.
- The coefficient at q = 0 is the overall level; its period is infinite. The largest quefrency is ⌊`n_fft`/2⌋ / fs (period: twice the frequency resolution).
- Conversion uses `quefrency_axis(c, unit)`, `from_seconds` and `to_seconds`.
  `unit` is `"s"` (= `"Hz^-1"`), `"ms"` (= `"kHz^-1"`) or `"period_Hz"`.

## 5. Lifters

- A lifter is a function from signed quefrency [s] to a weight, or an array of length `n_fft`.
- `lowpass`, `highpass`, `bandpass` and `notch` are provided.
- Boundaries are given in exactly one form: seconds (positional), `khz_inv=` or `period_hz=`.
  Example: `lowpass(period_hz=300)` keeps ripples with periods longer than 300 Hz.
  Other examples: `bandpass(period_hz=(167, 100))`, `notch(period_hz=125, width_khz_inv=2)`.
- `apply_lifter` works on both `Cepstrum` and `Cepstrogram`.

## 6. Reconstruction

Time signals are reconstructed only from a cepstrogram (`reconstruct_cepstrogram`, `Decomposition.reconstruct`).
The steps are:

1. Compute the complex spectrogram X from the time signal and store it in the `Cepstrogram`.
2. FFT the liftered cepstrogram and take exp to get the magnitude |X^ℓ|.
3. Compute the gain G = |X^ℓ| / |X|.
4. Inverse-FFT X · G, cut to `frame_len`, apply a synthesis window equal to the analysis window and overlap-add. Normalize by the sum of squared windows.

- The phase is kept; the lifter changes only the magnitude.
- Without a lifter, G = 1 and the original signal is returned.
- The denominator |X| of the gain is floored at "frame maximum × `eps`", the same floor as the log. Therefore, without a lifter, the original signal is returned for any `eps`. Silent frames stay 0.
- Only a cepstrogram made with `store_spectrogram=True` (default) can be reconstructed.
- If the window and shift do not meet the perfect-reconstruction condition of the definitions, `ReconstructionWarning` is emitted.
- When the signal reconstructed from `low` is split into frames and analyzed again, its magnitude spectra move toward the envelope, but do not match it.

The formulas are in [definitions.en.md](definitions.en.md).

## 7. API

### High level

One file and one function per analysis. The input is a time signal or a `Spectrum`; the result is a `Decomposition`.

| File | Function | Description |
|---|---|---|
| `real_cepstrum.py` | `analyze_real` | Real cepstrum |
| `cepstrogram_analysis.py` | `analyze_cepstrogram` | Cepstrogram. The same lifter is applied to every frame |

- The split is given by `lifter=` (function or array), or by one lowpass cutoff: `cutoff` (seconds), `khz_inv` or `period_hz`.
  If none is given, nothing is split: `low` equals `full` and `high` is 0.
- `Decomposition` has `full`, `low` and `high`. `low` is the liftered part (spectral envelope) and includes the q = 0 coefficient.
  `high` is `full - low` (fine structure), and `low + high == full` holds exactly.
- `log_spectrum(part)` returns the one-sided log-magnitude spectrum [dB] of that part. The frequency axis is `freqs`.
- `reconstruct(part)` returns the time signal of that part (results of `analyze_cepstrogram` only).
  `"full"` is the original signal, `"low"` has the envelope magnitude and the original phase, and `"high"` is `full - low` in the time domain.

### Low level

| Name | Description |
|---|---|
| `cepstrum(x, fs)` | Cepstrum of the whole signal (returns `Cepstrum`) |
| `cepstrogram(x, fs, frame_len=...)` | Cepstrogram (returns `Cepstrogram`) |
| `Spectrum(data, fs, n_fft, kind, sided)` | Spectrum input |
| `lowpass` / `highpass` / `bandpass` / `notch` / `apply_lifter` | Lifters |
| `reconstruct_cepstrogram(cg, lifter)` | Reconstruct a time signal from a liftered cepstrogram |
| `quefrency_axis(c, unit)` / `from_seconds` / `to_seconds` | Quefrency unit conversion |

Arguments and defaults are documented in the docstrings. The public API is collected in `cepstrum/__init__.py`.

## 8. Mapping to the definitions

definitions.en.md is formulated on the one-sided spectrum (I points). The implementation computes on two-sided arrays (M points), so the mapping is summarized here.

| Quantity in the definitions | Implementation |
|---|---|
| Spectrum | `Spectrum`. `sided="one"` is one-sided (I points), `sided="two"` is two-sided (M points) |
| Window | `window`: `"hann"`, `"hamming"`, `"rect"` (`None`) or an array of length M |
| Floor of the log and the gain ε | `eps` (default 1e-12) |
| dB scale | `db=True` |
| Lifters | `lowpass(c)`, `highpass(c)`, `bandpass(a, b)`, `notch(q_0, W)` |
| Low / high parts | `Decomposition.low`, `Decomposition.high` |
| Keeping the spectrogram | `store_spectrogram=True` |

Implementation notes:

- Mirror: `Spectrum.to_two_sided()` adds the conjugate-symmetric mirror to a one-sided spectrum to make M points. Two-sided data is used as given.
- Array length: `Cepstrum.data` is an even array of M points; its first L points are $\bm{z}$ in the definitions.
  `Cepstrogram.data` also has M rows; its first L rows are $\bm{Z}$. Lifter weights are applied to all M points as a function of $\lvert q\rvert$.
- Reconstruction: the stored one-sided spectrogram is multiplied by the gain, and each frame is converted back to a real signal with `irfft`.
- Framing: the frame start jH − (M − H) of the definitions is realized by prepending M − H zeros to the signal. The number of frames equals J in the definitions.
- FFT length: the default `n_fft` is `frame_len` (the signal length for `cepstrum()`), which is M in the definitions.
  If `n_fft` is larger than M, zeros are appended after the M windowed points before the DFT. The window length stays M,
  and the number of frequency and quefrency bins becomes ⌊`n_fft`/2⌋ + 1.

## 9. Non-functional requirements

- The only dependency is NumPy. Tests use pytest.
- The code has type hints.

## 10. Open issues

- Processing channels individually
- Tapered lifters
- Spectrum input in dB
- An upper limit on the reconstruction gain (to avoid over-raising spectral valleys)
