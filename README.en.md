# cepstrum

[日本語](README.md) | English

A Python library for the real cepstrum.

It separates a speech signal into a low-quefrency part, which represents the spectral envelope, and a high-quefrency part, which represents the fine structure.

- Mathematical definitions: [docs/definitions.en.md](docs/definitions.en.md)
- Specification and API: [docs/requirements.en.md](docs/requirements.en.md)

## Installation

```bash
pip install git+https://github.com/Oshima-F/CepstrumAnalysis
```

The only dependency is NumPy.

## Usage

```python
import cepstrum as cp

# Split the real cepstrum into low / high quefrency parts at a ripple period of 300 Hz
r = cp.analyze_real(x, fs, window="hann", period_hz=300)
r.low, r.high                # low + high == full
r.log_spectrum("low")        # spectral envelope [dB]

# Cepstrogram and reconstruction to time signals
g = cp.analyze_cepstrogram(x, fs, frame_len=1024, hop=256, period_hz=300)
g.reconstruct("low")
g.reconstruct("high")
```

## License

MIT
