# cepstrum

日本語 | [English](README.en.md)

実ケプストラムを計算するPythonライブラリ．

音声信号をスペクトル包絡を表す低ケフレンシ成分と，微細構造を表す高ケフレンシ成分に分離する．

- 数学的な定義：[docs/definitions.md](docs/definitions.md)
- 仕様と API：[docs/requirements.md](docs/requirements.md)

## インストール

```bash
pip install git+https://github.com/Oshima-F/CepstrumAnalysis
```

依存は NumPy のみ．

## 使い方

```python
import cepstrum as cp

# 実ケプストラムを周期 300 Hz で低・高ケフレンシに分ける
r = cp.analyze_real(x, fs, window="hann", period_hz=300)
r.low, r.high                # low + high == full
r.log_spectrum("low")        # スペクトル包絡 [dB]

# ケプストグラムと時間信号への再構成
g = cp.analyze_cepstrogram(x, fs, frame_len=1024, hop=256, period_hz=300)
g.reconstruct("low")
g.reconstruct("high")
```

## ライセンス

MIT
