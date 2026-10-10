# cepstrum

ケプストラム分析ライブラリ（Python / NumPy のみ）。他のアプリケーションに組み込める部品として、実・複素・パワーケプストラム、
短時間ケプストラム（ケプストグラム）、リフタリング、時間信号の再構成を提供します。

- 入力は時間信号、またはスペクトル（複素・振幅・パワー、片側・両側）
- 全区間の平均的なケプストラムと、フレームごとのケプストグラムを出力
- 任意のリフタをかけて、時間信号を再構成
- ケプストラムを低ケフレンシ（スペクトル包絡）と高ケフレンシ（ピッチ・微細構造）に分けて返す
- クエフレンシーの単位は kHz⁻¹（= 1 kHz あたりのスペクトルの山の数）。周期 [Hz] でも指定できる

詳細な仕様は [docs/requirements.md](docs/requirements.md) にあります。

## インストール

```bash
pip install -e .            # ライブラリ本体（依存: numpy）
pip install -e ".[test]"    # + pytest
pip install -e ".[plot]"    # + matplotlib（examples 用）
```

## クイックスタート

```python
import numpy as np
import cepstrum as cp

fs = 16000
x = ...  # 時間信号 [N]、または多チャネル [ch, N]（チャネル平均でモノラル化）

# 実ケプストラム。周期 300 Hz を境に、包絡（low）と微細構造（high）へ分割
r = cp.analyze_real(x, fs, window="hann", period_hz=300)
r.full, r.low, r.high        # Cepstrum オブジェクト（low + high == full）
r.log_spectrum("low")        # 包絡の対数スペクトル [dB]
r.freqs                      # その周波数軸 [Hz]
r.reconstruct("full")        # 時間信号へ戻す

# ケプストグラム [quefrency, frame]
g = cp.analyze_cepstrogram(x, fs, frame_len=1024, hop=256, kind="real", khz_inv=1)
g.low.data, g.high.data
g.log_spectrum("low")        # 包絡のスペクトログラム [freq, frame] (dB)
g.full.times                 # フレーム中心時刻 [s]

# スペクトルから入力する場合
spec = cp.Spectrum(np.abs(np.fft.rfft(x)), fs, len(x), kind="amplitude", sided="one")
c = cp.analyze_real(spec, khz_inv=2)
```

## API

### 高水準（分析ごとに 1 ファイル・1 関数）

| 関数 | ファイル | 備考 |
|---|---|---|
| `analyze_real` | `real_cepstrum.py` | IFFT(log\|X\|) |
| `analyze_complex` | `complex_cepstrum.py` | 位相が必要。n_fft は偶数。`low` と `high` は畳み込むと元信号になる成分 |
| `analyze_power` | `power_cepstrum.py` | 不可逆（`log_spectrum` / `reconstruct` は `ValueError`） |
| `analyze_cepstrogram` | `cepstrogram_analysis.py` | `kind` = real / complex / power、全フレームに同じリフタ |

共通の引数：`x`（時間信号または `Spectrum`）、`fs`、`n_fft`、`window`、`eps`、そして分割位置の指定
（`lifter=` / `cutoff=`[s] / `khz_inv=` / `period_hz=` のうち 1 つ。未指定なら分割なし）。
戻り値は `Decomposition`（`full` / `low` / `high` / `log_spectrum(part)` / `freqs` / `reconstruct(part)`）。

### 低水準

| 名前 | 役割 |
|---|---|
| `cepstrum(x, fs, kind=...)` | 全区間 1 回の FFT による平均ケプストラム → `Cepstrum` |
| `cepstrogram(x, fs, frame_len=...)` | 短時間ケプストラム → `Cepstrogram` |
| `Spectrum(data, fs, n_fft, kind, sided)` | スペクトル入力（複素・振幅・パワー、片側・両側） |
| `lowpass` / `highpass` / `bandpass` / `notch` / `apply_lifter` | リフタ。秒、`khz_inv=`、`period_hz=` で指定 |
| `reconstruct` / `reconstruct_cepstrogram` | リフタ後のケプストラムから時間信号へ |
| `quefrency_axis(c, unit)` / `unit_label` | 軸の単位変換 |

各モジュール冒頭の docstring に、入力・出力・パラメータを記載しています。

## クエフレンシーの単位

クエフレンシー q は [s] = [Hz⁻¹]。係数は対数スペクトル上の周期 1/q Hz の波（1 Hz あたり q 個）を表します。
SI 接頭辞により 1 kHz⁻¹ = 1 ms で、数値は ms と同じです。

| q | 山/kHz | 波の周期 |
|---|---|---|
| 1 kHz⁻¹（1 ms） | 1 | 1000 Hz |
| 8 kHz⁻¹（8 ms） | 8 | 125 Hz（f0=125 Hz の倍音間隔） |

内部データは秒で保持します。表示用は `quefrency_axis(c, "kHz^-1")`（unit: `"s"` / `"Hz^-1"` / `"ms"` / `"kHz^-1"` / `"period_Hz"`）。

## 注意事項

- 振幅・パワースペクトルには位相がないため、複素ケプストラムは計算できません（`ValueError`）。
- 実ケプストラムからの再構成は、位相を最小位相で補います（現状 `phase="minimum"` のみ）。
- 複素ケプストラムは FFT 長が偶数であることが必要です。
- リフタ後のケプストラムを再構成した信号は、元信号と一致しません（包絡や微細構造だけを残した信号になります）。

## サンプル（`examples/`）

`mao.wav`（AivisSpeech で合成した音声）を対象にしています。直接実行すると PNG を `examples/output/` に保存して開きます。
`run()` を他のファイルから呼ぶと、保存のみです。パラメータは各ファイル冒頭にまとめています。

| スクリプト | 内容 |
|---|---|
| `cepstrum_kinds.py` | 実・複素・パワーケプストラムと f0 の検出 |
| `liftering.py` | 低・高ケフレンシへの分割、各種リフタ、包絡の抽出 |
| `reconstruction.py` | 往復の誤差、エコー除去、最小位相による再構成、ケプストグラムの重畳加算 |
| `speech_cepstrogram.py` | 音声のケプストグラム、f0 軌跡、スペクトル包絡 |
| `synth_aivis.py` | ローカルの AivisSpeech エンジンで音声を合成（`data/mao.wav` を作る） |

```bash
cd examples
python synth_aivis.py "えーっと、今日はですね、ケプストラム分析について、ちょっと試してみようと思います。" --speaker 888753760 --out data/mao.wav
python speech_cepstrogram.py
```

`synth_aivis.py` は AivisSpeech エンジン（既定 `http://127.0.0.1:10101`）の起動が必要です。
`SEG_START` など切り出し区間は `mao.wav` に合わせた値なので、別の音声では調整してください。

## テスト

```bash
pytest
```

複素ケプストラムの厳密な往復、スペクトル入力の一致、低・高の分割（`low + high == full`）、リフタ指定の等価性、
単位変換、ピッチ追従などを確認します。

## ディレクトリ構成

```
src/cepstrum/    ライブラリ本体
tests/           pytest
examples/        サンプルスクリプト（出力は examples/output、音声は examples/data）
docs/            要件書
```
