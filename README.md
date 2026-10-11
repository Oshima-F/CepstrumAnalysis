# cepstrum

**実ケプストラム**のライブラリ（Python / NumPy のみ）です．**音声を，スペクトル包絡を表す低ケフレンシ成分と，
微細構造（ピッチによる倍音の山谷など）を表す高ケフレンシ成分に分離すること**を目的とします．
そのために，実ケプストラム・パワーケプストラム，短時間ケプストラム（ケプストグラム），リフタリング，低・高の分離，時間信号の再構成を提供します．

- 入力は時間信号，またはスペクトル（複素・振幅・パワー，片側・両側．ケプストラムは振幅から計算する）
- 全区間の平均的なケプストラムと，フレームごとのケプストグラムを出力
- ケプストラムを低ケフレンシ（スペクトル包絡）と高ケフレンシ（ピッチ・微細構造）に分けて返す
- ケプストグラムから時間信号を再構成：元の複素スペクトログラムに，リフタ後の振幅との比のゲインを掛けて重畳加算する
  （位相は元のもの，リフタで変わるのは振幅）
- ケフレンシの単位は kHz⁻¹（= 1 kHz あたりのスペクトルの山の数）．周期 [Hz] でも指定できる

`examples/` に，音声を低・高ケフレンシ成分に分ける様子を示すサンプルがあります．

- 数学的な定義（ケプストラム，リフタ，低・高分離，再構成，単位）：[docs/definitions.md](docs/definitions.md)
- 仕様：[docs/requirements.md](docs/requirements.md)

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
x = ...  # 時間信号 [N]，または多チャネル [ch, N]（チャネル平均でモノラル化）

# 実ケプストラム．周期 300 Hz を境に，包絡（low）と微細構造（high）へ分割
r = cp.analyze_real(x, fs, window="hann", period_hz=300)
r.full, r.low, r.high        # Cepstrum オブジェクト（low + high == full）
r.log_spectrum("low")        # 包絡の対数スペクトル [dB]
r.freqs                      # その周波数軸 [Hz]

# ケプストラムを dB 基準で返す（既定は自然対数基準）
rd = cp.analyze_real(x, fs, window="hann", period_hz=300, db=True)
rd.full.data                 # 自然対数基準の約 8.686 倍．FFT すると dB のスペクトルになる

# ケプストグラム [quefrency, frame]
g = cp.analyze_cepstrogram(x, fs, frame_len=1024, hop=256, khz_inv=1)
g.low.data, g.high.data
g.log_spectrum("low")        # 包絡のスペクトログラム [freq, frame] (dB)
g.full.times                 # フレーム中心時刻 [s]

# 時間信号への再構成（ケプストグラム）
g.reconstruct("full")        # 元の信号（リフタなし: ゲイン = 1）
g.reconstruct("low")         # 包絡の振幅 + 元の位相
g.reconstruct("high")        # 元の信号 − low（low + high == full）

# スペクトルから入力する場合
spec = cp.Spectrum(np.abs(np.fft.rfft(x)), fs, len(x), kind="amplitude", sided="one")
c = cp.analyze_real(spec, khz_inv=2)
```

## API

### 高水準（分析ごとに 1 ファイル・1 関数）

| 関数 | ファイル | 備考 |
|---|---|---|
| `analyze_real` | `real_cepstrum.py` | 実ケプストラム IFFT(ln\|X\|) と低・高の分離 |
| `analyze_power` | `power_cepstrum.py` | パワーケプストラム \|IFFT(ln\|X\|²)\|² と低・高の分離 |
| `analyze_cepstrogram` | `cepstrogram_analysis.py` | `kind` = real / power，全フレームに同じリフタ．時間信号への再構成（`reconstruct`）を持つ |

共通の引数：`x`（時間信号または `Spectrum`），`fs`，`n_fft`，`window`，`eps`，`db`（dB 基準で返すか．既定 `False`），そして分割位置の指定
（`lifter=` / `cutoff=`[s] / `khz_inv=` / `period_hz=` のうち 1 つ．未指定なら分割なし）．
戻り値は `Decomposition`（`full` / `low` / `high` / `log_spectrum(part)` / `freqs`，`analyze_cepstrogram` の結果は `reconstruct(part)` も持つ）．

### 低水準

| 名前 | 役割 |
|---|---|
| `cepstrum(x, fs, kind=...)` | 全区間 1 回の FFT による平均ケプストラム → `Cepstrum` |
| `cepstrogram(x, fs, frame_len=...)` | 短時間ケプストラム → `Cepstrogram` |
| `Spectrum(data, fs, n_fft, kind, sided)` | スペクトル入力（複素・振幅・パワー，片側・両側） |
| `lowpass` / `highpass` / `bandpass` / `notch` / `apply_lifter` | リフタ．秒，`khz_inv=`，`period_hz=` で指定 |
| `reconstruct_cepstrogram(cg, lifter)` | リフタ後のケプストグラムから時間信号へ（元の位相を使う） |
| `quefrency_axis(c, unit)` / `from_seconds` / `to_seconds` | 軸の単位変換 |

各公開関数・クラスの docstring に，Google スタイルの `Args` / `Returns` / `Raises` を記載しています（各引数の既定値も明記）．

## ケフレンシの単位

ケフレンシ q は [s] = [Hz⁻¹]．係数は対数スペクトル上の周期 1/q Hz の波（1 Hz あたり q 個）を表します．
SI 接頭辞により 1 kHz⁻¹ = 1 ms で，数値は ms と同じです．

| q | 山/kHz | 波の周期 |
|---|---|---|
| 1 kHz⁻¹（1 ms） | 1 | 1000 Hz |
| 8 kHz⁻¹（8 ms） | 8 | 125 Hz（f0=125 Hz の倍音間隔） |

内部データは秒で保持します．表示用は `quefrency_axis(c, "kHz^-1")`（unit: `"s"` / `"Hz^-1"` / `"ms"` / `"kHz^-1"` / `"period_Hz"`）．

## 仕様の補足

- **入力**：ケプストラムは振幅スペクトルから計算します．複素スペクトルを入力すると，その振幅を使います．
- **再構成の方式**：各フレームの元の複素スペクトル X に，ゲイン `G = |X^ℓ| / |X|`（= √(|X^ℓ|² / |X|²)，|X|² は元のパワースペクトログラム，
  |X^ℓ|² はリフタ後のケプストグラムが表すパワー）を掛けて逆 FFT し，合成窓つきで重畳加算します．位相は元のまま，リフタで変わるのは振幅です．
  リフタなしなら G = 1 で，元の信号に戻ります．
  - 逆 FFT の出力は `frame_len` で切り出します．ゲインが長い応答を持つときは `n_fft` を長く取ってください（低域のリフタのように滑らかなゲインなら
    `frame_len` で十分です）．
  - `low` 信号のフレームごとの振幅は，包絡へ近づきます（`mao.wav` の例で，包絡との距離が元の約 0.6 倍）．
  - `low` と `high` の和が元になるのは，ケプストラム（対数スペクトル）の領域（`low + high == full`）と，時間領域（`high = 元の信号 − low`）です．
  - 再構成には，`kind="real"` で作り，複素スペクトログラムを保持した（`store_spectrogram=True`，既定）ケプストグラムを使います．
- **入力の検査**：不正な入力（NaN・inf，複素の時間信号，空の信号，不正な `fs`，`hop` など）は `ValueError` になります．
  **無音（全ゼロ・定数）は `SilenceWarning`，クリッピング（信号自身の最大値または最小値に 3 サンプル以上連続して張り付く）は
  `ClippingWarning`** で知らせます（どちらも `UserWarning` の派生で，`warnings.simplefilter("ignore", cp.CepstrumWarning)` で抑制，
  `"error"` で例外化できます）．ケプストグラムの検査は信号全体に対して 1 回で，途中の無音区間（ポーズ）は正常として通ります．
- `db=True` は，逆 FFT の前の対数スペクトルを自然対数 ln|X| から 20·log10|X| [dB] にします．実ケプストラムは
  20/ln(10) ≈ 8.686 倍になり，パワーケプストラムはその二乗の大きさになります．`Decomposition.log_spectrum()` はどちらでも dB で返し，
  再構成も単位を自動で判別します（結果は同じ）．

## サンプル（`examples/`）

`mao.wav`（AivisSpeech で合成した音声）を対象にしています．直接実行すると PNG を `examples/output/` に保存して開きます．
`run()` を他のファイルから呼ぶと，PNG を保存します．パラメータは各ファイル冒頭にまとめています．

| スクリプト | 内容 |
|---|---|
| `cepstrum_kinds.py` | 実・パワーケプストラム（ピッチによる高ケフレンシ側のピークの位置を表示） |
| `liftering.py` | 低・高ケフレンシへの分離，各種リフタ，スペクトル包絡の抽出 |
| `reconstruction.py` | リフタなしの往復（元に戻る），低・高ケフレンシ成分の時間信号への分解，低成分の性質（フレームのスペクトルが包絡へ近づく），ファイル全体の再構成 |
| `speech_cepstrogram.py` | 音声のケプストグラム，低成分（包絡）と高成分（ピッチ）のスペクトログラム |
| `synth_aivis.py` | ローカルの AivisSpeech エンジンで音声を合成（`data/mao.wav` を作る） |

```bash
cd examples
python synth_aivis.py "えーっと、今日はですね、ケプストラム分析について、ちょっと試してみようと思います。" --speaker 888753760 --out data/mao.wav
python speech_cepstrogram.py
```

`synth_aivis.py` は AivisSpeech エンジン（既定 `http://127.0.0.1:10101`）を起動しておくと使えます．
`SEG_START` など切り出し区間は `mao.wav` に合わせた値なので，別の音声では調整してください．

## テスト

```bash
pytest
```

スペクトル入力の一致，振幅だけで決まること（位相に依存しない），リフタなしの再構成が元の信号に戻ること（窓・シフト・奇数長・ゼロパディング），
再構成のゲインが振幅比で位相が保たれること，低・高の分割（`low + high == full`），低成分のフレームのスペクトルが包絡へ近づくこと，
入力の検査（警告・エラー），`db` フラグ，リフタ指定の等価性，単位変換，ピッチ（高ケフレンシ側のピーク）の追従などを確認します．

## ディレクトリ構成

```
src/cepstrum/    ライブラリ本体
tests/           pytest
examples/        サンプルスクリプト（出力は examples/output，音声は examples/data）
docs/            定義（definitions.md），要件書（requirements.md）
```
