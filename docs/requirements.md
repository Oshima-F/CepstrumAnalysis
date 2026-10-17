# ケプストラム分析ライブラリ 要件 (v1)

日本語 | [English](requirements.en.md)

## 概要

音声を，スペクトル包絡を表す低ケフレンシ成分と，微細構造を表す高ケフレンシ成分に分離する．
微細構造は，ピッチによる倍音の山谷などである．

他のアプリケーションに組み込める DSP ライブラリとする．依存は NumPy のみ．
純関数と不変データクラスで構成し，結果は引数だけで決まる．
数学的な定義は [definitions.md](definitions.md) にまとめる．

提供する機能は次のとおり．

- 実ケプストラムの計算（振幅スペクトルから計算する）
- ケプストグラム（短時間ケプストラム）の計算
- スペクトル入力（複素・振幅・パワー，片側・両側）
- リフタリングと，低・高ケフレンシ成分への分離
- ケプストグラムからの時間信号の再構成
- ケフレンシの単位変換（秒，kHz⁻¹，周期 [Hz]）
- 入力の検査

## 1. 入力

| 入力 | 仕様 |
|---|---|
| 時間信号 | `[N]` または `[ch, N]`．多チャネルはチャネル平均でモノラル化する |
| スペクトル | `Spectrum(data, fs, n_fft, kind, sided)` |

- `kind` は `complex`，`amplitude`，`power` のいずれか．`power` は平方根を取って振幅にする．値は線形で与える．
- `sided` は `one`（`n_fft//2+1` 点）か `two`（`n_fft` 点）．内部では両側の複素スペクトルに揃える．
- ケプストラムは振幅から計算する．複素スペクトルを与えても位相は使わないので，`kind` によって結果は変わらない．
- 振幅・パワーの `data` は |X| としてそのまま使う．片側を 2 倍するなどの補正はしない．

## 2. 入力の検査

不正な入力は `ValueError` とする．

- 時間信号：NaN・inf，複素数，空，`[N]` と `[ch, N]` 以外の形状
- スペクトル：NaN・inf
- 引数：`fs` が有限の正数でない，`frame_len < 1`，`hop` が `1..frame_len` の範囲外，リフタのカットオフが負または非有限，`period_hz <= 0`

次の場合は警告を出す．いずれも `CepstrumWarning`（`UserWarning` の派生）で，`warnings` で抑制や例外化ができる．

- `SilenceWarning`：信号が全ゼロまたは定数，スペクトルが全ゼロ
- `ClippingWarning`：信号自身の最大値または最小値に，3 サンプル以上連続して張り付く
- `ReconstructionWarning`：再構成で，窓の 2 乗の和が 0 になるサンプルがある（周期ハン窓で `hop` が `frame_len` に等しい場合など）．そのサンプルは復元できず，ほぼ 0 になる

クリッピングの判定は信号自身の最大値・最小値を基準にするので，振幅のスケールによらない．多チャネルではチャネルごとに判定する．
ゼロへの張り付きはゼロ埋めと区別できないので，判定しない．

ケプストグラムでは，検査を信号全体に対して 1 回だけ行う．途中の無音フレームは正常として扱う．

## 3. ケプストラム

- 実ケプストラムは IFFT(ln|X|) で，実数かつ偶対称（`c[n] == c[-n]`）．
- `cepstrum()` は，信号全体を 1 回の FFT で処理したケプストラムを返す．
- `cepstrogram()` は，フレームごとのケプストラムを `[ケフレンシ, フレーム]` で返す．
  フレーム長，シフト，窓（hann，hamming，rect，配列），FFT 長を指定できる．
  時刻軸 `times` とケフレンシ軸 `quefrency`（秒）をもつ．
- 対数を取る前に，振幅を「振幅スペクトルの最大値 × `eps`」で下から抑える．

`db=True` にすると，逆 FFT の前の対数スペクトルを ln|X| から 20·log10|X| に替える（既定は `False`）．
実ケプストラムは 20/ln(10) 倍になり，FFT すると dB のスペクトルになる．
この設定は `Cepstrum.db` と `Cepstrogram.db` に保持し，`log_spectrum()` と再構成は単位を自動で判別する．
`log_spectrum()` は常に dB で返す．リフタの重みは `db` の設定によらない．

## 4. ケフレンシの単位

ケフレンシ q の単位は [s] = [Hz⁻¹] である．
ケフレンシ q の係数は，対数スペクトルに含まれる cos(2π·q·f) の成分の大きさで，周波数軸上で周期 1/q [Hz] の波を表す．

| q | 1 kHz あたりの山の数 | 波の周期 |
|---|---|---|
| 1 ms | 1 | 1000 Hz（フォルマント間隔） |
| 8 ms | 8 | 125 Hz（f0 = 125 Hz の倍音間隔） |

- 内部では秒で保持する．
- 表示の主単位は kHz⁻¹ とする．1 kHz⁻¹ = 1 ms なので，数値は ms と同じになる．
- 周期 [Hz] は補助の軸として併記する．
- q = 0 の係数は全体のレベルを表し，周期は無限大．最大のケフレンシは ⌊`n_fft`/2⌋ / fs（周期は周波数分解能の 2 倍）．
- 変換には `quefrency_axis(c, unit)`，`from_seconds`，`to_seconds` を使う．
  `unit` は `"s"`（= `"Hz^-1"`），`"ms"`（= `"kHz^-1"`），`"period_Hz"` のいずれか．

## 5. リフタ

- リフタは，符号付きケフレンシ [s] から重みを返す関数，または長さ `n_fft` の配列．
- `lowpass`，`highpass`，`bandpass`，`notch` を提供する．
- 境界は，秒（位置引数），`khz_inv=`，`period_hz=` のいずれか 1 つで指定する．
  例：`lowpass(period_hz=300)` は周期 300 Hz より長い波を残す．
  `bandpass(period_hz=(167, 100))`，`notch(period_hz=125, width_khz_inv=2)` のようにも書ける．
- `apply_lifter` は，`Cepstrum` と `Cepstrogram` のどちらにも使える．

## 6. 再構成

時間信号への再構成はケプストグラムだけで行う（`reconstruct_cepstrogram`，`Decomposition.reconstruct`）．
手順は次のとおり．

1. 時間信号から複素スペクトログラム X を計算し，`Cepstrogram` に保持する．
2. リフタ後のケプストグラムを FFT して exp を取り，振幅 |X^ℓ| を得る．
3. ゲイン G = |X^ℓ| / |X| を求める．
4. X · G を逆 FFT して `frame_len` で切り出し，分析窓と同じ合成窓を掛けて重畳加算する．窓の 2 乗の和で正規化する．

- 位相は元のままで，リフタで変わるのは振幅だけである．
- リフタをかけなければ G = 1 で，元の信号に戻る．
- ゲインの分母 |X| は，対数と同じく「フレーム内の最大値 × `eps`」で下から抑える．そのため，`eps` の値によらず，リフタなしなら元の信号に戻る．無音フレームは 0 のままにする．
- 再構成できるのは，`store_spectrogram=True`（既定）で作ったケプストグラムだけである．
- 窓とシフトが定義書の完全再構成の条件を満たさない場合は，`ReconstructionWarning` を出す．
- `low` で再構成した信号をもう一度フレームに分けて分析すると，振幅スペクトルは包絡に近づく．一致はしない．

式は [definitions.md](definitions.md) に示す．

## 7. API

### 高水準

分析ごとに 1 ファイル・1 関数とする．入力は時間信号または `Spectrum` で，戻り値は `Decomposition`．

| ファイル | 関数 | 内容 |
|---|---|---|
| `real_cepstrum.py` | `analyze_real` | 実ケプストラム |
| `cepstrogram_analysis.py` | `analyze_cepstrogram` | ケプストグラム．全フレームに同じリフタを掛ける |

- 分割位置は `lifter=`（関数または配列）で与えるか，低域通過のカットオフを `cutoff`（秒），`khz_inv`，`period_hz` のいずれか 1 つで与える．
  何も指定しなければ分割せず，`low` が `full` と等しく，`high` は 0 になる．
- `Decomposition` は `full`，`low`，`high` をもつ．`low` はリフタを通した成分（スペクトル包絡）で，q = 0 の係数を含む．
  `high` は `full - low`（微細構造）で，`low + high == full` が厳密に成り立つ．
- `log_spectrum(part)` は，その成分の片側対数振幅スペクトル [dB] を返す．周波数軸は `freqs`．
- `reconstruct(part)` は，その成分の時間信号を返す（`analyze_cepstrogram` の結果のみ）．
  `"full"` は元の信号，`"low"` は包絡の振幅と元の位相をもつ信号，`"high"` は `full - low`（時間領域の差）．

### 低水準

| 名前 | 内容 |
|---|---|
| `cepstrum(x, fs)` | 信号全体のケプストラム（`Cepstrum` を返す） |
| `cepstrogram(x, fs, frame_len=...)` | ケプストグラム（`Cepstrogram` を返す） |
| `Spectrum(data, fs, n_fft, kind, sided)` | スペクトル入力 |
| `lowpass` / `highpass` / `bandpass` / `notch` / `apply_lifter` | リフタ |
| `reconstruct_cepstrogram(cg, lifter)` | リフタ後のケプストグラムから時間信号を再構成 |
| `quefrency_axis(c, unit)` / `from_seconds` / `to_seconds` | ケフレンシの単位変換 |

引数と既定値は docstring に記載する．公開 API は `cepstrum/__init__.py` に集める．

## 8. 定義書との対応

definitions.md は片側スペクトル（I 点）を基準に定式化する．実装は両側（M 点）の配列で計算するので，対応を次にまとめる．

| 定義書の量 | 実装 |
|---|---|
| スペクトル | `Spectrum`．`sided="one"` が片側（I 点），`sided="two"` が両側（M 点） |
| 窓関数 | `window`：`"hann"`，`"hamming"`，`"rect"`（`None`），または長さ M の配列 |
| 対数とゲインの下限 ε | `eps`（既定 1e-12） |
| dB 基準 | `db=True` |
| リフタ | `lowpass(c)`，`highpass(c)`，`bandpass(a, b)`，`notch(q_0, W)` |
| 低域成分・高域成分 | `Decomposition.low`，`Decomposition.high` |
| スペクトログラムの保持 | `store_spectrogram=True` |

実装上の補足：

- 鏡像：`Spectrum.to_two_sided()` が，片側のスペクトルに共役対称の鏡像を補って M 点にする．両側で与えたデータは，そのまま使う．
- 配列の長さ：`Cepstrum.data` は M 点の偶対称な配列で，前半の L 点が定義書の $\bm{z}$ にあたる．
  `Cepstrogram.data` も M 行で，前半の L 行が $\bm{Z}$ にあたる．リフタの重みは $\lvert q\rvert$ の関数として M 点全体に掛ける．
- 再構成：保持した片側のスペクトログラムにゲインを掛け，`irfft` で各フレームを実信号に戻す．
- フレーム化：定義書のフレームの開始位置 jH − (M − H) を，信号の先頭に M − H 点のゼロを足して実現する．フレーム数は定義書の J と同じ．
- FFT 長：`n_fft` の既定は `frame_len`（`cepstrum()` では信号長）で，定義書の M にあたる．
  `n_fft` を M より大きくすると，窓を掛けた M 点の後ろにゼロを足して DFT する．窓の長さは M のままで，
  周波数ビン数とケフレンシビン数は ⌊`n_fft`/2⌋ + 1 になる．

## 9. 非機能

- 依存は NumPy のみ．テストには pytest を使う．
- 型ヒントを付ける．

## 10. 検討事項

- 多チャネルの個別処理
- 先細りのリフタ
- dB で与えたスペクトルの入力
- 再構成のゲインの上限（谷を持ち上げすぎないため）
