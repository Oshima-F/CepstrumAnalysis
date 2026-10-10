# ケプストラム分析ライブラリ 要件 (v1)

## 位置づけ
他アプリケーションに組み込める DSP ライブラリ。Python / NumPy のみに依存し、
副作用（描画・ファイルI/O・グローバル状態）を持たない純関数＋不変データクラスで構成する。

## 1. 入力
| 入力 | 仕様 |
|---|---|
| 時間信号 | 1次元 `[N]` または多チャネル `[ch, N]`。多チャネルはチャネル平均でモノラル化 |
| スペクトル | `Spectrum(data, fs, n_fft, kind, sided)` |

- `kind`: `complex` / `amplitude` / `power`（power は √ で振幅へ変換。単位は線形のみ）
- `sided`: `one`（`n_fft//2+1` 点）/ `two`（`n_fft` 点）。内部では両側複素スペクトルへ正規化
- 振幅・パワー入力は位相を持たない → 実/パワーケプストラムのみ計算可。複素ケプストラムは `ValueError`
- 振幅・パワー入力にスケーリング補正（片側の2倍等）は行わない。`data` を |X| としてそのまま扱う

## 2. ケプストラム
- 種類: `real` = IFFT(log|X|) / `complex` = IFFT(log|X| + jφ)（線形位相除去、遅延を保持）/ `power` = |IFFT(log|X|²)|²
- **平均ケプストラム**: 全区間を1回の FFT で計算（方式 (a)）。`cepstrum()`
- **ケプストグラム**: 短時間フレームごとに計算し `[quefrency, frame]` で返す。`cepstrogram()`
  - フレーム長・ホップ・窓（hann/hamming/rect/配列）・FFT 長を指定可
  - 時刻軸 `times`、ケフレンシ軸 `quefrency`（秒）を持つ
- 対数の下限は相対値 `eps`（最大値比）

## 3. リフタリングと再構成
- リフタ: 符号付きクエフレンシー(秒)→重み の関数、または長さ `n_fft` の配列。`lowpass / highpass / bandpass / notch` を提供
- `apply_lifter` は平均ケプストラム・ケプストグラムの両方に適用可能
- `reconstruct` / `reconstruct_cepstrogram`
  - complex: リフタ無しで元信号に数値誤差内で一致（遅延復元込み）
  - real: 位相が無いため最小位相で補う（位相の補い方は `phase` 引数で将来拡張。現状 `"minimum"` のみ）
  - power: 不可逆のため `ValueError`
  - ケプストグラム: フレームごと再構成 → 窓包絡で正規化した重畳加算

## 3.5 クエフレンシーの単位
クエフレンシー q は周波数の双対量で、[s] = [Hz⁻¹]。ケプストラム係数 c は、対数スペクトル上の cos(2π·q·f) の「波」の大きさであり、
**q 個/Hz の波（= 周期 1/q Hz の波）** を表す。

| q | kHz⁻¹（= 山/kHz） | 波の周期 |
|---|---|---|
| 1 ms | 1 | 1000 Hz（フォルマント間隔） |
| 8 ms | 8 | 125 Hz（f0=125 Hz の倍音間隔） |

- 内部の数値・`Cepstrum.quefrency` は秒（= Hz⁻¹）で保持し、API 互換を保つ
- 表示・軸の主単位は **kHz⁻¹**（SI 接頭辞により 1 kHz⁻¹ = 1 ms で、数値は ms と同じ）。「1 kHz あたりのスペクトルの山の数」と読む
- 周期 [Hz]（= 1/q）は補助軸として併記する（非線形のため主軸にはしない）
- cycle 単位であり、rad/Hz ではない（2π 倍の違い）
- q=0 の c[0] は波ではなく全体レベル（ゲイン）で、周期は定義できない（∞）。上限は q = N/(2fs)（周期 2Δf）
- API: `quefrency_axis(ceps, unit)`、unit は `"Hz^-1"`(=`"s"`) / `"kHz^-1"`(=`"ms"`) / `"period_Hz"`。`unit_label(unit)` で軸ラベル、
  `from_seconds` / `to_seconds` で変換
- リフタは秒の位置引数のほか、`khz_inv=` / `period_hz=` で指定可（例：`lowpass(period_hz=300)`＝300 Hz より幅広い波を残す、
  `bandpass(period_hz=(167, 100))`、`notch(period_hz=125, width_khz_inv=2)`）。複数指定はエラー

## 3.6 分析別の高水準 API（low / high 分割）
分析ごとに1ファイル・1関数。入力は時間信号（多チャネルは平均）または `Spectrum`、戻り値は共通の `Decomposition`。

| ファイル | 関数 | 備考 |
|---|---|---|
| `real_cepstrum.py` | `analyze_real` | |
| `complex_cepstrum.py` | `analyze_complex` | 位相が必要（複素 `Spectrum` か時間信号）。n_fft は偶数 |
| `power_cepstrum.py` | `analyze_power` | 不可逆：`log_spectrum` / `reconstruct` は `ValueError` |
| `cepstrogram_analysis.py` | `analyze_cepstrogram` | `kind` = real/complex/power、全フレームに同じリフタを適用 |

- リフタ指定：`lifter=`（関数または配列）、または低域通過のカットオフを `cutoff`[s=Hz⁻¹] / `khz_inv` / `period_hz` のいずれか1つで指定
  （併用はエラー）。未指定なら分割なし（`low`=`full`、`high`=0）
- `Decomposition.full / low / high`：`low` = リフタ通過（スペクトル包絡、c[0] を含む）、`high` = `full - low`（励振・微細構造）。
  `low + high == full` が厳密に成り立つ
- `log_spectrum(part)`：その成分が表す片側対数振幅スペクトル [dB]（`freqs` が周波数軸）。`low` は包絡、`high` は微細構造
- `reconstruct(part)`：その成分の時間信号。complex では `low` と `high` は畳み込みで `full` に一致する準同型成分
  （遅延・符号は `low` が保持し、`low`/`high` とも入力長で切り詰めない）

## 4. 非機能
- 依存: NumPy のみ（テストは pytest）
- 型ヒント付き、公開 API は `cepstrum/__init__.py` に集約
- 未決事項: 振幅/パワー入力時の位相補完方式の拡張、パワーケプストラム定義の流儀、多チャネル個別処理
