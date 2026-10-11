# ケプストラム分析の数学的な整理

## 0. 記号

| 記号 | 意味 |
|---|---|
| $n = 0, \cdots, N-1$ | サンプルビン |
| $i = 0, \cdots, I-1$ | 周波数ビン |
| $l = 0, \cdots, L-1$ | ケフレンシビン |
| $j = 0, \cdots, J-1$ | 時間ビン |
| $f_s$ | サンプリング周波数 |
| $M$ | FFT 長，窓長 |
| $H$ | STFT におけるシフト長 |
| $\mathrm{j}$ | 虚数単位 |
| $\varepsilon$ | 対数の下限（例：$10^{-12}$） |
| $\delta$ | ゲインの下限（例：$10^{-12}$） |
| $w[n]$ | 窓関数 |
| $x[n] \in \mathbb{R}^N$ | 時間信号 |
| $\bm{x} \in \mathbb{C}^I$ | スペクトル |
| $\lvert \bm{x} \rvert \in \mathbb{R}^I$ | 振幅スペクトル |
| $\lvert \bm{x} \rvert^{.2} \in \mathbb{R}^I$ | パワースペクトル |
| $\log \lvert \bm{x} \rvert \in \mathbb{R}^I$ | 対数振幅スペクトル |
| $\bm{z} \in \mathbb{R}^L$ | 実ケプストラム |
| $\mathcal{C}$ | ケプストラム変換 |
| $\mathcal{C}^{-1}$ | 逆ケプストラム変換 |
| $\bm{\ell} \in [0,1]^L$ | リフタの重み |
| $\lambda$ | リフタ係数 |
| $\bm{z}_{\lambda \geq}$ | ローパスリフタリングしたケプストラム |
| $\bm{z}_{\lambda <}$ | ハイパスリフタリングしたケプストラム |
| $\bm{X} \in \mathbb{C}^{I \times J}$ | 複素スペクトログラム |
| $\lvert\bm{X}\rvert \in \mathbb{R}^{I \times J}$ | 振幅スペクトログラム |
| $\lvert\bm{X}\rvert^{.2} \in \mathbb{R}^{I \times J}$ | パワースペクトログラム |
| $\bm{Z} \in \mathbb{R}^{L \times J}$ | 実ケプストグラム |
| $\bm{Z}_{\lambda \geq}$ | ローパスリフタリングした実ケプストグラム |
| $\bm{Z}_{\lambda <}$ | ハイパスリフタリングした実ケプストグラム |

補足

- 時間信号は FFT/STFT を適用してよい長さに整えられている単チャネルの信号とする
- $I$ は片側スペクトルの周波数ビン数であり，$I = \lfloor M/2 \rfloor + 1$ で求められる
- ケプストラムは片側スペクトルに鏡像を結合した $M$ 点から計算され，片側を取り出すため，周波数ビン数とケフレンシビン数は一致する（$L = I$）
- $i$ 番目の周波数ビンは $f_s \frac{i}{M}$ [Hz] で表される（$i = 0$ は直流）
- $\lvert \bm{A} \rvert$ は行列 $\bm{A}$ の要素ごとの絶対値を示す
- $\lvert \bm{A} \rvert^{.2}$ は行列 $\bm{A}$ の要素ごとの 2 乗を示す
- $\log \lvert \bm{A} \rvert$ は行列 $\bm{A}$ の要素ごとの絶対値の自然対数を取ったものを示し，$\varepsilon \lVert \bm{A} \rVert_\infty$ を下限としてフロアリングする
- $\lVert \bm{A} \rVert_\infty$ は，$\bm{A}$ の要素の絶対値の最大値を示す（行列では列ごと）
- $\mathcal{C}$ は，鏡像の結合，DFT 系の変換，片側の取り出しからなる，対数振幅スペクトルから実ケプストラムを求める変換であり，$\bm{z} = \mathcal{C}\{\log \lvert \bm{x} \rvert\}$ と書く．$\mathcal{C}^{-1}$ はその逆変換である
- $\bm{z}$ の $l$ 番目のビンは，スペクトル上の周期 $f_s/l$ [Hz]（ケフレンシ $q_l = l/f_s$ [1/Hz]）間隔の波の成分を示す
- $l = 0$ は全体のレベルを示し，$l$ が小さいほど周期が長い滑らかな包絡成分を示し，$l$ が大きいほど細かい微細構造成分を示す
- $\bm{z}_{\lambda \geq}$ は $l \le \lambda$ を残し，$\bm{z}_{\lambda <}$ は $l > \lambda$ を残す
- リフタリングは $\bm{z} = \bm{z}_{\lambda \geq} + \bm{z}_{\lambda <}$ を満たすように実行される

## 1. 時間信号と窓関数

### 1.1 モノラル化

多チャネル信号 $x_\mathrm{ch}[n]$（$\mathrm{ch} = 1, \dots, C$）はチャネル平均でモノラルにする．

$$x[n] = \frac{1}{C} \sum_{\mathrm{ch}=1}^{C} x_\mathrm{ch}[n]$$

### 1.2 窓関数

窓関数 $w[n]$ は，長さ $M$ の実数列（$n = 0, \dots, M-1$）で，DFT する区間に掛ける．FFT 長 $M$ が決まると，窓の長さも $M$ に決まる．
窓関数は，次のいずれか，または長さ $M$ の任意の実数列とする．

| 窓 | $w[n]$（$n = 0, \dots, M-1$） |
|---|---|
| ハン窓 | $\dfrac12 - \dfrac12\cos\dfrac{2\pi n}{M}$ |
| ハミング窓 | $0.54 - 0.46\cos\dfrac{2\pi n}{M}$ |
| 矩形窓 | $1$ |

- 窓は**周期窓**で，角度の分母が $M$ である（DFT の 1 周期分に対応する形）．
- FFT のみを適用する場合（平均ケプストラム）は，信号全体が DFT する区間で，$M = N$ である．

### 1.3 窓かけと DFT

FFT のみを適用する場合（$M = N$）は，$x[n]$ に窓を掛けて DFT し，非負周波数側の $I$ 点がスペクトル $\bm{x}$ である．

$$\bm{x}[i] = \sum_{n=0}^{M-1} w[n]\, x[n]\, e^{-\mathrm{j} 2\pi i n / M}, \qquad i = 0, \dots, I-1$$

以降の「スペクトル」は，この $\bm{x}$ を指す．STFT では，各フレームに同じ式を適用する．
冗長であるため，周波数および時間周波数領域の変数では「[i]」は省略する（和の中など，式に必要な場合を除く）．

## 2. スペクトルの入力形式

スペクトルは，振幅・パワー・複素のいずれかで与えられる．非負周波数側 $i = 0, \dots, I-1$ の値を，次のように $\bm{x}$ とする．

- **複素入力**：$\bm{x}$ が与えられる（ケプストラムの計算には振幅 $\lvert\bm{x}\rvert$ を使う）
- **振幅入力**：$\lvert\bm{x}\rvert$ が与えられる（位相は 0）
- **パワー入力**：$\lvert\bm{x}\rvert^{.2}$ が与えられる（位相は 0）．振幅は要素ごとの平方根 $\lvert\bm{x}\rvert = \sqrt{\lvert\bm{x}\rvert^{.2}}$ で得る

片側（$I$ 点）で与えると，そのまま $\bm{x}$ になる．両側（$M$ 点）で与えると，前半の $I$ 点が $\bm{x}$ に当たる．与えた値をそのまま使う．

## 3. 対数振幅スペクトル

$$\log\lvert\bm{x}\rvert = \ln \max\bigl(\lvert\bm{x}\rvert,\ \varepsilon \lVert\bm{x}\rVert_\infty\bigr)$$

- $\ln$ は自然対数．$\varepsilon$ は対数の下限を決める定数（例：$10^{-12}$）．振幅が $\varepsilon$ × 最大値より小さい周波数は，下限に切り上げる（$\ln 0$ を避けるため）．
  下限は，絶対値でも最小の正の浮動小数点数以上に保つ．
- **dB 基準**：自然対数を dB に換算する．

$$\bm{s}_\mathrm{dB} = \frac{20}{\ln 10}\,\log\lvert\bm{x}\rvert \;=\; 20\log_{10}\max(\cdots) \qquad \left(\frac{20}{\ln 10} \approx 8.686\right)$$

以降，$\bm{s} = \log\lvert\bm{x}\rvert$ と書く（dB 基準で扱うときは $\bm{s}_\mathrm{dB}$ に置き換える）．

## 4. 実ケプストラム

実ケプストラムは，片側の対数振幅スペクトル $\bm{s}$（$I$ 点）に鏡像を結合し，$M$ 点の DFT 系で変換し，片側だけを取り出して得る．

**(1) 鏡像の結合**：$\bm{s}$ の負の周波数側に当たる鏡像を結合した $M$ 点の配列 $\bm{s}^{\mathrm{m}}$ を作る．$U = \lceil M/2 \rceil - 1$ として，

$$\bm{s}^{\mathrm{m}}[i] = \bm{s}[i]\ (i = 0, \dots, I-1), \qquad \bm{s}^{\mathrm{m}}[M-i] = \bm{s}[i]\ (i = 1, \dots, U)$$

$\bm{s}^{\mathrm{m}}$ は実数で偶対称（$\bm{s}^{\mathrm{m}}[M-i] = \bm{s}^{\mathrm{m}}[i]$）である．

**(2) DFT・逆 DFT の定義**：長さ $M$ の列 $\bm{a}$，$\bm{b}$ に対して，

$$\mathcal{F}\{\bm{a}\}[k] = \sum_{n=0}^{M-1}\bm{a}[n]\,e^{-\mathrm{j}2\pi kn/M}, \qquad
\mathcal{F}^{-1}\{\bm{b}\}[n] = \frac1M\sum_{k=0}^{M-1}\bm{b}[k]\,e^{\mathrm{j}2\pi kn/M}$$

**(3) ケプストラム分析**：鏡像を結合した対数振幅スペクトルを逆 DFT する．

$$\bm{z}^{\mathrm{m}} = \mathcal{F}^{-1}\{\bm{s}^{\mathrm{m}}\}$$

$\bm{s}^{\mathrm{m}}$ が実数で偶対称なので，$\bm{z}^{\mathrm{m}}$ も実数で偶対称（$\bm{z}^{\mathrm{m}}[M-l] = \bm{z}^{\mathrm{m}}[l]$）になる．

**(4) 片側の取り出し**：$\bm{z}^{\mathrm{m}}$ の $l = 0, \dots, L-1$ を実ケプストラム $\bm{z}$ とする（$\bm{z}[l] = \bm{z}^{\mathrm{m}}[l]$）．

(1)〜(4) の全体を $\mathcal{C}$ と書く：$\bm{z} = \mathcal{C}\{\bm{s}\}$．

**逆変換**：$\bm{z}$ に同じ流れを順に適用する．(1) 鏡像を結合した $\bm{z}^{\mathrm{m}}$（$\bm{z}^{\mathrm{m}}[M-l] = \bm{z}[l]$，$l = 1, \dots, U$）を作り，
(3) DFT して実部を取り，(4) 前半の $I$ 点を取り出す．これを $\mathcal{C}^{-1}$ と書く．

$$\bm{s} = \mathcal{C}^{-1}\{\bm{z}\} = \bigl(\mathrm{Re}\,\mathcal{F}\{\bm{z}^{\mathrm{m}}\}\bigr)[0, \dots, I-1]$$

dB 基準のときは $\bm{s}$ を $\bm{s}_\mathrm{dB}$ にした $\bm{z}_\mathrm{dB} = \mathcal{C}\{\bm{s}_\mathrm{dB}\} = \frac{20}{\ln 10}\bm{z}$ になる．
余弦の和で直接書いた形は，付録に示す．

**性質**

- $\bm{z}$ は実数．
- $\bm{z}[0] = \frac1M\sum_{i=0}^{M-1}\bm{s}^{\mathrm{m}}[i]$：対数振幅の（全周波数にわたる）平均で，全体のレベルを表す．
- $\mathcal{C}^{-1}\{\mathcal{C}\{\bm{s}\}\} = \bm{s}$：リフタをかけなければ元の対数スペクトルに戻る．$\exp$ で振幅 $\lvert\bm{x}\rvert$ に戻る．位相は $\bm{z}$ に含まれない．
- **リフタ**：重みを $\lvert q\rvert$ の関数として，偶対称な $M$ 点全体に掛ける．鏡像側にも同じ重みが掛かり，片側の $\bm{\ell}[l]$ を掛けた結果と一致する．

## 5. パワーケプストラム

$$\lvert\mathcal{C}\{2\bm{s}\}\rvert^{.2}\ [l], \qquad l = 0, \dots, L-1$$

$2\bm{s} = \ln\lvert\bm{x}\rvert^{.2}$ はパワーの対数．dB 基準のときは $2\bm{s}$ の代わりに $\bm{s}_\mathrm{dB}$（$= 10\log_{10}\lvert\bm{x}\rvert^{.2}$）を使う．
パワーケプストラムは，パワースペクトルの対数に含まれる周期構造の強さを，大きさの 2 乗で表した量である．

## 6. ケフレンシと単位

$$q_l = \frac{l}{f_s}\ [\mathrm{s}], \qquad l = 0, \dots, L-1$$

$\bm{z}[l]$ は，対数スペクトルの周波数 $f_i = f_s\, i / M$ に対する余弦波 $\cos(2\pi q_l f_i) = \cos(2\pi i l / M)$ の係数である．
この波の**周期**（周波数軸上の山から山まで）と**密度**は次のとおり．

$$T_l = \frac{1}{q_l} = \frac{f_s}{l}\ [\mathrm{Hz}]\ (l \ge 1), \qquad \text{密度} = q_l\ [\text{個/Hz}] = q_l\ [\mathrm{s}]$$

つまり $[\mathrm{s}] = [\mathrm{Hz}^{-1}]$．SI 接頭辞により $1\ \mathrm{kHz}^{-1} = 1\ \mathrm{ms}$ で，数値は ms と同じになる．
「$q = 8\ \mathrm{kHz}^{-1}$」は「$1\ \mathrm{kHz}$ あたり 8 個の山」，すなわち周期 $125\ \mathrm{Hz}$ の波を表す．
波の数はサイクル単位で数える（$\mathrm{rad}$ 単位の値は $2\pi$ 倍）．$l = 0$ は周期が無限大の成分（全体のレベル）．
表現できる最大のケフレンシーは $q_{L-1} = (L-1)/f_s = \lfloor M/2\rfloor / f_s$（$M$ が偶数なら周期 $2 f_s / M$，周波数の分解能の 2 倍）．

## 7. リフタ

リフタは，各ビンに掛ける重み $\bm{\ell}[l] \in [0, 1]$（$l = 0, \dots, L-1$）で，ケフレンシ $q_l$ の関数として与える．
リフタリングしたケプストラムは，要素ごとの積である．

$$\bm{z}^{\ell}[l] = \bm{\ell}[l]\,\bm{z}[l]$$

| リフタ | 重み $\bm{\ell}[l]$（$q_l \ge 0$ の関数） |
|---|---|
| 低域通過（境界 $c$） | $1\ (q_l \le c)$，それ以外 $0$．$\bm{z}_{\lambda\geq}$（$\lambda = \lfloor c f_s \rfloor$）に対応 |
| 高域通過（境界 $c$） | $1\ (q_l \ge c)$，それ以外 $0$（$c > 0$ なら $\bm{z}[0]$ の重みは 0） |
| 帯域通過（境界 $a, b$） | $1\ (a \le q_l \le b)$，それ以外 $0$ |
| ノッチ（中心 $q_0$，幅 $W$） | $0\ (\lvert q_l - q_0\rvert \le W/2)$，それ以外 $1$ |

境界は，ケフレンシ（秒，$=\mathrm{Hz}^{-1}$）か，周期（$q = 1/T_l$）のいずれかで与える．

## 8. 低・高ケフレンシ成分への分離

低域通過のリフタ（係数 $\lambda$）で，ケプストラムを 2 つに分ける．

$$\bm{z}_{\lambda\geq}[l] = \begin{cases}\bm{z}[l] & (l \le \lambda)\\ 0 & (l > \lambda)\end{cases}, \qquad
\bm{z}_{\lambda<} = \bm{z} - \bm{z}_{\lambda\geq}$$

任意の重み $\bm{\ell}_\mathrm{low}$（先細りのリフタなど）では，$\bm{z}_\mathrm{low} = \bm{\ell}_\mathrm{low}\odot\bm{z}$，$\bm{z}_\mathrm{high} = \bm{z} - \bm{z}_\mathrm{low}$ とする．
いずれも $\bm{z}_\mathrm{low} + \bm{z}_\mathrm{high} = \bm{z}$ が厳密に成り立つ．
対応する対数スペクトルは，それぞれに $\mathcal{C}^{-1}$ を適用したものである．

$$\bm{s}_\mathrm{low} = \mathcal{C}^{-1}\{\bm{z}_\mathrm{low}\}, \qquad
\bm{s}_\mathrm{high} = \mathcal{C}^{-1}\{\bm{z}_\mathrm{high}\}, \qquad \bm{s} = \bm{s}_\mathrm{low} + \bm{s}_\mathrm{high}$$

- $\bm{s}_\mathrm{low}$：滑らかな成分（**スペクトル包絡**）．$\bm{z}_\mathrm{low}$ は $\bm{z}[0]$ を含み，全体のレベルを持つ．
- $\bm{s}_\mathrm{high}$：細かい成分（**微細構造**，音声ならピッチによる倍音の山谷）．平均は 0．
- 音声が，声道や口腔の形状に関する $P$ と，声帯振動に関する $Q$ の積 $\lvert\bm{x}\rvert = \lvert P\rvert\,\lvert Q\rvert$ で表せるなら，$\bm{s} = \ln\lvert P\rvert + \ln\lvert Q\rvert$
  と和になり，遅く変化する $\ln\lvert P\rvert$ が低ケフレンシ側，周期的な $\ln\lvert Q\rvert$ が高ケフレンシ側に分かれる．

自然対数基準の $\bm{s}_\mathrm{low}$，$\bm{s}_\mathrm{high}$ を dB で表すには，$\frac{20}{\ln 10}$ 倍する．

## 9. ケプストグラムと時間信号の再構成

### 9.1 ケプストグラム（短時間ケプストラム）

スペクトログラム $\bm{X}$ は，$x[n]$ の STFT（窓 $w$，フレーム長 $M$，シフト $H$）である．$j$ 番目のフレームのスペクトルを $\bm{X}$ の $j$ 列とする．

$$\bm{X}[i, j] = \sum_{n=0}^{M-1} w[n]\, x[jH + n]\, e^{-\mathrm{j} 2\pi i n/M}, \qquad i = 0, \dots, I-1,\ \ j = 0, \dots, J-1$$

各フレームのスペクトル $\bm{X}[\cdot, j]$ に，対数振幅スペクトルからケプストラムを求める変換（実ケプストラム，またはパワーケプストラム）を適用したものが，ケプストグラムの $j$ 列 $\bm{Z}[\cdot, j]$．
スペクトログラム $\bm{X}$ は再構成のために保持する．

### 9.2 再構成

元のスペクトログラム $\bm{X}$ と，リフタ後のケプストグラムが表す対数振幅から，**ゲイン**を作る．

$$\log\lvert\bm{X}^{\ell}\rvert\,[\cdot, j] = \mathcal{C}^{-1}\{\bm{\ell}\odot\bm{Z}[\cdot, j]\}$$

元のパワー $\lvert\bm{X}\rvert^{.2}$ と，リフタ後のパワー $\lvert\bm{X}^{\ell}\rvert^{.2}$ から，

$$\bm{G} = \frac{\lvert\bm{X}^{\ell}\rvert}{\max\bigl(\lvert\bm{X}\rvert,\ \delta \lVert\bm{X}\rVert_\infty\bigr)} = \sqrt{\frac{\lvert\bm{X}^{\ell}\rvert^{.2}}{\lvert\bm{X}\rvert^{.2}}}\quad(\delta = 10^{-12})$$

$$\bm{Y} = \bm{G} \odot \bm{X}$$

$\bm{Y}$ を逆 STFT して，時間信号 $\hat x[n]$ を得る．逆 STFT は，各フレームを（鏡像を補った）$M$ 点の逆 DFT で実信号に戻し，
合成窓（分析窓と同じ $w$）をかけて重畳加算し，窓の 2 乗の和で正規化する標準の方法である．

**性質**

- **リフタなし**：$\bm{\ell} \equiv 1$ なら $\lvert\bm{X}^{\ell}\rvert = \lvert\bm{X}\rvert$，$\bm{G} = 1$，$\bm{Y} = \bm{X}$ となり，逆 STFT は元の信号 $x$ を返す
  （窓とシフトが $\sum_j w^2 > 0$ を満たす限り，数値誤差 $10^{-15}$ 程度で一致）．
- **位相**：$\arg\bm{Y} = \arg\bm{X}$．ゲインは実数なので，位相は元のまま変わらず，**変わるのは振幅だけ**．
- **ゲインの形**：パワーの比 $\lvert\bm{X}^{\ell}\rvert^{.2}/\lvert\bm{X}\rvert^{.2}$ の平方根（振幅の比）を使う．これは，出力の振幅 $\lvert\bm{Y}\rvert = \lvert\bm{X}^{\ell}\rvert$ となる
  ゲインである．パワー比を振幅の複素スペクトルに掛けると，変化量が dB で 2 倍になる．
- **低域成分・高域成分**：低域成分は，低域の $\bm{Z}_{\lambda\geq}$ から作ったゲインによる再構成 $\hat x_\mathrm{low}$（包絡の振幅と元の位相を持つ信号）．
  高域成分は時間領域の差 $\hat x_\mathrm{high} = x - \hat x_\mathrm{low}$ で，$\hat x_\mathrm{low} + \hat x_\mathrm{high} = x$ が成り立つ．

**再構成の特性**

1. ゲインが実数なので，位相は元のまま（$\arg\bm{Y} = \arg\bm{X}$）．
2. 逆 DFT は $M$ 点で巡回するため，ゲインが長い応答を持つと，$M$ 点を超える成分は巡回して折り返す．滑らかなゲイン（低域のリフタ）では小さい．
3. 重畳加算で隣のフレームと混ざるため，再構成した信号をもう一度フレームに分けたときの振幅は，目標の $\lvert\bm{X}^{\ell}\rvert$ に近づく．信号の周期性は，元の位相の周期的な構造が決める．

再構成には，実ケプストグラムと，元の複素スペクトログラム $\bm{X}$ を用いる．

## 10. 入力の検査の定義

| 検出 | 定義 |
|---|---|
| 無音・定数 | モノラル信号の最大値と最小値の差が 0（全ゼロを含む）．スペクトルでは全要素が 0 |
| クリッピング | 信号の最大値，または最小値（0 でないもの）から $10^{-9}\max\lvert x\rvert$ 以内の値が，**3 サンプル以上連続**する．正負は別々に判定し，多チャネルはチャネルごと |
| 不正な入力 | NaN・inf，複素の時間信号，空の信号，形状が長さ $N$ の 1 次元信号またはチャネル数 × $N$ の 2 次元信号以外，$f_s$ が有限の正数以外，$H \notin [1, M]$，$M < 1$ など |

ケプストグラムでは，これらの検査を信号全体 $x$ に対して 1 回行う（フレーム単位の無音は正常として通す）．

## Appendix．余弦変換による表現

実ケプストラムの変換 $\mathcal{C}$，$\mathcal{C}^{-1}$ は，鏡像を畳み込んだ余弦変換（DCT-I 型）として直接書ける．
$\bm{s}^{\mathrm{m}}$ が偶対称で，$\bm{z}^{\mathrm{m}}$ が実数であることから，$U = \lceil M/2 \rceil - 1$ として，次が成り立つ．最後の項は $M$ が偶数のときだけ現れる．

$$\bm{z}[l] = \frac{1}{M}\Bigl(\bm{s}[0] + 2\sum_{i=1}^{U}\bm{s}[i]\cos\frac{2\pi i l}{M} + (-1)^{l}\,\bm{s}[M/2]\Bigr), \qquad l = 0, \dots, L-1$$

$$\bm{s}[i] = \bm{z}[0] + 2\sum_{l=1}^{U}\bm{z}[l]\cos\frac{2\pi i l}{M} + (-1)^{i}\,\bm{z}[M/2], \qquad i = 0, \dots, I-1$$

また $\bm{z}[0] = \frac1M\bigl(\bm{s}[0] + 2\sum_{i=1}^{U}\bm{s}[i] + \bm{s}[M/2]\bigr)$ である．
