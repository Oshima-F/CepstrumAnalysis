# ケプストラム分析の数学的な整理

日本語 | [English](definitions.en.md)

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
| $\varepsilon$ | 対数とゲインの下限（例：$10^{-12}$） |
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
| $\bm{z}_\mathrm{low}$ | ローパスリフタリングしたケプストラム（低ケフレンシ成分） |
| $\bm{z}_\mathrm{high}$ | 残りの高ケフレンシ成分 $\bm{z} - \bm{z}_\mathrm{low}$ |
| $\bm{X} \in \mathbb{C}^{I \times J}$ | 複素スペクトログラム |
| $\lvert\bm{X}\rvert \in \mathbb{R}^{I \times J}$ | 振幅スペクトログラム |
| $\lvert\bm{X}\rvert^{.2} \in \mathbb{R}^{I \times J}$ | パワースペクトログラム |
| $\bm{Z} \in \mathbb{R}^{L \times J}$ | 実ケプストグラム |
| $\bm{Z}_\mathrm{low}$ | ローパスリフタリングした実ケプストグラム（低ケフレンシ成分） |
| $\bm{Z}_\mathrm{high}$ | 残りの高ケフレンシ成分 $\bm{Z} - \bm{Z}_\mathrm{low}$ |

- 時間信号は，FFT・STFT を適用できる長さに整えた単チャネルの信号とする．
- $I$ は片側スペクトルの周波数ビン数で，$I = \lfloor M/2 \rfloor + 1$ である．
- ケプストラムは，片側スペクトルに鏡像を結合した $M$ 点から計算し，片側を取り出す．したがって $L = I$ である．
- $i$ 番目の周波数ビンは $f_s \frac{i}{M}$ [Hz] である（$i = 0$ は直流）．
- $\lvert \bm{A} \rvert$ は行列 $\bm{A}$ の要素ごとの絶対値，$\lvert \bm{A} \rvert^{.2}$ は要素ごとの 2 乗である．
- $\log \lvert \bm{A} \rvert$ は要素ごとの絶対値の自然対数で，$\varepsilon \lVert \bm{A} \rVert_\infty$ を下限とする．
- $\lVert \bm{A} \rVert_\infty$ は要素の絶対値の最大値である．行列では列ごとに取る．
- $\mathcal{C}$ は対数振幅スペクトルから実ケプストラムを求める変換で，$\bm{z} = \mathcal{C}\{\log \lvert \bm{x} \rvert\}$ と書く．$\mathcal{C}^{-1}$ はその逆変換である．
- $\bm{z}_\mathrm{low}$ は低域通過のリフタを掛けた成分，$\bm{z}_\mathrm{high}$ はその残りである．$\bm{Z}_\mathrm{low}$，$\bm{Z}_\mathrm{high}$ は各フレームの $\bm{z}_\mathrm{low}$，$\bm{z}_\mathrm{high}$ を並べたものである．

## 1. 時間信号と窓関数

### 1.1 モノラル化

多チャネル信号 $x_\mathrm{ch}[n]$（$\mathrm{ch} = 1, \dots, C$）は，チャネル平均でモノラルにする．

$$x[n] = \frac{1}{C} \sum_{\mathrm{ch}=1}^{C} x_\mathrm{ch}[n]$$

### 1.2 窓関数

窓関数 $w[n]$（$n = 0, \dots, M-1$）は，DFT の区間に掛ける長さ $M$ の実数列である．
次のいずれか，または長さ $M$ の任意の実数列とする．

| 窓 | $w[n]$（$n = 0, \dots, M-1$） |
|---|---|
| ハン窓 | $\dfrac12 - \dfrac12\cos\dfrac{2\pi n}{M}$ |
| ハミング窓 | $0.54 - 0.46\cos\dfrac{2\pi n}{M}$ |
| 矩形窓 | $1$ |

いずれも周期窓で，余弦の分母は $M$ である．

### 1.3 窓かけと DFT

信号全体に 1 回の DFT を適用する場合は $M = N$ とする．$x[n]$ に窓を掛けて DFT し，非負周波数側の $I$ 点をスペクトル $\bm{x}$ とする．

$$\bm{x}[i] = \sum_{n=0}^{M-1} w[n]\, x[n]\, e^{-\mathrm{j} 2\pi i n / M}, \qquad i = 0, \dots, I-1$$

以降，スペクトルはこの $\bm{x}$ を指す．STFT では，各フレームに同じ式を適用する．
周波数領域と時間周波数領域の変数では，式に必要な場合を除き，添字 $[i]$ を省略する．

## 2. スペクトルの入力形式

スペクトルは，複素・振幅・パワーのいずれかで与える．非負周波数側 $i = 0, \dots, I-1$ の値から，次のように $\bm{x}$ を定める．

- 複素：$\bm{x}$ をそのまま使う．ケプストラムの計算には振幅 $\lvert\bm{x}\rvert$ だけを使う．
- 振幅：$\lvert\bm{x}\rvert$ が与えられる．位相は 0 とする．
- パワー：$\lvert\bm{x}\rvert^{.2}$ が与えられる．振幅は要素ごとの平方根 $\lvert\bm{x}\rvert = \sqrt{\lvert\bm{x}\rvert^{.2}}$ で得る．位相は 0 とする．

片側（$I$ 点）で与えた場合はそのまま $\bm{x}$ とし，両側（$M$ 点）で与えた場合は前半の $I$ 点を $\bm{x}$ とする．値の補正はしない．

## 3. 対数振幅スペクトル

$$\log\lvert\bm{x}\rvert = \ln \max\bigl(\lvert\bm{x}\rvert,\ \varepsilon \lVert\bm{x}\rVert_\infty\bigr)$$

$\ln$ は自然対数である．振幅が最大値の $\varepsilon$ 倍より小さい周波数では，$\ln 0$ を避けるため，その値で置き換える．

dB で表すときは，自然対数を換算する．

$$\bm{s}_\mathrm{dB} = \frac{20}{\ln 10}\,\log\lvert\bm{x}\rvert \;=\; 20\log_{10}\max(\cdots) \qquad \left(\frac{20}{\ln 10} \approx 8.686\right)$$

以降，$\bm{s} = \log\lvert\bm{x}\rvert$ と書く．dB 基準で扱うときは $\bm{s}$ を $\bm{s}_\mathrm{dB}$ に置き換える．

## 4. 実ケプストラム

実ケプストラムは，片側の対数振幅スペクトル $\bm{s}$（$I$ 点）に鏡像を結合し，$M$ 点の逆 DFT をとり，片側を取り出して得る．

(1) 鏡像の結合．$U = \lceil M/2 \rceil - 1$ として，$\bm{s}$ に負の周波数側の鏡像を結合した $M$ 点の列 $\bm{s}^{\mathrm{m}}$ を作る．

$$\bm{s}^{\mathrm{m}}[i] = \bm{s}[i]\ (i = 0, \dots, I-1), \qquad \bm{s}^{\mathrm{m}}[M-i] = \bm{s}[i]\ (i = 1, \dots, U)$$

$\bm{s}^{\mathrm{m}}$ は実数で偶対称（$\bm{s}^{\mathrm{m}}[M-i] = \bm{s}^{\mathrm{m}}[i]$）である．

(2) DFT と逆 DFT．長さ $M$ の列 $\bm{a}$，$\bm{b}$ に対して，次のように定義する．

$$\mathcal{F}\{\bm{a}\}[k] = \sum_{n=0}^{M-1}\bm{a}[n]\,e^{-\mathrm{j}2\pi kn/M}, \qquad
\mathcal{F}^{-1}\{\bm{b}\}[n] = \frac1M\sum_{k=0}^{M-1}\bm{b}[k]\,e^{\mathrm{j}2\pi kn/M}$$

(3) 逆 DFT．鏡像を結合した対数振幅スペクトルを逆 DFT する．

$$\bm{z}^{\mathrm{m}} = \mathcal{F}^{-1}\{\bm{s}^{\mathrm{m}}\}$$

$\bm{s}^{\mathrm{m}}$ が実数で偶対称なので，$\bm{z}^{\mathrm{m}}$ も実数で偶対称（$\bm{z}^{\mathrm{m}}[M-l] = \bm{z}^{\mathrm{m}}[l]$）になる．

(4) 片側の取り出し．$\bm{z}^{\mathrm{m}}$ の $l = 0, \dots, L-1$ を実ケプストラム $\bm{z}$ とする（$\bm{z}[l] = \bm{z}^{\mathrm{m}}[l]$）．

(1)〜(4) をまとめて $\mathcal{C}$ と書く：$\bm{z} = \mathcal{C}\{\bm{s}\}$．

逆変換 $\mathcal{C}^{-1}$ は，$\bm{z}$ に鏡像を結合した $\bm{z}^{\mathrm{m}}$（$\bm{z}^{\mathrm{m}}[M-l] = \bm{z}[l]$，$l = 1, \dots, U$）を DFT し，実部の前半 $I$ 点を取り出す．

$$\bm{s} = \mathcal{C}^{-1}\{\bm{z}\} = \bigl(\mathrm{Re}\,\mathcal{F}\{\bm{z}^{\mathrm{m}}\}\bigr)[0, \dots, I-1]$$

dB 基準では $\bm{z}_\mathrm{dB} = \mathcal{C}\{\bm{s}_\mathrm{dB}\} = \frac{20}{\ln 10}\bm{z}$ である．
余弦の和で直接書いた形は付録に示す．

実ケプストラムは次の性質をもつ．

- $\bm{z}[0] = \frac1M\sum_{i=0}^{M-1}\bm{s}^{\mathrm{m}}[i]$ は，全周波数にわたる対数振幅の平均で，全体のレベルを表す．
- $\mathcal{C}^{-1}\{\mathcal{C}\{\bm{s}\}\} = \bm{s}$ である．リフタをかけなければ元の対数スペクトルに戻り，$\exp$ をとると振幅 $\lvert\bm{x}\rvert$ に戻る．
- 位相の情報は $\bm{z}$ に含まれない．

## 5. ケフレンシと単位

$$q_l = \frac{l}{f_s}\ [\mathrm{s}], \qquad l = 0, \dots, L-1$$

$\bm{z}[l]$ は，対数スペクトルの周波数 $f_i = f_s\, i / M$ に対する余弦波 $\cos(2\pi q_l f_i) = \cos(2\pi i l / M)$ の係数である．
この波の周期（周波数軸上の山から山までの幅）と密度は次のとおりである．

$$T_l = \frac{1}{q_l} = \frac{f_s}{l}\ [\mathrm{Hz}]\ (l \ge 1), \qquad \text{密度} = q_l\ [\text{個/Hz}] = q_l\ [\mathrm{s}]$$

したがって $[\mathrm{s}] = [\mathrm{Hz}^{-1}]$ である．SI 接頭辞により $1\ \mathrm{kHz}^{-1} = 1\ \mathrm{ms}$ なので，数値は ms と同じになる．
たとえば $q = 8\ \mathrm{kHz}^{-1}$ は，$1\ \mathrm{kHz}$ あたり 8 個の山，すなわち周期 $125\ \mathrm{Hz}$ の波を表す．

$l = 0$ の成分は周期が無限大で，全体のレベルにあたる．
最大のケフレンシは $q_{L-1} = (L-1)/f_s = \lfloor M/2\rfloor / f_s$ である．$M$ が偶数なら，その周期 $2 f_s / M$ は周波数分解能の 2 倍になる．

## 6. リフタ

リフタは，各ビンに掛ける重み $\bm{\ell}[l] \in [0, 1]$（$l = 0, \dots, L-1$）で，ケフレンシ $q_l$ の関数として与える．
リフタリングは要素ごとの積である．

$$\bm{z}^{\ell}[l] = \bm{\ell}[l]\,\bm{z}[l]$$

| リフタ | 重み $\bm{\ell}[l]$（$q_l \ge 0$ の関数） |
|---|---|
| 低域通過（境界 $c$） | $q_l \le c$ で $1$，それ以外で $0$．$\lambda = \lfloor c f_s \rfloor$ とすると，$l \le \lambda$ の成分を残す |
| 高域通過（境界 $c$） | $q_l > c$ で $1$，それ以外で $0$．$\lambda = \lfloor c f_s \rfloor$ とすると，$l > \lambda$ の成分を残す．$\bm{z}[0]$ の重みは常に 0 |
| 帯域通過（境界 $a, b$） | $a < q_l \le b$ で $1$，それ以外で $0$．境界 $a$ の高域通過と境界 $b$ の低域通過の積に等しい |
| ノッチ（中心 $q_0$，幅 $W$） | $\lvert q_l - q_0\rvert \le W/2$ で $0$，それ以外で $1$ |

境界は，ケフレンシ [s] か周期 [Hz]（$q = 1/T$）で与える．
$M$ 点の $\bm{z}^{\mathrm{m}}$ に掛ける場合は，重みを $\lvert q\rvert$ の関数として鏡像側にも同じ値を掛ける．結果は片側に $\bm{\ell}$ を掛けたものと一致する．

## 7. 低・高ケフレンシ成分への分離

低域通過のリフタの重みを $\bm{\ell}_\mathrm{low}$ とし，ケプストラムを 2 つに分ける．

$$\bm{z}_\mathrm{low} = \bm{\ell}_\mathrm{low}\odot\bm{z}, \qquad \bm{z}_\mathrm{high} = \bm{z} - \bm{z}_\mathrm{low}$$

矩形の低域通過リフタでは，リフタ係数 $\lambda$ を境界のビンとして，$\bm{\ell}_\mathrm{low}[l]$ は $l \le \lambda$ で 1，$l > \lambda$ で 0 である．
このとき $\bm{z}_\mathrm{low}$ は $l \le \lambda$ の成分を，$\bm{z}_\mathrm{high}$ は $l > \lambda$ の成分を残す．
先細りのリフタなど任意の重みでも，$\bm{z}_\mathrm{low} + \bm{z}_\mathrm{high} = \bm{z}$ が厳密に成り立つ．
対応する対数スペクトルは，それぞれに $\mathcal{C}^{-1}$ を適用して得る．

$$\bm{s}_\mathrm{low} = \mathcal{C}^{-1}\{\bm{z}_\mathrm{low}\}, \qquad
\bm{s}_\mathrm{high} = \mathcal{C}^{-1}\{\bm{z}_\mathrm{high}\}, \qquad \bm{s} = \bm{s}_\mathrm{low} + \bm{s}_\mathrm{high}$$

- $\bm{s}_\mathrm{low}$ は滑らかな成分で，スペクトル包絡を表す．$\bm{z}[0]$ を含むので，全体のレベルももつ．
- $\bm{s}_\mathrm{high}$ は細かい成分で，微細構造を表す．音声ではピッチによる倍音の山谷にあたる．平均は 0 である．

音声の振幅スペクトルが，声道の特性 $P$ と声帯振動の特性 $Q$ の積 $\lvert\bm{x}\rvert = \lvert P\rvert\,\lvert Q\rvert$ で表せるとする．
このとき $\bm{s} = \ln\lvert P\rvert + \ln\lvert Q\rvert$ と和になり，周波数方向に緩やかに変化する $\ln\lvert P\rvert$ は低ケフレンシ側に，周期的な $\ln\lvert Q\rvert$ は高ケフレンシ側に分かれる．

$\bm{s}_\mathrm{low}$，$\bm{s}_\mathrm{high}$ を dB で表すには，$\frac{20}{\ln 10}$ 倍する．

## 8. ケプストグラムと時間信号の再構成

### 8.1 ケプストグラム

スペクトログラム $\bm{X}$ は，窓 $w$，フレーム長 $M$，シフト $H$ による $x[n]$ の STFT で，$j$ 列が $j$ 番目のフレームのスペクトルである．
$x[n]$ は $0 \le n \le N-1$ の外では 0 とする．$j$ 番目のフレームの開始位置 $t_j$ とフレーム数 $J$ を次のように定める．

$$t_j = jH - (M - H), \qquad J = \left\lceil \frac{N + M - H}{H} \right\rceil$$

先頭のフレームは信号の先頭より $M - H$ 点前から始まり，最後のフレームは信号の末尾を含む．
$H$ が $M$ を割り切るとき，$0 \le n \le N-1$ のすべてのサンプルは，ちょうど $M/H$ 個のフレームに含まれる．

$$\bm{X}[i, j] = \sum_{m=0}^{M-1} w[m]\, x[t_j + m]\, e^{-\mathrm{j} 2\pi i m/M}, \qquad i = 0, \dots, I-1,\ \ j = 0, \dots, J-1$$

ケプストグラム $\bm{Z}$ の $j$ 列は，各フレームの対数振幅スペクトルに $\mathcal{C}$ を適用したものである．

$$\bm{Z}[\cdot, j] = \mathcal{C}\{\log\lvert\bm{X}[\cdot, j]\rvert\}$$

### 8.2 再構成

リフタ後のケプストグラムから，各フレームの対数振幅を求める．

$$\log\lvert\bm{X}^{\ell}\rvert\,[\cdot, j] = \mathcal{C}^{-1}\{\bm{\ell}\odot\bm{Z}[\cdot, j]\}$$

元の振幅との比をゲイン $\bm{G}$ とし，元のスペクトログラムに掛ける．分母は，対数と同じ $\varepsilon \lVert\bm{X}\rVert_\infty$ で下から抑える．

$$\bm{G} = \frac{\lvert\bm{X}^{\ell}\rvert}{\max\bigl(\lvert\bm{X}\rvert,\ \varepsilon \lVert\bm{X}\rVert_\infty\bigr)}, \qquad \bm{Y} = \bm{G} \odot \bm{X}$$

$\bm{Y}$ を逆 STFT して，時間信号 $\hat x[n]$ を得る．
まず，各フレーム $\bm{Y}[\cdot, j]$ に共役対称の鏡像（$\bm{Y}^{\mathrm{m}}[M-i, j] = \overline{\bm{Y}[i, j]}$，$i = 1, \dots, U$）を結合して $M$ 点にし，逆 DFT で実信号 $y_j$ に戻す．

$$y_j[m] = \mathcal{F}^{-1}\{\bm{Y}^{\mathrm{m}}[\cdot, j]\}[m], \qquad m = 0, \dots, M-1$$

次に，分析窓と同じ合成窓 $w$ を掛けて重畳加算し，窓の 2 乗の和で正規化する．$w[m]$ と $y_j[m]$ は $0 \le m \le M-1$ の外では 0 とする．

$$\hat x[n] = \frac{\sum_{j=0}^{J-1} w[n - t_j]\, y_j[n - t_j]}{\sum_{j=0}^{J-1} w[n - t_j]^2}, \qquad n = 0, \dots, N-1$$

この式が定まるには，すべての $n$ で分母が正である必要がある．

$$\sum_{j=0}^{J-1} w[n - t_j]^2 > 0, \qquad n = 0, \dots, N-1$$

たとえば周期ハン窓は $w[0] = 0$ なので，$H = M$ ではフレームの境目のサンプルで分母が 0 になる．周期ハン窓でも $H < M$ なら，この条件は満たされる．ハミング窓と矩形窓は $w[m] > 0$ なので，常に満たされる．

再構成は次の性質をもつ．

- リフタをかけない（$\bm{\ell} \equiv 1$）と，$\lvert\bm{X}^{\ell}\rvert = \max(\lvert\bm{X}\rvert, \varepsilon \lVert\bm{X}\rVert_\infty)$ なので，$\varepsilon$ の値によらず $\bm{G} = 1$，$\bm{Y} = \bm{X}$ となる．このとき $y_j[m] = w[m]\, x[t_j + m]$ なので，分子は分母と $x[n]$ の積になり，$\hat x = x$ が成り立つ（数値誤差は $10^{-15}$ 程度）．
- $\bm{G}$ は実数なので $\arg\bm{Y} = \arg\bm{X}$ で，変わるのは振幅だけである．
- ゲインは振幅の比，すなわちパワーの比の平方根で，出力の振幅は $\lvert\bm{Y}\rvert = \lvert\bm{X}^{\ell}\rvert$ になる．パワーの比をそのまま掛けると，変化量が dB で 2 倍になる．
- 逆 DFT は $M$ 点で巡回するので，ゲインが長い応答をもつと，$M$ 点を超える成分は折り返す．低域のリフタのようにゲインが滑らかなら，影響は小さい．
- 重畳加算で隣のフレームと混ざるので，再構成した信号をもう一度フレームに分けて分析すると，振幅は $\lvert\bm{X}^{\ell}\rvert$ に近づくが一致はしない．

低域成分 $\hat x_\mathrm{low}$ は，$\bm{Z}_\mathrm{low}$ から作ったゲインによる再構成で，包絡の振幅と元の位相をもつ．
高域成分は時間領域の差 $\hat x_\mathrm{high} = x - \hat x_\mathrm{low}$ とし，$\hat x_\mathrm{low} + \hat x_\mathrm{high} = x$ が成り立つ．

## 9. 入力の検査

| 検出 | 定義 |
|---|---|
| 無音・定数 | モノラル信号の最大値と最小値の差が 0（全ゼロを含む）．スペクトルでは全要素が 0 |
| クリッピング | 信号の 0 でない最大値または最小値から $10^{-9}\max\lvert x\rvert$ 以内の値が，3 サンプル以上連続する．正負は別々に，多チャネルはチャネルごとに判定する |
| 不正な入力 | NaN・inf を含む，複素数の時間信号，空の信号，長さ $N$ の 1 次元またはチャネル数 × $N$ の 2 次元以外の形状，$f_s$ が有限の正数でない，$H \notin [1, M]$，$M < 1$ など |

ケプストグラムでは，これらの検査を信号全体 $x$ に対して 1 回だけ行う．途中の無音フレームは正常として扱う．

## 付録 余弦変換による表現

$\bm{s}^{\mathrm{m}}$ が偶対称で $\bm{z}^{\mathrm{m}}$ が実数なので，$\mathcal{C}$ と $\mathcal{C}^{-1}$ は余弦変換として直接書ける．
$U = \lceil M/2 \rceil - 1$ として，次が成り立つ．最後の項は $M$ が偶数のときだけ現れる．

$$\bm{z}[l] = \frac{1}{M}\Bigl(\bm{s}[0] + 2\sum_{i=1}^{U}\bm{s}[i]\cos\frac{2\pi i l}{M} + (-1)^{l}\,\bm{s}[M/2]\Bigr), \qquad l = 0, \dots, L-1$$

$$\bm{s}[i] = \bm{z}[0] + 2\sum_{l=1}^{U}\bm{z}[l]\cos\frac{2\pi i l}{M} + (-1)^{i}\,\bm{z}[M/2], \qquad i = 0, \dots, I-1$$

特に $\bm{z}[0] = \frac1M\bigl(\bm{s}[0] + 2\sum_{i=1}^{U}\bm{s}[i] + \bm{s}[M/2]\bigr)$ である．
