# Mathematical Formulation of Cepstrum Analysis

[日本語](definitions.md) | English

## 0. Notation

| Symbol | Meaning |
|---|---|
| $n = 0, \cdots, N-1$ | sample bin |
| $i = 0, \cdots, I-1$ | frequency bin |
| $l = 0, \cdots, L-1$ | quefrency bin |
| $j = 0, \cdots, J-1$ | time bin |
| $f_s$ | sampling frequency |
| $M$ | FFT length, window length |
| $H$ | STFT shift length |
| $\mathrm{j}$ | imaginary unit |
| $\varepsilon$ | floor of the log and the gain (e.g. $10^{-12}$) |
| $w[n]$ | window function |
| $x[n] \in \mathbb{R}^N$ | time signal |
| $\bm{x} \in \mathbb{C}^I$ | spectrum |
| $\lvert \bm{x} \rvert \in \mathbb{R}^I$ | amplitude spectrum |
| $\lvert \bm{x} \rvert^{.2} \in \mathbb{R}^I$ | power spectrum |
| $\log \lvert \bm{x} \rvert \in \mathbb{R}^I$ | log-amplitude spectrum |
| $\bm{z} \in \mathbb{R}^L$ | real cepstrum |
| $\mathcal{C}$ | cepstrum transform |
| $\mathcal{C}^{-1}$ | inverse cepstrum transform |
| $\bm{\ell} \in [0,1]^L$ | lifter weights |
| $\lambda$ | lifter index |
| $\bm{z}_\mathrm{low}$ | lowpass-liftered cepstrum (low-quefrency part) |
| $\bm{z}_\mathrm{high}$ | remaining high-quefrency part $\bm{z} - \bm{z}_\mathrm{low}$ |
| $\bm{X} \in \mathbb{C}^{I \times J}$ | complex spectrogram |
| $\lvert\bm{X}\rvert \in \mathbb{R}^{I \times J}$ | amplitude spectrogram |
| $\lvert\bm{X}\rvert^{.2} \in \mathbb{R}^{I \times J}$ | power spectrogram |
| $\bm{Z} \in \mathbb{R}^{L \times J}$ | real cepstrogram |
| $\bm{Z}_\mathrm{low}$ | lowpass-liftered real cepstrogram (low-quefrency part) |
| $\bm{Z}_\mathrm{high}$ | remaining high-quefrency part $\bm{Z} - \bm{Z}_\mathrm{low}$ |

- The time signal is a single-channel signal whose length has been prepared for the FFT or STFT.
- $I$ is the number of frequency bins of the one-sided spectrum: $I = \lfloor M/2 \rfloor + 1$.
- The cepstrum is computed from the $M$ points obtained by joining the one-sided spectrum with its mirror, and then the one-sided part is taken. Hence $L = I$.
- Frequency bin $i$ corresponds to $f_s \frac{i}{M}$ [Hz] ($i = 0$ is DC).
- $\lvert \bm{A} \rvert$ is the element-wise absolute value of a matrix $\bm{A}$, and $\lvert \bm{A} \rvert^{.2}$ is the element-wise square.
- $\log \lvert \bm{A} \rvert$ is the element-wise natural log of the absolute value, floored at $\varepsilon \lVert \bm{A} \rVert_\infty$.
- $\lVert \bm{A} \rVert_\infty$ is the maximum absolute value of the elements. For a matrix it is taken per column.
- $\mathcal{C}$ is the transform from the log-amplitude spectrum to the real cepstrum, written $\bm{z} = \mathcal{C}\{\log \lvert \bm{x} \rvert\}$. $\mathcal{C}^{-1}$ is its inverse.
- $\bm{z}_\mathrm{low}$ is the cepstrum after a lowpass lifter, and $\bm{z}_\mathrm{high}$ is the rest. $\bm{Z}_\mathrm{low}$ and $\bm{Z}_\mathrm{high}$ collect $\bm{z}_\mathrm{low}$ and $\bm{z}_\mathrm{high}$ of every frame.

## 1. Time signal and window

### 1.1 Conversion to mono

A multichannel signal $x_\mathrm{ch}[n]$ ($\mathrm{ch} = 1, \dots, C$) is averaged over channels.

$$x[n] = \frac{1}{C} \sum_{\mathrm{ch}=1}^{C} x_\mathrm{ch}[n]$$

### 1.2 Window

The window $w[n]$ ($n = 0, \dots, M-1$) is a real sequence of length $M$ applied to the DFT interval.
It is one of the following or any real sequence of length $M$.

| Window | $w[n]$ ($n = 0, \dots, M-1$) |
|---|---|
| Hann | $\dfrac12 - \dfrac12\cos\dfrac{2\pi n}{M}$ |
| Hamming | $0.54 - 0.46\cos\dfrac{2\pi n}{M}$ |
| Rectangular | $1$ |

All of them are periodic windows: the denominator in the cosine is $M$.

### 1.3 Windowing and DFT

When a single DFT is applied to the whole signal, $M = N$. The signal $x[n]$ is windowed and transformed, and the $I$ non-negative-frequency points form the spectrum $\bm{x}$.

$$\bm{x}[i] = \sum_{n=0}^{M-1} w[n]\, x[n]\, e^{-\mathrm{j} 2\pi i n / M}, \qquad i = 0, \dots, I-1$$

Below, "spectrum" refers to this $\bm{x}$. In the STFT, the same formula is applied to each frame.
For variables in the frequency and time-frequency domains, the index $[i]$ is omitted unless it is needed in a formula.

## 2. Spectrum input

A spectrum is given as complex, amplitude or power values. From the values at the non-negative frequencies $i = 0, \dots, I-1$, $\bm{x}$ is defined as follows.

- Complex: $\bm{x}$ is used as given. Only the magnitude $\lvert\bm{x}\rvert$ is used for the cepstrum.
- Amplitude: $\lvert\bm{x}\rvert$ is given. The phase is 0.
- Power: $\lvert\bm{x}\rvert^{.2}$ is given. The magnitude is the element-wise square root $\lvert\bm{x}\rvert = \sqrt{\lvert\bm{x}\rvert^{.2}}$. The phase is 0.

A one-sided input ($I$ points) is used as $\bm{x}$ directly; for a two-sided input ($M$ points) the first $I$ points are $\bm{x}$. No correction is applied to the values.

## 3. Log-amplitude spectrum

$$\log\lvert\bm{x}\rvert = \ln \max\bigl(\lvert\bm{x}\rvert,\ \varepsilon \lVert\bm{x}\rVert_\infty\bigr)$$

$\ln$ is the natural log. At frequencies where the magnitude is below $\varepsilon$ times the maximum, it is replaced by that value to avoid $\ln 0$.

For dB, the natural log is converted:

$$\bm{s}_\mathrm{dB} = \frac{20}{\ln 10}\,\log\lvert\bm{x}\rvert \;=\; 20\log_{10}\max(\cdots) \qquad \left(\frac{20}{\ln 10} \approx 8.686\right)$$

Below, $\bm{s} = \log\lvert\bm{x}\rvert$. On the dB scale, $\bm{s}$ is replaced by $\bm{s}_\mathrm{dB}$.

## 4. Real cepstrum

The real cepstrum is obtained by joining the one-sided log-amplitude spectrum $\bm{s}$ ($I$ points) with its mirror, taking an $M$-point inverse DFT, and taking the one-sided part.

(1) Mirroring. With $U = \lceil M/2 \rceil - 1$, form the $M$-point sequence $\bm{s}^{\mathrm{m}}$ by joining $\bm{s}$ with its mirror on the negative-frequency side.

$$\bm{s}^{\mathrm{m}}[i] = \bm{s}[i]\ (i = 0, \dots, I-1), \qquad \bm{s}^{\mathrm{m}}[M-i] = \bm{s}[i]\ (i = 1, \dots, U)$$

$\bm{s}^{\mathrm{m}}$ is real and even ($\bm{s}^{\mathrm{m}}[M-i] = \bm{s}^{\mathrm{m}}[i]$).

(2) DFT and inverse DFT. For sequences $\bm{a}$, $\bm{b}$ of length $M$:

$$\mathcal{F}\{\bm{a}\}[k] = \sum_{n=0}^{M-1}\bm{a}[n]\,e^{-\mathrm{j}2\pi kn/M}, \qquad
\mathcal{F}^{-1}\{\bm{b}\}[n] = \frac1M\sum_{k=0}^{M-1}\bm{b}[k]\,e^{\mathrm{j}2\pi kn/M}$$

(3) Inverse DFT. Take the inverse DFT of the mirrored log-amplitude spectrum.

$$\bm{z}^{\mathrm{m}} = \mathcal{F}^{-1}\{\bm{s}^{\mathrm{m}}\}$$

Since $\bm{s}^{\mathrm{m}}$ is real and even, $\bm{z}^{\mathrm{m}}$ is also real and even ($\bm{z}^{\mathrm{m}}[M-l] = \bm{z}^{\mathrm{m}}[l]$).

(4) One-sided part. The points $l = 0, \dots, L-1$ of $\bm{z}^{\mathrm{m}}$ form the real cepstrum $\bm{z}$ ($\bm{z}[l] = \bm{z}^{\mathrm{m}}[l]$).

Steps (1)–(4) together are written $\mathcal{C}$: $\bm{z} = \mathcal{C}\{\bm{s}\}$.

The inverse $\mathcal{C}^{-1}$ mirrors $\bm{z}$ into $\bm{z}^{\mathrm{m}}$ ($\bm{z}^{\mathrm{m}}[M-l] = \bm{z}[l]$, $l = 1, \dots, U$), takes the DFT, and keeps the first $I$ points of the real part.

$$\bm{s} = \mathcal{C}^{-1}\{\bm{z}\} = \bigl(\mathrm{Re}\,\mathcal{F}\{\bm{z}^{\mathrm{m}}\}\bigr)[0, \dots, I-1]$$

On the dB scale, $\bm{z}_\mathrm{dB} = \mathcal{C}\{\bm{s}_\mathrm{dB}\} = \frac{20}{\ln 10}\bm{z}$.
The form written directly as a sum of cosines is given in the appendix.

The real cepstrum has the following properties.

- $\bm{z}[0] = \frac1M\sum_{i=0}^{M-1}\bm{s}^{\mathrm{m}}[i]$ is the mean log amplitude over all frequencies and represents the overall level.
- $\mathcal{C}^{-1}\{\mathcal{C}\{\bm{s}\}\} = \bm{s}$. Without a lifter the original log spectrum is recovered, and $\exp$ gives back the magnitude $\lvert\bm{x}\rvert$.
- $\bm{z}$ contains no phase information.

## 5. Quefrency and units

$$q_l = \frac{l}{f_s}\ [\mathrm{s}], \qquad l = 0, \dots, L-1$$

$\bm{z}[l]$ is the coefficient of the cosine $\cos(2\pi q_l f_i) = \cos(2\pi i l / M)$ in the log spectrum at frequency $f_i = f_s\, i / M$.
The period of this ripple (peak-to-peak width along the frequency axis) and its density are:

$$T_l = \frac{1}{q_l} = \frac{f_s}{l}\ [\mathrm{Hz}]\ (l \ge 1), \qquad \text{density} = q_l\ [\text{peaks/Hz}] = q_l\ [\mathrm{s}]$$

Hence $[\mathrm{s}] = [\mathrm{Hz}^{-1}]$. By the SI prefixes, $1\ \mathrm{kHz}^{-1} = 1\ \mathrm{ms}$, so the numbers are the same as in ms.
For example, $q = 8\ \mathrm{kHz}^{-1}$ means 8 peaks per $1\ \mathrm{kHz}$, that is, a ripple with a period of $125\ \mathrm{Hz}$.

The component at $l = 0$ has an infinite period and corresponds to the overall level.
The largest quefrency is $q_{L-1} = (L-1)/f_s = \lfloor M/2\rfloor / f_s$. For even $M$, its period $2 f_s / M$ is twice the frequency resolution.

## 6. Lifters

A lifter is a weight $\bm{\ell}[l] \in [0, 1]$ ($l = 0, \dots, L-1$) applied to each bin, given as a function of the quefrency $q_l$.
Liftering is an element-wise product:

$$\bm{z}^{\ell}[l] = \bm{\ell}[l]\,\bm{z}[l]$$

| Lifter | Weight $\bm{\ell}[l]$ (function of $q_l \ge 0$) |
|---|---|
| Lowpass (boundary $c$) | $1$ for $q_l \le c$, $0$ otherwise. With $\lambda = \lfloor c f_s \rfloor$, it keeps the components with $l \le \lambda$ |
| Highpass (boundary $c$) | $1$ for $q_l > c$, $0$ otherwise. With $\lambda = \lfloor c f_s \rfloor$, it keeps the components with $l > \lambda$. The weight of $\bm{z}[0]$ is always 0 |
| Bandpass (boundaries $a, b$) | $1$ for $a < q_l \le b$, $0$ otherwise. Equal to the product of a highpass with boundary $a$ and a lowpass with boundary $b$ |
| Notch (center $q_0$, width $W$) | $0$ for $\lvert q_l - q_0\rvert \le W/2$, $1$ otherwise |

Boundaries are given as a quefrency [s] or a period [Hz] ($q = 1/T$).
When the weights are applied to the $M$-point $\bm{z}^{\mathrm{m}}$, they are a function of $\lvert q\rvert$, so the mirror side gets the same values. The result equals applying $\bm{\ell}$ to the one-sided part.

## 7. Separation into low / high quefrency parts

With the weights $\bm{\ell}_\mathrm{low}$ of a lowpass lifter, the cepstrum is split into two parts.

$$\bm{z}_\mathrm{low} = \bm{\ell}_\mathrm{low}\odot\bm{z}, \qquad \bm{z}_\mathrm{high} = \bm{z} - \bm{z}_\mathrm{low}$$

For a rectangular lowpass lifter whose lifter index $\lambda$ is the boundary bin, $\bm{\ell}_\mathrm{low}[l]$ is 1 for $l \le \lambda$ and 0 for $l > \lambda$.
Then $\bm{z}_\mathrm{low}$ keeps the components with $l \le \lambda$, and $\bm{z}_\mathrm{high}$ those with $l > \lambda$.
For any weights, including tapered lifters, $\bm{z}_\mathrm{low} + \bm{z}_\mathrm{high} = \bm{z}$ holds exactly.
The corresponding log spectra are obtained by applying $\mathcal{C}^{-1}$ to each part.

$$\bm{s}_\mathrm{low} = \mathcal{C}^{-1}\{\bm{z}_\mathrm{low}\}, \qquad
\bm{s}_\mathrm{high} = \mathcal{C}^{-1}\{\bm{z}_\mathrm{high}\}, \qquad \bm{s} = \bm{s}_\mathrm{low} + \bm{s}_\mathrm{high}$$

- $\bm{s}_\mathrm{low}$ is the smooth part and represents the spectral envelope. It contains $\bm{z}[0]$, so it also carries the overall level.
- $\bm{s}_\mathrm{high}$ is the detailed part and represents the fine structure. In speech it corresponds to the harmonic peaks and valleys caused by the pitch. Its mean is 0.

Suppose the amplitude spectrum of speech is a product of the vocal-tract characteristic $P$ and the vocal-fold vibration characteristic $Q$: $\lvert\bm{x}\rvert = \lvert P\rvert\,\lvert Q\rvert$.
Then $\bm{s} = \ln\lvert P\rvert + \ln\lvert Q\rvert$ is a sum, and $\ln\lvert P\rvert$, which varies slowly along frequency, goes to the low-quefrency side, while the periodic $\ln\lvert Q\rvert$ goes to the high-quefrency side.

To express $\bm{s}_\mathrm{low}$ and $\bm{s}_\mathrm{high}$ in dB, multiply by $\frac{20}{\ln 10}$.

## 8. Cepstrogram and reconstruction of time signals

### 8.1 Cepstrogram

The spectrogram $\bm{X}$ is the STFT of $x[n]$ with window $w$, frame length $M$ and shift $H$; column $j$ is the spectrum of frame $j$.
$x[n]$ is 0 outside $0 \le n \le N-1$. The start $t_j$ of frame $j$ and the number of frames $J$ are:

$$t_j = jH - (M - H), \qquad J = \left\lceil \frac{N + M - H}{H} \right\rceil$$

The first frame starts $M - H$ points before the signal, and the last frame contains the end of the signal.
If $H$ divides $M$, every sample $0 \le n \le N-1$ is contained in exactly $M/H$ frames.

$$\bm{X}[i, j] = \sum_{m=0}^{M-1} w[m]\, x[t_j + m]\, e^{-\mathrm{j} 2\pi i m/M}, \qquad i = 0, \dots, I-1,\ \ j = 0, \dots, J-1$$

Column $j$ of the cepstrogram $\bm{Z}$ is $\mathcal{C}$ applied to the log-amplitude spectrum of each frame.

$$\bm{Z}[\cdot, j] = \mathcal{C}\{\log\lvert\bm{X}[\cdot, j]\rvert\}$$

### 8.2 Reconstruction

The log amplitude of each frame is obtained from the liftered cepstrogram.

$$\log\lvert\bm{X}^{\ell}\rvert\,[\cdot, j] = \mathcal{C}^{-1}\{\bm{\ell}\odot\bm{Z}[\cdot, j]\}$$

The ratio to the original magnitude is the gain $\bm{G}$, which multiplies the original spectrogram. The denominator is floored at $\varepsilon \lVert\bm{X}\rVert_\infty$, the same floor as the log.

$$\bm{G} = \frac{\lvert\bm{X}^{\ell}\rvert}{\max\bigl(\lvert\bm{X}\rvert,\ \varepsilon \lVert\bm{X}\rVert_\infty\bigr)}, \qquad \bm{Y} = \bm{G} \odot \bm{X}$$

The time signal $\hat x[n]$ is the inverse STFT of $\bm{Y}$.
First, each frame $\bm{Y}[\cdot, j]$ is joined with its conjugate-symmetric mirror ($\bm{Y}^{\mathrm{m}}[M-i, j] = \overline{\bm{Y}[i, j]}$, $i = 1, \dots, U$) to make $M$ points, and converted back to a real signal $y_j$ by the inverse DFT.

$$y_j[m] = \mathcal{F}^{-1}\{\bm{Y}^{\mathrm{m}}[\cdot, j]\}[m], \qquad m = 0, \dots, M-1$$

Then the frames are multiplied by a synthesis window equal to the analysis window $w$, overlap-added, and normalized by the sum of squared windows. $w[m]$ and $y_j[m]$ are 0 outside $0 \le m \le M-1$.

$$\hat x[n] = \frac{\sum_{j=0}^{J-1} w[n - t_j]\, y_j[n - t_j]}{\sum_{j=0}^{J-1} w[n - t_j]^2}, \qquad n = 0, \dots, N-1$$

For this to be defined, the denominator must be positive for every $n$:

$$\sum_{j=0}^{J-1} w[n - t_j]^2 > 0, \qquad n = 0, \dots, N-1$$

For example, the periodic Hann window has $w[0] = 0$, so with $H = M$ the denominator is 0 at the frame boundaries. For the periodic Hann window, $H < M$ satisfies the condition. The Hamming and rectangular windows have $w[m] > 0$, so they always satisfy it.

Reconstruction has the following properties.

- Without a lifter ($\bm{\ell} \equiv 1$), $\lvert\bm{X}^{\ell}\rvert = \max(\lvert\bm{X}\rvert, \varepsilon \lVert\bm{X}\rVert_\infty)$, so $\bm{G} = 1$ and $\bm{Y} = \bm{X}$ for any $\varepsilon$. Then $y_j[m] = w[m]\, x[t_j + m]$, so the numerator equals the denominator times $x[n]$, and $\hat x = x$ (numerical error about $10^{-15}$).
- $\bm{G}$ is real, so $\arg\bm{Y} = \arg\bm{X}$; only the magnitude changes.
- The gain is the amplitude ratio, i.e. the square root of the power ratio, so the output magnitude is $\lvert\bm{Y}\rvert = \lvert\bm{X}^{\ell}\rvert$. Applying the power ratio directly would double the change in dB.
- The inverse DFT is circular over $M$ points, so if the gain has a long response, components beyond $M$ points wrap around. The effect is small for smooth gains, such as those of lowpass lifters.
- Overlap-add mixes neighboring frames, so when the reconstructed signal is split into frames and analyzed again, its magnitude moves toward $\lvert\bm{X}^{\ell}\rvert$ but does not match it.

The low part $\hat x_\mathrm{low}$ is the reconstruction with the gain made from $\bm{Z}_\mathrm{low}$; it has the envelope magnitude and the original phase.
The high part is the time-domain difference $\hat x_\mathrm{high} = x - \hat x_\mathrm{low}$, so $\hat x_\mathrm{low} + \hat x_\mathrm{high} = x$ holds.

## 9. Input validation

| Check | Definition |
|---|---|
| Silence / constant | The difference between the maximum and minimum of the mono signal is 0 (including all zeros). For a spectrum, all elements are 0 |
| Clipping | Values within $10^{-9}\max\lvert x\rvert$ of the signal's non-zero maximum or minimum continue for 3 or more samples. Positive and negative sides are checked separately, and multichannel signals per channel |
| Invalid input | NaN or inf, complex time signal, empty signal, shape other than a 1-D signal of length $N$ or a 2-D signal of channels × $N$, $f_s$ not a finite positive number, $H \notin [1, M]$, $M < 1$, etc. |

For a cepstrogram, these checks run once on the whole signal $x$. Silent frames inside the signal are treated as normal.

## Appendix: Cosine-transform form

Since $\bm{s}^{\mathrm{m}}$ is even and $\bm{z}^{\mathrm{m}}$ is real, $\mathcal{C}$ and $\mathcal{C}^{-1}$ can be written directly as cosine transforms.
With $U = \lceil M/2 \rceil - 1$, the following holds. The last term appears only for even $M$.

$$\bm{z}[l] = \frac{1}{M}\Bigl(\bm{s}[0] + 2\sum_{i=1}^{U}\bm{s}[i]\cos\frac{2\pi i l}{M} + (-1)^{l}\,\bm{s}[M/2]\Bigr), \qquad l = 0, \dots, L-1$$

$$\bm{s}[i] = \bm{z}[0] + 2\sum_{l=1}^{U}\bm{z}[l]\cos\frac{2\pi i l}{M} + (-1)^{i}\,\bm{z}[M/2], \qquad i = 0, \dots, I-1$$

In particular, $\bm{z}[0] = \frac1M\bigl(\bm{s}[0] + 2\sum_{i=1}^{U}\bm{s}[i] + \bm{s}[M/2]\bigr)$.
