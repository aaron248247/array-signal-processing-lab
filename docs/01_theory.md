# 1. Theory

This document collects the models and derivations used by the code. Equations carry labels of the form (A.1) so that the other documents and the source code can refer to them. Bracketed numbers point to [06_references.md](06_references.md).

Contents

- [A. Narrowband uniform linear array](#a-narrowband-uniform-linear-array)
- [B. Covariance, conventional and MVDR beamforming](#b-covariance-conventional-and-mvdr-beamforming)
- [C. Finite-sample effects, diagonal loading and mismatch](#c-finite-sample-effects-diagonal-loading-and-mismatch)
- [D. Pulsed radar: delay, Doppler, matched filtering, range–Doppler processing](#d-pulsed-radar-delay-doppler-matched-filtering-rangedoppler-processing)
- [E. Detection in Gaussian noise and CA-CFAR](#e-detection-in-gaussian-noise-and-ca-cfar)

---

## A. Narrowband uniform linear array

### A.1 Geometry and steering vector

Symbols: $N$ number of elements; $d$ element spacing; $\lambda$ carrier wavelength; $k = 2\pi/\lambda$; $\theta$ angle of arrival measured from broadside (the array normal), positive toward $+x$.

Elements sit at $x_n = n d$, $n = 0,\dots,N-1$. A far-field plane wave from $\theta$ has wavevector component $k\sin\theta$ along the array, so, relative to element 0, element $n$ sees the phase

$$
a_n(\theta) = e^{\,j k x_n \sin\theta} = e^{\,j n \psi_0(\theta)}, \qquad \psi_0(\theta) = k d \sin\theta. \tag{A.1}
$$

$\mathbf a(\theta) = [a_0,\dots,a_{N-1}]^T$ is the steering vector [1, 2]. It is not normalized: $\mathbf a^H\mathbf a = N$. The sign convention ($e^{+j\dots}$) is used everywhere in the code; changing it only conjugates every pattern.

Assumptions: narrowband (the signal envelope does not change over the aperture transit time $(N-1)d|\sin\theta|/c$); isotropic, identical, uncoupled elements; far field. Two consequences used in the tests: $\mathbf a(-\theta) = \mathbf a(\theta)^*$, and $\mathbf a(0) = \mathbf 1$.

### A.2 Array factor

A beamformer forms $y = \mathbf w^H \mathbf x$. Its response to a unit plane wave is $\mathbf w^H\mathbf a(\theta)$ (the *numerical* array factor in the code). For delay-and-sum (conventional) weights steered to $\theta_0$ [1, 2],

$$
\mathbf w_c = \frac{\mathbf a(\theta_0)}{N}, \tag{A.2}
$$

the response is a geometric series in $e^{j\psi}$ with $\psi = kd(\sin\theta - \sin\theta_0)$:

$$
AF(\theta) = \frac1N\sum_{n=0}^{N-1} e^{jn\psi}
= e^{\,j(N-1)\psi/2}\,\frac{\sin(N\psi/2)}{N\sin(\psi/2)}. \tag{A.3}
$$

The phase factor comes from referencing the phase to element 0. Properties used for verification:

* Main lobe at $\psi = 0$, i.e. $\theta = \theta_0$, with $AF(\theta_0) = 1$.
* Nulls where $N\psi/2 = m\pi$, $m \not\equiv 0 \pmod N$:
  $$\sin\theta_m = \sin\theta_0 + m\frac{\lambda}{Nd}. \tag{A.4}$$
* Grating lobes where $\psi = 2\pi m$, $m\neq0$; for $d = \lambda/2$ they stay out of visible space for all scan angles (one sits exactly at endfire when $\theta_0 = \pm 90^\circ$).
* First-null beamwidth $\arcsin(\sin\theta_0 + \lambda/Nd) - \arcsin(\sin\theta_0 - \lambda/Nd)$; half-power beamwidth $\approx 0.886\,\lambda/(Nd\cos\theta_0)$ for large $N$ [1].
* As $N\to\infty$ (A.3) tends to a sinc in $N\psi/2$. Its first sidelobe sits at the root of $\tan x = x$ near $x = 1.43\pi$, giving $-13.26$ dB. Finite $N$ gives a slightly higher (less negative) peak sidelobe.

### A.3 Bartlett (conventional) angle scan

For snapshots $\mathbf x_k$ the conventional spatial spectrum is

$$
P(\theta) = \frac1K\sum_k |\mathbf w_c(\theta)^H\mathbf x_k|^2 = \mathbf w_c^H \hat{\mathbf R}\,\mathbf w_c . \tag{A.5}
$$

For one noiseless plane wave of power $p$ from $\theta_s$, $P(\theta) = p\,|AF(\theta;\theta_s)|^2$. This gives an independent check of (A.3) by a different code path.

![Figure 1 — ULA array factor, closed form vs numerical](../figures/fig01_ula_verification.png)

*Figure 1.* Array factor of the conventional weights (A.2) for $N = 16$, $d=\lambda/2$, steered to 0° and 30°. Panel (a) shows the main lobe at $\theta_0$, the nulls at the angles predicted by (A.4) (triangles), and the sidelobe structure of (A.3). Panel (b) shows that the numerical response $\mathbf w^H\mathbf a(\theta)$ agrees with the closed form (A.3) to round-off. Panel (c) shows the null-location error. The beam-scan check of (A.5) is reported in [03 §3.1](03_verification_validation.md#31-ula-model-exact). Full caption: [Figure 1](07_figure_index.md#figure-1--ula-analytical-verification).

---

## B. Covariance, conventional and MVDR beamforming

### B.1 Data model

$$
\mathbf x_k = \sum_i s_i[k]\,\mathbf a(\theta_i) + \mathbf n_k, \qquad
s_i[k]\sim\mathcal{CN}(0,p_i),\quad \mathbf n_k\sim\mathcal{CN}(\mathbf 0,\sigma^2\mathbf I), \tag{B.1}
$$

with all sources mutually independent and independent from snapshot to snapshot. The ensemble covariance is

$$
\mathbf R = E[\mathbf x\mathbf x^H] = \sum_i p_i\,\mathbf a_i\mathbf a_i^H + \sigma^2\mathbf I . \tag{B.2}
$$

$\mathbf R$ is Hermitian and positive definite for $\sigma^2 > 0$. Its $N-r$ smallest eigenvalues equal $\sigma^2$ when $r$ sources are present. The interference-plus-noise covariance $\mathbf R_{in}$ is (B.2) without the target term. Per-element ratios are written $\mathrm{SNR} = p_s/\sigma^2$ and $\mathrm{JNR} = p_j/\sigma^2$.

The sample covariance is $\hat{\mathbf R} = \frac1K\sum_k \mathbf x_k\mathbf x_k^H$. Its rank is at most $\min(K,N)$, so it is singular when $K<N$.

### B.2 Output SINR

For weights $\mathbf w$ and a target with steering vector $\mathbf a_s$:

$$
\mathrm{SINR}(\mathbf w) = \frac{p_s|\mathbf w^H\mathbf a_s|^2}{\mathbf w^H\mathbf R_{in}\mathbf w}. \tag{B.3}
$$

By the Cauchy–Schwarz inequality in the $\mathbf R_{in}$ inner product this is at most [1]

$$
\mathrm{SINR}_{\rm opt} = p_s\,\mathbf a_s^H\mathbf R_{in}^{-1}\mathbf a_s, \tag{B.4}
$$

with equality for $\mathbf w \propto \mathbf R_{in}^{-1}\mathbf a_s$. Throughout, the SINR of *estimated* weights is evaluated with the *true* $\mathbf R_{in}$. This is the standard figure of merit in [4].

### B.3 Conventional beamformer, single jammer

Let $\mathbf R_{in} = \sigma^2(\mathbf I + \mathrm{JNR}\,\mathbf a_j\mathbf a_j^H)$ and $\beta = \mathbf a_s^H\mathbf a_j$. For $\mathbf w_c = \mathbf a_s/N$, $|\mathbf w_c^H\mathbf a_s|^2 = 1$ and $\mathbf w_c^H\mathbf R_{in}\mathbf w_c = \sigma^2(1/N + \mathrm{JNR}|\beta|^2/N^2)$, so

$$
\mathrm{SINR}_c = \mathrm{SNR}\,\frac{N^2}{N + \mathrm{JNR}\,|\beta|^2}. \tag{B.5}
$$

$|\beta|/N = |AF(\theta_j;\theta_s)|$ is the conventional sidelobe level toward the jammer. A jammer placed exactly on a pattern null ($\beta = 0$) has no effect. The jammer at $30^\circ$ that was first used in the tests is an example for $N=16$, $d=\lambda/2$; it was moved to $35^\circ$ for that reason.

### B.4 MVDR / Capon beamformer

Minimize output power subject to unit gain in the look direction [3]:

$$
\min_{\mathbf w}\ \mathbf w^H\mathbf R\mathbf w \quad\text{s.t.}\quad \mathbf w^H\mathbf a_0 = 1
\ \ \Longrightarrow\ \
\mathbf w_{\rm MVDR} = \frac{\mathbf R^{-1}\mathbf a_0}{\mathbf a_0^H\mathbf R^{-1}\mathbf a_0}. \tag{B.6}
$$

(Lagrangian $\mathbf w^H\mathbf R\mathbf w - \mu(\mathbf w^H\mathbf a_0 - 1) - \mu^*(\mathbf a_0^H\mathbf w - 1)$; setting the gradient to zero gives $\mathbf w = \mu\mathbf R^{-1}\mathbf a_0$, and the constraint fixes $\mu$.) Limiting behavior checked in the tests:

* $\mathbf R = c\mathbf I$ gives $\mathbf w_{\rm MVDR} = \mathbf a_0/N = \mathbf w_c$.
* With $\mathbf R = \mathbf R_{in}$ and $\mathbf a_0 = \mathbf a_s$, (B.6) is proportional to the optimum weight, so $\mathrm{SINR} = \mathrm{SINR}_{\rm opt}$.
* With the *full* covariance $\mathbf R = \mathbf R_{in} + p_s\mathbf a_s\mathbf a_s^H$ and no mismatch, the matrix inversion lemma shows $\mathbf R^{-1}\mathbf a_s \propto \mathbf R_{in}^{-1}\mathbf a_s$, so the clairvoyant weights are the same. This no longer holds when the steering vector is mismatched (C.4) or when $\mathbf R$ is estimated (C.2).

**Single jammer, closed form.** By the matrix inversion lemma,

$$
\mathbf R_{in}^{-1} = \frac{1}{\sigma^2}\Big(\mathbf I - \frac{\mathrm{JNR}\,\mathbf a_j\mathbf a_j^H}{1 + N\,\mathrm{JNR}}\Big),
$$

so

$$
\mathrm{SINR}_{\rm opt} = \mathrm{SNR}\Big(N - \frac{\mathrm{JNR}\,|\beta|^2}{1 + N\,\mathrm{JNR}}\Big), \tag{B.7}
$$

$$
|\mathbf w_{\rm MVDR}^H\mathbf a_j| = \frac{|\beta|}{\big(N - \mathrm{JNR}|\beta|^2/(1+N\,\mathrm{JNR})\big)(1+N\,\mathrm{JNR})}. \tag{B.8}
$$

(B.8) gives the null depth: the jammer response falls as $1/(N\,\mathrm{JNR})$ relative to the conventional sidelobe $|\beta|/N$. As the jammer approaches the main lobe, $|\beta|\to N$ and $\mathrm{SINR}_{\rm opt}\to \mathrm{SNR}\,N/(1+N\,\mathrm{JNR})$: no linear beamformer can separate a jammer that arrives from the target direction.

![Figure 2 — conventional vs MVDR beampatterns and SINR loss](../figures/fig02_conventional_vs_mvdr.png)

*Figure 2.* Conventional (A.2) and MVDR (B.6) patterns for two sidelobe jammers. The MVDR solution keeps unit gain at the look direction, as the constraint in (B.6) requires, and minimizes output power by placing nulls on the jammers; with the true $\mathbf R_{in}$ it reaches $\mathrm{SINR}_{\rm opt}$ (B.4). The null depths are idealized numerical values for an exactly known covariance, not hardware rejection figures. Panel (b) also compares sample-matrix estimates with (C.1) and (C.2). Full caption: [Figure 2](07_figure_index.md#figure-2--conventional-vs-mvdr-beampattern).

---

## C. Finite-sample effects, diagonal loading and mismatch

### C.1 Sample-matrix inversion with signal-free training (RMB)

When $\hat{\mathbf w} \propto \hat{\mathbf R}^{-1}\mathbf a_s$ is computed from $K\ge N$ i.i.d. complex Gaussian snapshots that do **not** contain the target, the normalized SINR

$$
\rho = \frac{\mathrm{SINR}(\hat{\mathbf w})}{\mathrm{SINR}_{\rm opt}} \sim \mathrm{Beta}(K-N+2,\ N-1),\qquad
E[\rho] = \frac{K-N+2}{K+1}, \tag{C.1}
$$

independent of $\mathbf R_{in}$ [4]. So $K\approx 2N-3$ snapshots give an average loss of 3 dB [4]. The distribution does not depend on JNR, which is checked in Figure 6(c).

### C.2 Target present in the training data

The RMB result applies to whatever covariance the training data come from. If the data contain the target, that covariance is $\mathbf R = \mathbf R_{in} + p_s\mathbf a_s\mathbf a_s^H$, and

$$
\rho_R = \frac{|\hat{\mathbf w}^H\mathbf a_s|^2}{(\hat{\mathbf w}^H\mathbf R\hat{\mathbf w})(\mathbf a_s^H\mathbf R^{-1}\mathbf a_s)} \sim \mathrm{Beta}(K-N+2,N-1).
$$

Write $S = \mathrm{SINR}_{\rm opt}$. The matrix inversion lemma gives $p_s\mathbf a_s^H\mathbf R^{-1}\mathbf a_s = S/(1+S)$. Let $x = p_s|\hat{\mathbf w}^H\mathbf a_s|^2/(\hat{\mathbf w}^H\mathbf R\hat{\mathbf w}) = \rho_R\,S/(1+S)$. Since $\hat{\mathbf w}^H\mathbf R_{in}\hat{\mathbf w} = \hat{\mathbf w}^H\mathbf R\hat{\mathbf w} - p_s|\hat{\mathbf w}^H\mathbf a_s|^2$, the output SINR is $x/(1-x)$, and

$$
\frac{\mathrm{SINR}(\hat{\mathbf w})}{S} = \frac{\rho_R}{1 + S(1-\rho_R)}. \tag{C.2}
$$

This derivation was written for this project. The qualitative result, that a target in the training data raises the number of snapshots needed by roughly a factor of $S$, is well known (see [1], ch. 7). The mean of (C.2) is evaluated by quadrature over the Beta density and compared with Monte Carlo in Figures 2 and 3.

![Figure 3 — finite-snapshot SMI/MVDR performance](../figures/fig03_sinr_vs_snapshots.png)

*Figure 3.* Mean normalized SINR against the number of training snapshots. Signal-free training follows the Reed–Mallett–Brennan law (C.1) [4]; training that contains the target follows (C.2), which was derived for this project and requires many more snapshots. Panel (b) shows the condition number of $\hat{\mathbf R}$ against the Marchenko–Pastur lower-edge estimate discussed in §C.3 [18]. Full caption: [Figure 3](07_figure_index.md#figure-3--output-sinr-versus-number-of-snapshots).

### C.3 Diagonal loading

Diagonal loading replaces $\hat{\mathbf R}$ by $\hat{\mathbf R} + \delta\mathbf I$ in (B.6). With the eigen-decomposition $\hat{\mathbf R} = \sum_i \hat\lambda_i\hat{\mathbf u}_i\hat{\mathbf u}_i^H$,

$$
(\hat{\mathbf R}+\delta\mathbf I)^{-1}\mathbf a_0 = \sum_i \frac{\hat{\mathbf u}_i^H\mathbf a_0}{\hat\lambda_i + \delta}\,\hat{\mathbf u}_i ,
$$

so each eigen-direction is weighted by $1/(\hat\lambda_i+\delta)$. With finite $K$ the $N-r$ noise eigenvalues, which would all equal $\sigma^2$ for the true $\mathbf R$, are spread over an interval whose width scales as $\sqrt{N/K}$ (the Marchenko–Pastur law [18]). The resulting unequal weighting of the noise subspace distorts the quiescent pattern and causes the RMB loss. Loading with $\delta$ large compared with that spread equalizes the noise-subspace weights. Interference eigenvalues $\lambda_j \gg \delta$ are still weighted by nearly zero, so the jammers stay canceled [5].

Limits:

* $\delta\to 0$: unloaded SMI.
* $\delta \gg \lambda_{\max}$: $\mathbf w \to \mathbf a_0/N$ (conventional), so jammer suppression is lost.
* $K < N$: $\hat{\mathbf R}$ is singular, but $\hat{\mathbf R}+\delta\mathbf I$ is invertible. The beamformer then nulls the $K$-dimensional subspace spanned by the training data, which contains the interference subspace if $K \ge r$.

Loading also limits the depth of self-nulling under mismatch (C.4) [7]. No optimal $\delta$ is derived here; it is swept.

![Figure 4 — diagonal-loading sensitivity](../figures/fig04_loading_sensitivity.png)

*Figure 4.* Normalized SINR and jammer suppression against the loading level $\delta/\sigma^2$. Small loading recovers the finite-sample loss [5]; loading comparable to the interference eigenvalues drives the weights toward the conventional limit $\mathbf a_0/N$ and removes the jammer nulls, as listed above. Full caption: [Figure 4](07_figure_index.md#figure-4--diagonal-loading-sensitivity).

### C.4 Steering-vector mismatch and self-nulling

Let the beamformer be steered to $\mathbf a_0$ while the target arrives along $\mathbf a_s$. Define the generalized cosine [6]

$$
c^2 = \cos^2(\mathbf a_0,\mathbf a_s;\mathbf R_{in}^{-1}) = \frac{|\mathbf a_0^H\mathbf R_{in}^{-1}\mathbf a_s|^2}{(\mathbf a_0^H\mathbf R_{in}^{-1}\mathbf a_0)(\mathbf a_s^H\mathbf R_{in}^{-1}\mathbf a_s)}.
$$

* MVDR from $\mathbf R_{in}$ (target absent from the covariance): $\mathrm{SINR} = S\,c^2$.
* MPDR, i.e. (B.6) with the full $\mathbf R$ containing the target [6]:
  $$\mathrm{SINR}_{\rm MPDR} = \frac{S\,c^2}{1 + (2S + S^2)(1-c^2)}. \tag{C.3}$$

For $S \gg 1$ even a small mismatch ($1-c^2 \ll 1$) produces a large loss. The beamformer treats the slightly displaced target as interference and partially nulls it. Both expressions are checked to machine precision against the numerically computed weights in `tests/test_beamforming.py`.

![Figure 5 — steering mismatch and self-nulling](../figures/fig05_steering_mismatch.png)

*Figure 5.* SINR loss against pointing error and SNR, comparing MVDR from $\mathbf R_{in}$ ($S c^2$) with MPDR (C.3) [6] and with sample-matrix estimates. Panel (c) shows the pattern notch that MPDR places on a target 1° from the look direction. Full caption: [Figure 5](07_figure_index.md#figure-5--steering-mismatch-sensitivity).

---

## D. Pulsed radar: delay, Doppler, matched filtering, range–Doppler processing

### D.1 Signal model and conventions

Symbols: $c$ speed of light; $f_c$ carrier; $\lambda = c/f_c$; $T$ pulse length; $B$ swept bandwidth; $f_s$ complex sample rate; $T_r = 1/\mathrm{PRF}$ pulse repetition interval; $M$ pulses per coherent processing interval (CPI); $t$ fast time measured from each pulse's transmission; $t_m = mT_r$ slow time.

Radial velocity $v>0$ means **closing**: $R(t) = R_0 - v t$. A monostatic echo has delay $\tau = 2R/c$ and carrier phase $-2\pi f_c\tau(t) = -4\pi R_0/\lambda + 2\pi(2v/\lambda)t$. The Doppler shift is therefore [10]

$$
f_D = \frac{2v}{\lambda} \quad(\text{positive for closing targets}). \tag{D.1}
$$

The baseband echo of pulse $m$ is

$$
x_m(t) = A\,u(t-\tau_m)\,e^{-j4\pi R_0/\lambda}\,e^{\,j2\pi f_D(t_m + t)} + w_m(t),\qquad \tau_m = \frac{2(R_0 - v t_m)}{c}. \tag{D.2}
$$

Assumptions: point target with constant amplitude over the CPI; the $(1-2v/c)$ time-compression of the envelope is neglected (relative error $\sim10^{-6}$); range migration of the envelope across the CPI is kept through $\tau_m$ but is $\le 1.6$ m for the parameters used, far below one range cell; white complex Gaussian receiver noise $w\sim\mathcal{CN}(0,\sigma^2)$; no clutter.

### D.2 LFM waveform

$$
u(t) = e^{\,j\pi\mu(t - T/2)^2},\quad 0\le t<T,\qquad \mu = B/T. \tag{D.3}
$$

The instantaneous frequency $\mu(t-T/2)$ sweeps from $-B/2$ to $+B/2$.

### D.3 Matched filter

For white noise the linear filter that maximizes output SNR at a chosen time is matched to the signal: impulse response $h(t) = u^*(-t)$ [8, 10]. Equivalently, the output is the correlation

$$
y(\tau) = \int x(t)\,u^*(t-\tau)\,dt. \tag{D.4}
$$

Its peak SNR is $E/\sigma^2$, where $E$ is the pulse energy, independent of the waveform shape. In discrete time with a unit-modulus replica of $L = Tf_s$ samples normalized by $\|\mathbf h\| = \sqrt L$, white noise keeps variance $\sigma^2$ and the peak SNR gain is $L$.

The zero-Doppler output of an LFM pulse is the autocorrelation [9]

$$
\frac{|\chi(\tau,0)|}{T} = \Big|\Big(1-\frac{|\tau|}{T}\Big)\,\mathrm{sinc}\Big(B\tau\Big(1-\frac{|\tau|}{T}\Big)\Big)\Big|,\quad |\tau|<T, \tag{D.5}
$$

with $\mathrm{sinc}(x) = \sin(\pi x)/(\pi x)$. The first null is at $B\tau(1-\tau/T) = 1$, i.e. $\tau\approx 1/B$. Hence the nominal range resolution is [10]

$$
\Delta R = \frac{c}{2B}. \tag{D.6}
$$

**Range–Doppler coupling.** A Doppler shift $f_D$ multiplies $u(t)$ by $e^{j2\pi f_D t}$. Over the pulse this equals an LFM delayed by $-f_D/\mu$ (up to a phase and a small amplitude taper), so the matched-filter peak moves to [9]

$$
\tau_{\rm peak} = -\frac{f_D T}{B}. \tag{D.7}
$$

![Figure 7 — LFM matched-filter response and range–Doppler coupling](../figures/fig07_matched_filter.png)

*Figure 7.* Panel (a): sampled matched-filter output against the continuous autocorrelation (D.5) [9]. Panel (b): two equal targets at the resolution (D.6), whose separability depends on their relative phase. Panel (c): measured peak shift against the coupling (D.7) [9]. Full caption: [Figure 7](07_figure_index.md#figure-7--matched-filter-range-response).

### D.4 Pulse-Doppler processing

After matched filtering, each range cell holds a slow-time sequence $y_m \propto e^{j2\pi f_D m T_r}$ (D.2). A length-$K\ge M$ DFT with window $g_m$,

$$
Y[k] = \sum_{m=0}^{M-1} g_m\,y_m\,e^{-j2\pi km/K},
$$

peaks at $k = f_D T_r K$. The Doppler bin spacing is $\mathrm{PRF}/M$ (for $K = M$), the velocity bin spacing is $\lambda\,\mathrm{PRF}/(2M)$, and velocities are unambiguous in $[-\lambda\mathrm{PRF}/4,\ \lambda\mathrm{PRF}/4)$. The window is scaled to $\sum g_m^2 = 1$, so noise keeps variance $\sigma^2$ and the coherent gain is $(\sum g_m)^2$ ($=M$ for the rectangular window) [10, 11].

The *range–Doppler map* is $|Y[k,l]|^2$ over Doppler bin $k$ and range lag $l$. For a target exactly on the grid, the peak SNR is $\mathrm{SNR}_{\rm in}\cdot L\cdot(\sum g)^2$.

![Figure 8 — simulated range–Doppler map](../figures/fig08_range_doppler_map.png)

*Figure 8.* Range–Doppler map of three simulated targets after matched filtering and a Hann-windowed slow-time DFT. Closing targets appear at positive velocity, per the sign convention of (D.1). Full caption: [Figure 8](07_figure_index.md#figure-8--rangedoppler-map).

### D.5 Grid effects (straddle and scalloping)

If the true delay lies a fraction $\epsilon$ of a sample off the grid, the largest sample of $|y|$ is $|\chi(\epsilon/f_s,0)|/T$ (range straddle loss) [10]. If the true Doppler lies $\epsilon$ bins off a bin center, the largest DFT sample is

$$
\frac{\big|\sum_m g_m e^{j2\pi\epsilon m/M}\big|}{\sum_m g_m}, \tag{D.8}
$$

which equals $-3.92$ dB (rectangular) and $-1.42$ dB (Hann) at $\epsilon = 1/2$ [11]. Picking the grid maximum gives an estimation error uniformly distributed in $\pm 1/2$ cell, with RMS $1/\sqrt{12}$ cell. A three-point parabolic fit to $|y|$ around the maximum removes most of this error but leaves a bias that depends on the peak shape.

![Figure 9 — FFT binning, straddle and scalloping](../figures/fig09_binning_effects.png)

*Figure 9.* Grid and interpolated estimation errors, straddle loss from (D.5), scalloping loss from (D.8) [11], and Monte Carlo RMS error against SNR compared with the $1/\sqrt{12}$ quantization level. Full caption: [Figure 9](07_figure_index.md#figure-9--fft-binning-effects).

---

## E. Detection in Gaussian noise and CA-CFAR

### E.1 Square-law detection and the ideal detector

Under $H_0$ a cell holds $x\sim\mathcal{CN}(0,\sigma^2)$, so $z = |x|^2$ is exponential with mean $\sigma^2$. With known $\sigma^2$, the threshold $T$ gives

$$
P_{FA} = e^{-T/\sigma^2}\ \Rightarrow\ T = -\sigma^2\ln P_{FA}. \tag{E.1}
$$

For a non-fluctuating target of amplitude $A$ (Swerling 0) with $\mathrm{SNR} = |A|^2/\sigma^2$, $2z/\sigma^2$ is non-central $\chi^2$ with 2 degrees of freedom and non-centrality $2\,\mathrm{SNR}$:

$$
P_D = Q_1\big(\sqrt{2\,\mathrm{SNR}},\ \sqrt{-2\ln P_{FA}}\big), \tag{E.2}
$$

where $Q_1$ is the Marcum Q-function [14, 17]. For a Rayleigh-fluctuating target (Swerling 1, $A\sim\mathcal{CN}(0,\mathrm{SNR}\,\sigma^2)$), $z$ is exponential with mean $\sigma^2(1+\mathrm{SNR})$:

$$
P_D = P_{FA}^{1/(1+\mathrm{SNR})}. \tag{E.3}
$$

[15]. Albersheim's empirical formula [16] gives an independent check of (E.2).

### E.2 Cell-averaging CFAR

In practice $\sigma^2$ is unknown. CA-CFAR [12] estimates it from $N_t$ training cells around the cell under test (CUT). It excludes $N_g$ guard cells on each side so that the target's own energy does not enter the estimate. The test is

$$
z_{\rm CUT} \gtrless \frac{\alpha}{N_t}\sum_{i=1}^{N_t} z_i. \tag{E.4}
$$

Assume the training cells are i.i.d. exponential and independent of the CUT. Then $Z = \sum z_i \sim \mathrm{Gamma}(N_t,\sigma^2)$, and averaging (E.1) over $Z$ (moment-generating function of the Gamma distribution) gives [10, 12]

$$
P_{FA} = E_Z\big[e^{-\alpha Z/(N_t\sigma^2)}\big] = \Big(1+\frac{\alpha}{N_t}\Big)^{-N_t}
\ \Longrightarrow\ \alpha = N_t\big(P_{FA}^{-1/N_t}-1\big). \tag{E.5}
$$

$\sigma^2$ cancels, which is the CFAR property. As $N_t\to\infty$, $\alpha\to-\ln P_{FA}$ and the detector approaches the ideal one. For finite $N_t$ the threshold estimate is noisy, so a larger $\alpha$ is needed. The extra SNR required to reach the same $P_D$ is the *CFAR loss*.

**Detection probability.** Swerling 1: replace $\sigma^2$ in the CUT by $\sigma^2(1+\mathrm{SNR})$ in (E.5):

$$
P_D = \Big(1 + \frac{\alpha}{N_t(1+\mathrm{SNR})}\Big)^{-N_t}. \tag{E.6}
$$

Swerling 0: no simple closed form. The code evaluates

$$
P_D = \int_0^\infty P\big(z_{\rm CUT} > \alpha Z/N_t \mid Z\big)\,f_{\Gamma(N_t)}(Z)\,dZ \tag{E.7}
$$

by quadrature, with the conditional probability from (E.2).

![Figure 12 — detection probability against SNR](../figures/fig12_pd_vs_snr.png)

*Figure 12.* $P_D$ against SNR at $P_{FA}=10^{-4}$ for the ideal detector, (E.2) [14] and (E.3) [15], and for CA-CFAR, (E.6) and (E.7), with Monte Carlo points. Panel (b) shows the CFAR loss against $N_t$. Full caption: [Figure 12](07_figure_index.md#figure-12--probability-of-detection-versus-snr).

### E.3 Ordered-statistic CFAR

OS-CFAR [13] uses the $k$-th smallest training cell $z_{(k)}$ and the threshold $\alpha z_{(k)}$. For i.i.d. exponential training cells,

$$
P_{FA} = \prod_{i=0}^{k-1}\frac{N_t - i}{N_t - i + \alpha}, \tag{E.8}
$$

which is solved numerically for $\alpha$. With $k<N_t$, up to $N_t-k$ strong interfering cells in the window do not raise the threshold [13]. CA-CFAR has no such protection: a strong target in the training window raises the mean in (E.4) and can mask a weaker neighbor.

![Figure 10 — CA- vs OS-CFAR threshold with a masking neighbor](../figures/fig10_cfar_threshold_example.png)

*Figure 10.* One realization showing the masking mechanism: the CA threshold is raised over the weak target by its stronger neighbor, while the OS threshold is not [13]. The geometry was chosen from a saved 20-seed scan (`results/cfar_masking_seed_scan.json`). Full caption: [Figure 10](07_figure_index.md#figure-10--ca-cfar-threshold-example).

### E.4 What breaks the CA-CFAR assumptions here

(E.5) assumes independent training cells. In the processed range–Doppler map this is not true:

* Matched filtering correlates the noise in neighboring range cells. The complex correlation coefficient between lags $l$ apart equals the normalized replica autocorrelation, which is large at one lag when $f_s = 2B$.
* A non-rectangular slow-time window correlates neighboring Doppler bins. The coefficient is $\sum g_m^2 e^{j2\pi\Delta k\,m/M}$, about 0.67 at $\Delta k = 1$ for Hann.

For complex Gaussian fields the correlation between powers is $|\rho|^2$. A moment-matching estimate replaces $N_t$ in (E.5) by the effective number of independent cells

$$
N_{\rm eff} = \frac{N_t^2}{\sum_{i,j}|\rho_{ij}|^2}, \tag{E.9}
$$

i.e. a Gamma distribution with the same mean and variance as the correlated sum. This is an approximation introduced here to interpret the measurements. It is not an exact result, and Figure 11(b) shows where it falls short.
