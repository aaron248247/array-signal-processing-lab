# 3. Verification and validation

This document records how each component was checked and what the checks showed. Every check is one of four kinds, and each table states which kind it is:

| Kind | Meaning | Pass criterion |
|---|---|---|
| **Exact** | Compared with a closed-form or algebraically equivalent result | Agreement to floating-point precision, or to a stated discretization error that is itself explained |
| **Statistical** | Monte Carlo compared with a theoretical distribution or moment | Theory inside a stated confidence interval, with the interval method justified (§2.7) |
| **Sensitivity** | Numerical behavior under a parameter sweep, compared with a known scaling or limit | Trend and limit agree; no pass/fail threshold claimed beyond that |
| **Exploratory** | Demonstration without an independent reference | No agreement is claimed |

All numbers are read from `results/*.json` produced by `python experiments/run_all.py` (and, for the masking seed scan in §3.8, `python experiments/cfar_masking_seed_scan.py`). Equation labels refer to [01_theory.md](01_theory.md), references to [06_references.md](06_references.md). The automated tests (`pytest`, 114 tests) cover the exact checks and smaller versions of the statistical ones. The experiments run the larger versions.

---

## 3.1 ULA model (exact)

Source: `experiments/ula_verification.py`, `tests/test_array_model.py`; Figure 1.

The numerical response $\mathbf w^H\mathbf a(\theta)$ was compared with the closed form (A.3) on a $0.001^\circ$ grid over $[-90^\circ, 90^\circ]$ (180 001 points). The table also gives the main-lobe peak after refinement, the null locations against (A.4), and the beamwidths.

| Case | max $\lvert AF_{num}-AF_{closed}\rvert$ | peak error (deg) | interior nulls | max null-location error (deg) | max $\lvert AF\rvert$ at analytic nulls | PSL (dB) | HPBW (deg) | $0.886\lambda/(Nd\cos\theta_0)$ (deg) |
|---|---|---|---|---|---|---|---|---|
| N=16, θ₀=0° | 1.5e-15 | 6e-11 | 14 | 1.1e-13 | 3.8e-16 | −13.147 | 6.359 | 6.346 |
| N=16, θ₀=+30° | 1.8e-15 | 2.8e-8 | 14 | 1.3e-13 | 7.9e-16 | −13.147 | 7.349 | 7.327 |
| N=16, θ₀=−45° | 2.1e-15 | 4.9e-8 | 15 | 8.2e-14 | 7.5e-16 | −13.147 | 9.025 | 8.974 |
| N=8, θ₀=0° | 7.7e-16 | 5.1e-8 | 6 | 6.4e-15 | 1.1e-16 | −12.797 | 12.803 | 12.691 |
| N=8, θ₀=+20° | 1.1e-15 | 1.8e-8 | 7 | 1.4e-13 | 3.7e-16 | −12.797 | 13.641 | 13.505 |

Observations:

* The array factor agrees with (A.3) to $\le 2.1\times10^{-15}$ everywhere, which is round-off level for a sum of 16 unit phasors.
* Peak-location errors of $10^{-8}$ deg are at the floor set by the flat top of the main lobe (§2.2). They are not a model error. They are numerical-verification figures for a noiseless pattern and say nothing about angle-estimation accuracy in noise.
* The peak sidelobe level does not depend on steering angle ($-13.147$ dB for $N = 16$ at all three steering angles), as expected because the pattern is a function of $\psi$ only. It approaches the large-$N$ sinc limit of $-13.261$ dB from above as $N$ grows ($N=8$: $-12.797$ dB).
* The half-power beamwidth exceeds the $0.886\lambda/(Nd\cos\theta_0)$ approximation [1] by 0.2–1.0 %. The approximation is a large-$N$, small-angle result, and the excess grows with steering angle and with smaller $N$.
* Endfire zeros ($\sin\theta = \pm1$) are excluded from the "interior" null count but are included in the check that $|AF| = 0$ at every analytic null.

Additional exact checks:

* **Beam scan.** The Bartlett beam scan (A.5) of 4096 simulated noiseless plane-wave snapshots from $30^\circ$ reproduces $p|AF|^2$ to a relative error of $6.8\times10^{-16}$. This is a separate code path (snapshot simulation and sample-covariance quadratic form).
* **Grating lobe.** With $d = \lambda$ steered to $20^\circ$, the response at the predicted grating-lobe direction $\arcsin(\sin 20^\circ - 1) = -41.15^\circ$ is $1.000$.

The unit tests also cover steering-vector shape and unit modulus, the phase progression $\arg(a_{n+1}/a_n) = kd\sin\theta$, $\mathbf a(0) = \mathbf 1$, the conjugate symmetry $\mathbf a(-\theta) = \mathbf a(\theta)^*$, unit look-direction gain and $\|\mathbf w_c\|^2 = 1/N$, and the main-lobe direction to $0.001^\circ$ for several steering angles.

![Figure 1 — ULA analytical verification](../figures/fig01_ula_verification.png)

*Figure 1.* Numerical vs closed-form array factor (a), their difference on the 0.001° grid (b), and the null-location errors in the table above (c). Full caption: [Figure 1](07_figure_index.md#figure-1--ula-analytical-verification).

## 3.2 Covariance and beamforming (exact)

Source: `tests/test_covariance.py`, `tests/test_beamforming.py`, `experiments/jammer_suppression.py`.

| Check | Result |
|---|---|
| $\hat{\mathbf R}$ Hermitian | exactly (symmetrized); rank $=K$ for $K<N$; eigenvalues $\ge -10^{-10}\lambda_{\max}$ |
| $\mathbf R$ from (B.2): diagonal $=\sum p_i+\sigma^2$, $N-r$ eigenvalues $=\sigma^2$ | to $10^{-10}$ |
| $\hat{\mathbf R}\to\mathbf R$ | Frobenius error falls by more than 5× from $K=100$ to $K=10^4$ (expected $10\times$, $K^{-1/2}$) and is below 3 % at $K=10^4$ |
| Same seed gives identical data; a different seed gives different data | yes |
| $\mathbf R = c\mathbf I$ gives $\mathbf w_{\rm MVDR} = \mathbf a_0/N$ (B.6) [3] | to $10^{-14}$ |
| Distortionless constraint $\mathbf w^H\mathbf a_0 = 1$ (sample covariance, loading 0, 1, 100) | to $10^{-10}$ (test tolerance); $\max\lvert20\log_{10}\lvert\mathbf w^H\mathbf a_s\rvert\rvert = 3.9\times10^{-15}$ dB over 8000 SMI trials |
| Clairvoyant MVDR SINR = (B.4) = (B.7) | relative $10^{-10}$; baseline scenario SINR loss $3\times10^{-12}$ dB |
| Conventional SINR = (B.5) | relative $10^{-12}$ |
| MVDR jammer response = (B.8) | relative $10^{-8}$ |
| $\mathbf R_{in}$- and $\mathbf R$-based clairvoyant weights identical (no mismatch) | max difference $4.6\times10^{-13}$ |
| Mismatched MVDR $=S c^2$ and MPDR = (C.3) | relative $10^{-9}$ for three (δ, SNR) pairs |
| Identity $\mathrm{SINR} = x/(1-x)$ used in (C.2) | relative $10^{-12}$ for random $\mathbf w$ |
| Unloaded SMI with $K<N$ | raises `SingularCovarianceError`; loaded version returns finite weights |

**Controlled jammer-suppression case** (test): one jammer at $35^\circ$ with JNR 40 dB, signal-free training with $K = 20N$. The MVDR jammer response is more than 30 dB below the conventional sidelobe, and $\rho\in(0.85, 1]$. The RMB mean [4] is 0.950.

**Note on test design.** The first version of this test put the jammer at $30^\circ$. For $N = 16$, $d=\lambda/2$, $\sin 30^\circ = 4\cdot(2/16)$ is an exact conventional null (A.4). The conventional response was then $-313$ dB, and the "MVDR suppresses more than conventional" assertion failed for a correct reason. The jammer was moved to $35^\circ$ (near a sidelobe peak) and the reason is recorded in the test file.

![Figure 2 — conventional vs MVDR beamforming](../figures/fig02_conventional_vs_mvdr.png)

*Figure 2.* Baseline scenario: conventional and clairvoyant-MVDR patterns (a) and the SINR losses of §3.2–3.3 against the optimum, RMB [4] and (C.2) (b). The clairvoyant null depths (> 100 dB) are idealized numerical values for an exactly known covariance. Full caption: [Figure 2](07_figure_index.md#figure-2--conventional-vs-mvdr-beampattern).

## 3.3 Finite-snapshot behavior (statistical)

Source: `experiments/jammer_suppression.py`, `experiments/snapshot_sensitivity.py`; Figures 2 and 3. Baseline: N = 16, target 0° (SNR 0 dB/element), jammers 35° (40 dB) and −42° (30 dB), $\mathrm{SINR}_{\rm opt} = 12.00$ dB.

**Reed–Mallett–Brennan law (C.1) [4], signal-free training.** Mean normalized SINR $\rho$ over independent training sets:

| K | trials | MC mean ρ | RMB (K−N+2)/(K+1) | z |
|---|---|---|---|---|
| 32 (=2N) | 2000 | 0.5426 ± 0.0019 | 0.5455 | −1.5 |
| 160 (=10N) | 2000 | 0.9072 ± 0.0005 | 0.9068 | +0.7 |

In the 20-point sweep $16\le K\le 640$ (600 trials per point) the largest $|z|$ against the RMB prediction [4] is 1.62.

**Target present in training (C.2).** (C.2) was derived for this project from the RMB distribution [4]; see 01 §C.2.

| K | MC mean ρ (dB) | theory (dB) | z |
|---|---|---|---|
| 160 | −4.231 (0.3775 ± 0.0015) | −4.255 (0.3754) | +1.4 |
| 16 | −21.12 | −20.77 | −2.85 |
| 32 | −11.48 | −11.53 | +0.86 |
| 128 | −5.01 | −4.95 | −1.92 |
| 640 | −1.43 | −1.45 | +1.65 |

The largest deviation is at $K = N$. There $\rho_R\sim\mathrm{Beta}(2, 15)$ and the transformed variable (C.2) is very skewed, so the normal-approximation interval for the mean understates the uncertainty. One $|z| = 2.85$ among 17 comparisons is consistent with chance.

![Figure 3 — finite-snapshot SMI performance](../figures/fig03_sinr_vs_snapshots.png)

*Figure 3.* Monte Carlo means of $\rho$ against the RMB law [4] and (C.2) (a), and the conditioning of $\hat{\mathbf R}$ against the Marchenko–Pastur estimate [18] (b). Full caption: [Figure 3](07_figure_index.md#figure-3--output-sinr-versus-number-of-snapshots).

## 3.4 Pulse compression, range and Doppler (exact and statistical)

Source: `experiments/range_doppler_demo.py`, `tests/test_radar.py`; Figures 7–9.

| Check | Kind | Result |
|---|---|---|
| FFT matched filter [8, 10] vs direct correlation | exact | agree to $10^{-11}$ |
| Integer delay $l_0$ gives peak at sample $l_0$ with amplitude $A\sqrt L$ | exact | for $l_0 = 0, 1, 137, 899$; amplitude to $10^{-12}$ relative |
| Rectangular pulse gives a triangular MF output | exact | to $10^{-12}$ |
| On-grid range and Doppler: peak amplitude $=\sqrt L\sqrt M\cdot$Dirichlet$(f_D)$ (intra-pulse Doppler mismatch) | exact | relative $10^{-9}$ |
| Sampled MF ($f_s = 2B$) vs continuous LFM autocorrelation (D.5) [9], noiseless, fractional delay | exact up to sampling | max deviation 0.0070 of the peak near the main lobe |
| Same, $f_s = 50$ MHz | exact up to sampling | < 0.02 everywhere (test) |
| First null: measured (200 MHz proxy) vs root of $B\tau(1-\tau/T)=1$ | exact | $\tau B = 1.025$ vs $1.021$ (proxy grid step 0.025 in $\tau B$) |
| LFM range–Doppler coupling (D.7) [9], $\lvert f_D\rvert\le 500$ kHz | exact | max deviation 0.015 samples over 41 Doppler values |
| Doppler sign: closing target appears at positive frequency | exact | test |
| White-noise variance after MF | statistical | 0.9996 (4000 pulses × 300 cells) |
| Matched-filter SNR gain $=L$ [8]: input −10 dB, expected output 10.000 (linear) | statistical | 10.039 ± 0.072, $z = 0.5$ |
| Scalloping loss (D.8), half bin | exact | rect −3.9215 dB, Hann −1.4236 dB (cf. 3.92 and 1.42 dB in [11]); FFT vs direct sum differ by $<10^{-13}$ dB |

**Sampled vs continuous autocorrelation.** The 0.007 residual and the 0.085 dB difference in worst-case range straddle loss (−0.912 dB sampled vs −0.946 dB from (D.5)) arise because (D.5) is the continuous-time autocorrelation, while the code correlates a sampled chirp. At $f_s = 2B$ the sampled sidelobes and peak differ slightly. The difference shrinks with oversampling (test at 50 MHz). This is a discretization effect, not a model error.

**Two-target resolution.** Two equal targets separated by $\Delta R$ are resolved or not depending on their relative phase, which in a real scene is set by the carrier phase and is effectively random. Using a 100 MHz-oversampled matched filter:

| ΔR / (c/2B) | in phase | quadrature | anti-phase |
|---|---|---|---|
| 0.75 | 1 peak | 1 peak | 2 peaks, dip −24.5 dB |
| 1.00 | 1 peak | 2 peaks, dip −0.9 dB | 2 peaks, dip −60.6 dB |
| 1.25 | 1 peak | 2 peaks, dip −3.5 dB | 2 peaks, dip −25.1 dB |
| 1.50 | 2 peaks, dip −2.1 dB | 2 peaks, dip −7.3 dB | 2 peaks |
| 2.00 | 2 peaks, dip −27.6 dB | 2 peaks, dip −30.8 dB | 2 peaks |

The nominal resolution $c/2B$ (D.6) [10] matches the quadrature (power-addition) case: the −0.9 dB dip is the familiar Rayleigh-like criterion. In-phase targets need about $1.5\,c/2B$ to be separated. A first version of this check used targets "with the same phase" in `simulate_cpi`. Because the simulator includes the carrier phase $e^{-j4\pi R/\lambda}$, their relative phase was not actually zero. The check was rewritten to set the relative phase explicitly.

![Figure 7 — matched-filter and coupling checks](../figures/fig07_matched_filter.png)

*Figure 7.* Sampled matched-filter output against (D.5) [9] (a), the two-target resolution cases of the table above (b), and the measured coupling shift against (D.7) [9] (c). Full caption: [Figure 7](07_figure_index.md#figure-7--matched-filter-range-response).

**Range and Doppler recovery in the noisy map** (Figure 8; one CPI, Hann window). Target SNRs are per raw sample. For grid-centered targets, processing adds 20 dB (pulse compression) + 15.75 dB (Hann coherent gain).

| Target (R, v) | input SNR | range error, grid / parabolic (m) | velocity error, grid / parabolic (m/s) | peak SNR measured / on-grid expectation (dB) |
|---|---|---|---|---|
| 4321.7 m, +101.3 m/s | −15 dB | −4.7 / −2.2 | +0.19 / −0.30 | 21.6 / 21.3 |
| 6200.0 m, −45.6 m/s | −20 dB | +5.7 / +1.4 | −1.24 / +0.34 | 15.9 / 16.3 |
| 9007.4 m, +12.2 m/s | −25 dB | −13.6 / −11.2 | −4.39 / −3.41 | 12.4 / 11.3 |

All errors are within one resolution cell (29.98 m, 7.81 m/s). The weakest target (≈11 dB) has errors close to one range sample and half a velocity bin, consistent with the Monte Carlo spread at that SNR below.

![Figure 8 — range–Doppler map used for the recovery table](../figures/fig08_range_doppler_map.png)

*Figure 8.* The CPI from which the range and velocity estimates in the table above were taken. Full caption: [Figure 8](07_figure_index.md#figure-8--rangedoppler-map).

**Estimation accuracy vs SNR** (statistical, Figure 9d; 150 trials per point, random sub-cell positions, rectangular window).

| post-processing SNR (dB) | 8.1 | 13.1 | 18.1 | 23.1 | 28.1 |
|---|---|---|---|---|---|
| range RMS, grid (m) | 17.7 | 6.05 | 4.72 | 4.24 | 4.66 |
| range RMS, parabolic (m) | 19.0 | 4.39 | 1.78 | 1.14 | 0.71 |
| velocity RMS, grid (m/s) | 9.44 | 3.59 | 2.17 | 2.36 | 2.29 |
| velocity RMS, parabolic (m/s) | 9.37 | 3.31 | 1.47 | 1.35 | 1.32 |

At high SNR the grid estimator reaches the uniform-quantization floor (4.33 m, 2.25 m/s). The parabolic estimator keeps improving in range. In velocity it saturates at ≈1.3 m/s (0.17 bin), which is its deterministic interpolation bias for the rectangular window (max 0.23 bin, Figure 9b). At 8 dB, occasional peaks are picked from noise, and both estimators are dominated by these outliers.

![Figure 9 — binning, straddle and scalloping checks](../figures/fig09_binning_effects.png)

*Figure 9.* Noiseless grid and interpolation errors (a, b), straddle and scalloping losses against (D.5) and (D.8) [11] (c), and the Monte Carlo RMS errors tabulated above (d). Full caption: [Figure 9](07_figure_index.md#figure-9--fft-binning-effects).

## 3.5 CFAR (exact and statistical)

Source: `experiments/cfar_false_alarm_check.py`, `experiments/cfar_pd_snr.py`, `tests/test_cfar.py`; Figures 10–12.

**Exact checks (tests):**

* The CA-CFAR multiplier $\alpha$ from (E.5) [10, 12] reproduces $P_{FA}$ to $10^{-10}$ for $N_t \in\{4,16,32,128\}$ and $P_{FA}\in\{10^{-2},10^{-4},10^{-6}\}$; $\alpha\to-\ln P_{FA}$ as $N_t\to\infty$.
* OS-CFAR $\alpha$ satisfies (E.8) [13] to $10^{-8}$.
* 1-D and 2-D thresholds match brute-force loops to $10^{-12}$, including wrapped Doppler edges.
* The `valid`, `truncate` and `wrap` edge modes behave as specified at the first and last cells.
* The 2-D detector with zero Doppler extent reduces to the 1-D detector.
* Scaling the data by $10^6$ leaves all decisions unchanged (CFAR property).
* Detection limits: $P_D\to P_{FA}$ as SNR → 0; CA → ideal as $N_t\to\infty$.

**False-alarm rate in i.i.d. exponential noise** (statistical; $2\times10^7$ cells per CA point, $4\times10^6$ per OS point; standard errors from per-sequence spread):

| Detector | requested | measured | s.e. | measured/requested | z | false alarms |
|---|---|---|---|---|---|---|
| CA, N_t=16 | 1e-2 | 9.966e-3 | 2.2e-5 | 0.997 | −1.51 | 198 930 |
| CA, N_t=16 | 1e-3 | 1.016e-3 | 7.1e-6 | 1.016 | +2.24 | 20 278 |
| CA, N_t=16 | 1e-4 | 9.579e-5 | 2.2e-6 | 0.958 | −1.89 | 1 912 |
| CA, N_t=16 | 1e-5 | 9.218e-6 | 6.9e-7 | 0.922 | −1.13 | 184 |
| CA, N_t=32 | 1e-2 | 1.001e-2 | 2.1e-5 | 1.001 | +0.38 | 199 438 |
| CA, N_t=32 | 1e-3 | 1.021e-3 | 7.1e-6 | 1.021 | +2.90 | 20 341 |
| CA, N_t=32 | 1e-4 | 1.012e-4 | 2.3e-6 | 1.012 | +0.53 | 2 017 |
| CA, N_t=32 | 1e-5 | 9.886e-6 | 7.1e-7 | 0.989 | −0.16 | 197 |
| OS, N_t=16, k=12 | 1e-2 | 1.002e-2 | 5.2e-5 | 1.002 | +0.37 | 39 997 |
| OS, N_t=16, k=12 | 1e-3 | 9.97e-4 | 1.6e-5 | 0.997 | −0.19 | 3 980 |
| OS, N_t=16, k=12 | 1e-4 | 1.082e-4 | 5.3e-6 | 1.082 | +1.55 | 432 |

All eleven measurements are within 8 % of the requested value. Ten are within $|z|<2.5$. One configuration (CA, $N_t = 32$, $10^{-3}$) gave $z = +2.90$. An independent re-run with seed 20261009 gave $1.0048\times10^{-3} \pm 7.2\times10^{-6}$ ($z = +0.67$), so a systematic error in the threshold is unlikely. Taken together, $\sum z^2 = 23.6$ for 11 degrees of freedom ($p = 0.014$). This is somewhat larger than expected, mostly because of the $z = 2.90$ point. The re-run, the exactness of (E.5) and the brute-force tests together point to a fluctuation rather than a bias, but the aggregate statistic is reported as measured. The smaller pytest checks (independent windows with exact Clopper–Pearson 99.9 % intervals [19]; sliding windows with $4\sigma$ bounds) pass.

![Figure 11 — CFAR false-alarm verification](../figures/fig11_cfar_pfa_verification.png)

*Figure 11.* Panel (a): the i.i.d.-noise measurements tabulated above, against the nominal CA-CFAR [12] and OS-CFAR [13] values. Panel (b): the processed-noise cases of §3.6. Full caption: [Figure 11](07_figure_index.md#figure-11--measured-versus-requested-false-alarm-probability).

**$P_D$ vs SNR** (statistical; $P_{FA}=10^{-4}$, $N_t=16$, 20 000 windows per point, Clopper–Pearson 95 % [19]). Swerling 0 against quadrature (E.7) and Swerling 1 [15] against (E.6) were compared at 19 SNR values each. All 38 theory values fall inside the 95 % intervals. With 38 points and 95 % intervals, about 2 misses would be typical, so 0 misses is unremarkable (probability ≈ 0.14).

**Ideal-detector cross-check.** Albersheim's empirical formula [16] gives 9.396 dB (Pd = 0.5) and 11.674 dB (Pd = 0.9), against 9.398 dB and 11.749 dB from the Marcum-Q expression (E.2) [14, 17]. The differences, 0.002 dB and 0.075 dB, are within its quoted ≈0.2 dB accuracy.

![Figure 12 — Pd vs SNR verification](../figures/fig12_pd_vs_snr.png)

*Figure 12.* Monte Carlo $P_D$ with Clopper–Pearson intervals [19] against (E.2) [14], (E.3) [15], (E.6) and (E.7) (a), and the CA-CFAR loss against $N_t$ (b). Full caption: [Figure 12](07_figure_index.md#figure-12--probability-of-detection-versus-snr).

## 3.6 Discrepancy investigated: CA-CFAR on processed (correlated) noise

The verification above uses i.i.d. exponential cells, but the CFAR in a radar processor sees noise that has passed through the matched filter and the Doppler FFT. Noise-only CPIs were processed exactly as in the range–Doppler demo and passed to the 2-D CA-CFAR ($N_t = 132$). $P_{FA}$ was measured over full-overlap range lags (60 maps per case; ± values are 95 % intervals):

| Processing | lag-1 field corr., range / Doppler | measured/requested at 1e-3 | at 1e-4 | $N_{\rm eff}$ (E.9) | $N_{\rm eff}$ prediction at 1e-3 / 1e-4 |
|---|---|---|---|---|---|
| i.i.d. exponential (control) | 0 / 0 | 0.977 ± 0.038 | 0.971 ± 0.124 | 132 | 1.00 / 1.00 |
| MF, $f_s = 2B$, rect | 0.64 / 0 | 1.121 ± 0.038 | 1.187 ± 0.119 | 71.0 | 1.16 / 1.30 |
| MF, $f_s = 2B$, Hann | 0.64 / 0.67 | 1.327 ± 0.047 | 1.450 ± 0.148 | 40.6 | 1.45 / 1.90 |
| MF, $f_s = B$, rect | 0.02 / 0 | 0.960 ± 0.045 | 0.981 ± 0.176 | 128.6 | 1.00 / 1.01 |
| MF, $f_s = B$, Hann | 0.02 / 0.67 | 1.081 ± 0.048 | 1.284 ± 0.162 | 73.6 | 1.15 / 1.27 |

Findings:

1. The control and the critically sampled rectangular-window case match the nominal $P_{FA}$, so the 2-D implementation is not the cause.
2. Correlation between training cells raises $P_{FA}$, by up to 45 % at $10^{-4}$ for the $f_s = 2B$ + Hann chain used in the demos. Correlation reduces the effective number of independent cells, which makes the threshold estimate noisier, and (E.5) does not account for this.
3. The moment-matching estimate (E.9) gets the direction and ordering right. It agrees within the intervals for the weakly correlated cases, but it overestimates the inflation in the most correlated case (1.90 predicted vs 1.45 ± 0.15 measured at $10^{-4}$). A two-moment Gamma approximation is too crude for the tail at low $P_{FA}$.
4. Possible remedies, not implemented: calibrate $\alpha$ by Monte Carlo for the actual processing chain; decimate the training cells to every other range lag; or sample at $f_s\approx B$.

**Partial-overlap lags** (the last $L$ range lags, where the matched-filter output contains less noise energy) were measured separately with `truncate` edges. The ratios are 1.05 ± 0.17, 1.85 ± 0.62, 1.22 ± 0.15, 1.15 ± 0.37 ($f_s=2B$, rect and Hann, at $10^{-3}$ and $10^{-4}$) and 0.96 ± 0.16, 0.73 ± 0.40, 1.00 ± 0.17, 0.89 ± 0.44 ($f_s = B$). The noise level ramps down across this region, so the training cells there are not identically distributed. Some of the inflation at $f_s = 2B$ may come from this, but the intervals are wide. This region is excluded from all other statistics.

## 3.7 Parameter-sensitivity studies (sensitivity)

These are sweeps rather than pass/fail checks. They are compared with limits and closed forms wherever one exists.

| Study | Reference used | Agreement |
|---|---|---|
| SINR vs K (Fig. 3a) | RMB (C.1), (C.2) | §3.3 |
| cond($\hat{\mathbf R}$) vs K (Fig. 3b) | $\lambda_1/(1-\sqrt{(N-r)/K})^2$ (Marchenko–Pastur edge [18]) and cond($\mathbf R_{in}$) = 1.6e5 | ratio of median to estimate 1.42 at $K=N$, 0.72–1.05 for $17\le K\le640$, approaching 1 at large $K$; median tends toward cond($\mathbf R_{in}$) |
| Loading sweep (Fig. 4) | diagonal-loading limits [5, 7]: $\delta\to0$: unloaded SMI; $\delta\gg\lambda_1$: conventional (−29.99 dB) | at −30 dB loading: −2.64 dB (K=2N, signal-free; RMB −2.63 dB); at +60 dB: −28.8 dB, still converging toward −29.99 dB |
| Mismatch sweep (Fig. 5) | $S c^2$ and (C.3) [6] | closed forms exact (tests). SMI (K = 10N, target in training) is within 1 dB of MPDR for $\lvert\delta\rvert\ge 1^\circ$. Near $\delta = 0$ it shows the additional finite-sample loss of (C.2) (−4.3 dB at δ = 0), which the clairvoyant MPDR does not have. In the SNR sweep the difference is −0.95…+1.95 dB, and SMI is less deeply self-nulled than MPDR at SNR ≥ 15 dB |
| Jammer-angle sweep (Fig. 6a) | (B.5), (B.7) | Monte Carlo SMI (K = 2N) is 2.52–2.75 dB below (B.7) at all 45 sampled jammer angles, including inside the main lobe (RMB: −2.63 dB, independent of the interference) |
| JNR sweep (Fig. 6c) | RMB is independent of JNR [4] | SMI K = 2N at 9.29–9.45 dB for JNR −10…60 dB, i.e. 2.56–2.72 dB below optimum; cond($\mathbf R_{in}$) from 2.6 to 1.6e7 |
| FFT binning (Fig. 9a–c) | uniform quantization, (D.5), (D.8) [11] | grid error max 0.5 cell, RMS 4.28 m vs 4.33 m ($\Delta/\sqrt{12}$); scalloping exact |


![Figure 4 — diagonal-loading sweep](../figures/fig04_loading_sensitivity.png)

*Figure 4.* Loading sweep summarized in the table: both limits are reached, and the plateau between them depends on the scenario. Full caption: [Figure 4](07_figure_index.md#figure-4--diagonal-loading-sensitivity).

![Figure 5 — steering-mismatch sweep](../figures/fig05_steering_mismatch.png)

*Figure 5.* Closed-form $S c^2$ and MPDR (C.3) [6] curves with the sample-matrix points compared in the table. Full caption: [Figure 5](07_figure_index.md#figure-5--steering-mismatch-sensitivity).

![Figure 6 — jammer-angle and JNR dependence](../figures/fig06_jammer_angle_jnr.png)

*Figure 6.* Closed-form optimum (B.7) and conventional (B.5) SINR against jammer angle (a) and JNR (c), with Monte Carlo SMI points offset from the optimum by the RMB loss [4]; panel (b) shows the MVDR beam-pointing offset. Full caption: [Figure 6](07_figure_index.md#figure-6--jammer-angle-and-jnr).

## 3.8 Exploratory results (no agreement claimed)

* **CFAR masking example** (Figure 10). A −15 dB target 9 range cells from a −10 dB target in the same Doppler bin is missed by CA-CFAR (threshold 27.6 dB above noise vs 22.0 dB cell power) and detected by OS-CFAR (threshold 19.5 dB). One cell at 6.5 km is a false alarm for both detectors. This is a single realization and shows a known mechanism [13]; it is not a performance statistic. The geometry was chosen from a 20-seed scan over six candidate geometries, saved in `results/cfar_masking_seed_scan.json`. For the selected geometry CA-CFAR detected the weak target in 0/20 realizations and OS-CFAR in 20/20.
* **Integrated array + radar case** (Figure 13). One realization. The beam-output noise floors agree with $\mathbf w^H\mathbf R_{in}\mathbf w$ (conventional: 17.4 dB measured vs 17.7 dB; MVDR: −12.1 dB vs −11.9 dB). The false-alarm counts (10 and 14 cells, forming 7 and 12 clusters) are compared with an expectation of 8.2 that includes the measured inflation of §3.6. One map is not enough to test this.

![Figure 10 — CFAR masking example](../figures/fig10_cfar_threshold_example.png)

*Figure 10.* The single masking realization described above; the CA threshold over the weak target is raised by its neighbor, the OS threshold is not [13]. Full caption: [Figure 10](07_figure_index.md#figure-10--ca-cfar-threshold-example).

![Figure 13 — integrated array + radar example](../figures/fig13_integrated_array_radar.png)

*Figure 13.* The single integrated realization described above, after conventional (a) and MVDR (b) beamforming. Full caption: [Figure 13](07_figure_index.md#figure-13--integrated-array--radar-case).

## 3.9 Reproducibility

* All experiments use fixed seeds stored in `results/*.json`.
* `python experiments/run_all.py` was run twice in succession. `python tools/compare_results.py <first copy> results` reported 10 result files compared, 0 differences at `rtol = 0` (metrics and configuration bitwise identical). This holds for the same NumPy/SciPy versions and platform (recorded in each JSON). Different BLAS builds may change the last digits.
* `pytest` checks determinism directly (identical snapshots, weights, CFAR decisions and radar data for identical seeds).

## 3.10 Test inventory

| File | Tests | Scope |
|---|---|---|
| `tests/test_array_model.py` | 39 | geometry, steering vector, closed-form AF, main lobe, nulls, symmetry, beam scan, grating-lobe flag |
| `tests/test_covariance.py` | 9 | complex Gaussian generator, covariance structure, Hermitian/PSD, rank, convergence, seeds, loading |
| `tests/test_beamforming.py` | 17 | normalization, MVDR identities, distortionless constraint, closed-form SINR and null depth, jammer suppression, singular guard, RMB, mismatch formulas |
| `tests/test_radar.py` | 20 | parameters, LFM, FFT vs direct MF, delay recovery, noise variance, closed-form autocorrelation, range/Doppler recovery, Doppler sign, coherent gain, coupling, scalloping, windows, seeds |
| `tests/test_cfar.py` | 28 | thresholds, edges, brute-force 1-D/2-D/OS, CFAR property, seeds, Monte Carlo $P_{FA}$ (independent and sliding windows), Pd closed forms and limits, Swerling-1 MC, binomial interval |
| `tests/test_saved_results.py` | 1 | integrity of the stored CFAR masking seed scan (seed list, one row per seed, aggregates consistent with per-seed rows and the NPZ) |
